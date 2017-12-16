#############################################################################
#Copyright 2017-2018, Anh Tuan Tran, Tal Hassner, Iacopo Masi, Eran Paz, Yuval Nirkin, and Gerard Medioni
#The SOFTWARE provided in this page is provided "as is", without any guarantee
#made as to its suitability or fitness for any particular use. It may contain
#bugs, so use of this tool is at your own risk. We take no responsibility for
#any damage of any sort that may unintentionally be caused through its use.
# Please, cite the paper:
# @article{tran16_3dmm_cnn,
#   title={Extreme {3D} Face Reconstruction: Looking Past Occlusions},
#   author={Anh Tran 
#       and Tal Hassner 
#       and Iacopo Masi
#       and Eran Paz
#       and Yuval Nirkin
#       and G\'{e}rard Medioni}
#   journal={arXiv preprint},
#   year={2017}
# }
# if you find our code useful.
##############################################################################
import os
## To suppress the noise output of Caffe when loading a model
## polish the output (see http://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe)
os.environ['GLOG_minloglevel'] = '2' 
###################
import numpy as np
from PIL import Image
from glob import glob
import caffe
import cv2
import time
import ntpath
import os.path
import scipy.io
import shutil
import sys
from skimage import io
import dlib
import utils
import bumpMapRegressor
# --------------------------------------------------------------------------- #
# Usage: python testBatchModel.py <inputList> <outputDir> [<landmarkDir>]
# --------------------------------------------------------------------------- #
# CNN network spec
deploy_path = '../CNN/deploy_network.prototxt'
model_path  = '../CNN/3dmm_cnn_resnet_101.caffemodel'
mean_path = '../CNN/mean.binaryproto'
layer_name      = 'fc_ftnew'
bumpModel_path = '../CNN/ckpt_109_grad.pth.tar'
#GPU ID we want to use
GPU_ID = 0	
## Modifed Basel Face Model
#BFM_path = '../3DMM_model/BaselFaceModel_mod.mat'
## CNN template size
trg_size = 224
#### Initiate ################################
predictor_path = "../dlib_model/shape_predictor_68_face_landmarks.dat"
if len(sys.argv) < 3 or len(sys.argv) > 4:
		print("Usage: python testBatchModel.py <inputList> <outputDir> [<landmarkDir>]")
		exit(1)
fileList = sys.argv[1]
data_out = sys.argv[2]
landmarkDir = ''
if len(sys.argv) > 3:
	landmarkDir = sys.argv[3]

if not os.path.exists(data_out):
	os.makedirs(data_out)
if not os.path.exists(data_out + "/imgs"):
	os.makedirs(data_out + "/imgs")
if not os.path.exists(data_out + "/shape"):
	os.makedirs(data_out + "/shape")
if not os.path.exists(data_out + "/bump"):
	os.makedirs(data_out + "/bump")
if not os.path.exists(data_out + "/3D"):
	os.makedirs(data_out + "/3D")

if len(landmarkDir) == 0:
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)

###################################################
##### Prepare images ##############################
countIms = 0
with open(fileList, "r") as ins, open(data_out + "/imList.txt","w") as outs:
	for image_path in ins:
		if len(image_path) < 6:
			print('Skipping ' + image_path + ' file path too short')
			continue
	image_path = image_path[:-1]
	print("> Prepare image "+image_path + ":")
	imname = ntpath.basename(image_path)
	#imname = imname[:-4]
	imname = imname.split(imname.split('.')[-1])[0][0:-1]
	img = cv2.imread(image_path)
	## If we have input landmarks
	if len(landmarkDir) > 0:
		lms = np.loadtxt(landmarkDir + '/' + imname + '.pts')
		img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
		nLM = lms.shape[0]
		for i in range(0,nLM):
			cv2.circle(img2, (lms[i,0], lms[i,1]), 5, (255,0,0))
		img, lms = utils.cropByInputLM(img, lms, img2)
	else:
		dlib_img = io.imread(image_path)
		img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
		dets = detector(img, 1)
		print(">     Number of faces detected: {}".format(len(dets)))
		if len(dets) == 0:
			print('> Could not detect the face, skipping the image...' + image_path)
			continue
		if len(dets) > 1:
			print("> Process only the first detected face!")
		detected_face = dets[0]
		## If we are using landmarks to crop
		shape = predictor(dlib_img, detected_face)
		nLM = shape.num_parts
		for i in range(0,nLM):
			cv2.circle(img2, (shape.part(i).x, shape.part(i).y), 5, (255,0,0))
		img, lms = utils.cropByLM(img, shape, img2)
	cv2.imwrite(data_out + "/imgs/"+imname+"_detect.png",img2)

	lms = lms * 500.0/img.shape[0]
	fileout = open(data_out + "/imgs/"+imname + ".pts","w")
	for i in range(0,lms.shape[0]):
		fileout.write("%f %f\n" % (lms[i,0], lms[i,1]))
	fileout.close()
	img = cv2.resize(img,(500, 500))
	cv2.imwrite(data_out + "/imgs/"+imname+ ".png",img)
	outs.write("%s\n" % (data_out + "/imgs/"+imname+ ".png"))
	countIms = countIms + 1

##################################################
##### Shape fitting ############################## 
# load net
try: 
	caffe.set_mode_gpu()
	caffe.set_device(GPU_ID)
	pass
except Exception as ex:
	print('> Could not setup Caffe in GPU ' +str(GPU_ID) + ' - Error: ' + ex)
	print('> Reverting into CPU mode')
	caffe.set_mode_cpu()
## Opening mean average image
proto_data = open(mean_path, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
## Loading the CNN
net = caffe.Classifier(deploy_path, model_path)
## Setting up the right transformer for an input image
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformer.set_mean('data',mean)
print('> CNN Model loaded to regress 3D Shape and Texture!')
## For loop over the input images
count = 0
with open(data_out + "/imList.txt", "r") as ins:
   for image_path in ins:
	if len(image_path) < 3:
		continue
	image_path = image_path[:-1]
	count = count + 1
	fig_name = ntpath.basename(image_path)
	outFile = data_out + "/shape/" + fig_name[:-4]
	print('> Processing image: ', image_path, ' ', fig_name, ' ', str(count) + '/' + str(countIms))
	net.blobs['data'].reshape(1,3,trg_size,trg_size)
	im = caffe.io.load_image(image_path)
	## Transforming the image into the right format
	net.blobs['data'].data[...] = transformer.preprocess('data', im)
	## Forward pass into the CNN
	net_output = net.forward()
	## Getting the output
	features = np.hstack( [net.blobs[layer_name].data[0].flatten()] )
	## Writing the regressed 3DMM parameters
	np.savetxt(outFile + '.ply.alpha', features[0:99])

##################################################
##### Bump map regression ########################
print("Regress bump maps")
bumpMapRegressor.estimateBump(bumpModel_path, data_out + "/imList.txt", data_out + "/bump/")
##################################################
##### Recover the 3D models ##################
print("Recover the 3D models")
print("cd ../bin; ./TestBump -batch " + data_out + "/imList.txt " + data_out + "/3D/ " + data_out + "/shape " + data_out + "/bump " + data_out + "/bump ../3DMM_model/BaselFaceModel_mod.h5 ../dlib_model/shape_predictor_68_face_landmarks.dat " + data_out + "/imgs; cd ../demoCode");
os.system("cd ../bin; ./TestBump -batch " + data_out + "/imList.txt " + data_out + "/3D/ " + data_out + "/shape " + data_out + "/bump " + data_out + "/bump ../3DMM_model/BaselFaceModel_mod.h5 ../dlib_model/shape_predictor_68_face_landmarks.dat " + data_out + "/imgs; cd ../demoCode");

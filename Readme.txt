## 3rd-party libraries:
  The same as 3DMM-CNN code + PyTorch

## Data requirements
Before running the code, please, make sure to have all the required data in the following specific folder:
- **[Download our Bump-CNN](https://drive.google.com/open?id=1uuDWbTo9hn96Hn_DcKLdopYyUm6oAeVl)** and move the CNN model (1 file: `ckpt_109_grad.pth.tar`) into the `CNN` folder
- **[Download our CNN](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)** and move the CNN model (3 files: `3dmm_cnn_resnet_101.caffemodel`,`deploy_network.prototxt`,`mean.binaryproto`) into the `CNN` folder
- **[Download the Basel Face Model](http://faces.cs.unibas.ch/bfm/main.php?nav=1-2&id=downloads)** and move `01_MorphableModel.mat` into the `3DMM_model` folder
- **[Acquire 3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip)**, run its code to generate `Model_Expression.mat` and move this file the `3DMM_model` folder
- Go into `3DMM_model` folder. Run the script `python trimBaselFace.py`. This should output 2 files `BaselFaceModel_mod.mat` and `BaselFaceModel_mod.h5`.
Note that I have modified the model file from the 3DMM-CNN paper. Therefore, if you generated these files before, you need to re-create them.

- **[Download dlib face prediction model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)** and move the `.dat` file into the `dlib_model` folder.


## Build C++: The same way as the pose+expression estimation code
Usage: 
* Single input:  
TestBump <imPath> <out prefix> <input 3D alpha> <input bump map> <input segmentation map> <BaselFaceModel_mod.h5 path> <dlib path> [<LM path> [<symmetry flag>]]
* Batch process:
TestBump -batch <imList> <out prefix> <input alpha folder> <input bump folder> <input segmentation folder> <BaselFaceModel_mod.h5 path> <dlib path> [<LM folder> [<symmetry flag>]]

- In the first case, the output 3D models will be "<out prefix>_<postfix>.ply"
  In the second case, the output 3D models will be "<out prefix>/<image name>_<postfix>.ply"
  <postfix> = <modelType>_<poseType>
  <modelType> can be "foundation", "withBump" (before soft-symmetry),"sparseFull" (soft-symmetry on the sparse mesh), and "final"
  <poseType> can be "frontal" or "aligned" (based on the estimated pose)
- <LM folder>: path to the precomputed landmark folder (optional)
  By default, detect landmark with Dlib
- <symmetry flag>: 0/1 (default 1)
  This indicates whether soft-symmetry is performed. Soft-symmetry is time-comsuming but it provides the complete 3D models.

## Test: go to demoCode folder
* 3D modeling from images from scratch (no occlusion filling):
python testBatchModel.py testImages.txt ../output/
* 3D modeling results in our paper (using processed bump maps):
./testPaperResults.sh



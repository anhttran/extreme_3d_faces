# Extreme 3D Face Reconstruction: Seeing Through Occlusions

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/extreme_3d_teaser.png)

**Please note that the main part of the code has been released, though we are still testing it to fix possible glitches. Thank you.**

Python and C++ code for realistic 3D face modeling from single image using **[our shape and detail regression networks](https://arxiv.org/abs/1712.05083)** published in CVPR 2018 [1] (follow the link to our PDF which has many, **many** more reconstruction results.)

This page contains end-to-end demo code that estimates the 3D facial shape with realistic details directly from an unconstrained 2D face image. For a given input image, it produces standard ply files of the 3D face shape. It accompanies the deep networks described in our paper [1] and [2]. The occlusion recovery code, however, will be published in a future release. We also include demo code and data presented in [1].

## Dependencies

## Library requirements

* [Dlib Python and C++ library](http://dlib.net/)
* [OpenCV Python and C++ library](http://opencv.org/)
* [Caffe](caffe.berkeleyvision.org) (**version 1.0.0-rc3 or above required**)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)
* [PyTorch](http://pytorch.org/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. A bit more effort is required to install caffé, dlib, and libhdf5.

## Data requirements

Before running the code, please, make sure to have all the required data in the following specific folder:
- **[Download our Bump-CNN](https://docs.google.com/forms/d/11zprdPz9DaBiOJakMixis1vylHps7yn8XcSw72fecGo)** and move the CNN model (1 file: `ckpt_109_grad.pth.tar`) into the `CNN` folder
- **[Download our CNN](http://www.openu.ac.il/home/hassner/projects/CNN3DMM)** and move the CNN model (3 files: `3dmm_cnn_resnet_101.caffemodel`,`deploy_network.prototxt`,`mean.binaryproto`) into the `CNN` folder
- **[Download the Basel Face Model](http://faces.cs.unibas.ch/bfm/main.php?nav=1-2&id=downloads)** and move `01_MorphableModel.mat` into the `3DMM_model` folder
- **[Acquire 3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip)**, run its code to generate `Model_Expression.mat` and move this file the `3DMM_model` folder
- Go into `3DMM_model` folder. Run the script `python trimBaselFace.py`. This should output 2 files `BaselFaceModel_mod.mat` and `BaselFaceModel_mod.h5`.
- **[Download dlib face prediction model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)** and move the `.dat` file into the `dlib_model` folder.

Note that we modified the model files from the 3DMM-CNN paper. Therefore, if you generated these files before, you need to re-create them for this code.

## Installation (C++ code)

- Install **cmake**: 
```
	apt-get install cmake
```
- Install **opencv** (2.4.6 or higher is recommended):
```
	(http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html)
```
- Install **libboost** (1.5 or higher is recommended):
```
	apt-get install libboost-all-dev
```
- Install **OpenGL, freeglut, and glew**
```
	sudo apt-get install freeglut3-dev
	sudo apt-get install libglew-dev
```
- Install **libhdf5-dev** library
```
	sudo apt-get install libhdf5-dev
```
- Install **Dlib C++ library** Dlib should be compiled to shared objects. Check the comments in its CMakeList.txt.
```
	(http://dlib.net/)
```
- Update Dlib directory paths (`DLIB_INCLUDE_DIR` and `DLIB_LIB_DIR`) in `CMakeLists.txt`
- Make build directory (temporary). Make & install to bin folder
```
	mkdir build
	cd build
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../bin ..
	make
	make install
```
  This code should generate `TestBump` in `bin` folder

## Usage

### 3D face modeling with realistic details from a set of input images
* Go into `demoCode` folder. The demo script can be used from the command line with the following syntax:

```bash
$ Usage: python testBatchModel.py <inputList> <outputDir>
```

where the parameters are the following:
- `<inputList>` is a text file containing the paths to each of the input images, one in each line.
- `<outputDir>` is the path to the output directory, where ply files are stored.

An example for `<inputList>` is `demoCode/testImages.txt`
<pre>
../data/test/03f245cb652c103e1928b1b27028fadd--smith-glasses-too-faced.jpg
../data/test/20140420_011855_News1-Apr-25.jpg
....
</pre>

The output 3D models will be `<outputDir>/<imageName>_<postfix>.ply` with `<postfix>` = `<modelType>_<poseType>`. `<modelType>` can be `"foundation"`, `"withBump"` (before soft-symmetry),`"sparseFull"` (soft-symmetry on the sparse mesh), and `"final"`. `<poseType>` can be `"frontal"` or `"aligned"` (based on the estimated pose).
The final 3D shape has `<postfix>` as `"final_frontal"`.

The PLY files can be displayed using standard off-the-shelf 3D (ply file) visualization software such as [MeshLab](http://meshlab.sourceforge.net).

Note that our occlusion recovery code is not included in this release.

### Demo code and data in our paper
* Go into `demoCode` folder. The demo script can be used from the command line with the following syntax:

```bash
$ Usage: ./testPaperResults.sh
```

## Citation

If you find this work useful, please cite our paper [1] with the following bibtex:

```latex
@inproceedings{tran2017extreme,
  title={Extreme {3D} Face Reconstruction: Seeing Through Occlusions},
  author={Tran, Anh Tuan and Hassner, Tal and Masi, Iacopo and Paz, Eran and Nirkin, Yuval and Medioni, G\'{e}rard},
  booktitle={IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year=2018
}
```

## References

[1] A. Tran, T. Hassner, I. Masi, E. Paz, Y. Nirkin, G. Medioni, "[Extreme 3D Face Reconstruction: Seeing Through Occlusions](https://arxiv.org/abs/1712.05083)", IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, June 2018 

[2] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tran_Regressing_Robust_and_CVPR_2017_paper.pdf)", CVPR 2017 

## Changelog
- Dec. 2017, First Release 

## License and Disclaimer
Please, see [the LICENSE here](LICENSE.txt)

## Contacts

If you have any questions, drop an email to _anhttran@usc.edu_ , _hassner@isi.edu_ and _iacopoma@usc.edu_  or leave a message below with GitHub (log-in is needed).

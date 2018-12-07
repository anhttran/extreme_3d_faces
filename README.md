# Extreme 3D Face Reconstruction: Seeing Through Occlusions

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/extreme_3d_teaser.png)

**Please note that the main part of the code has been released, though we are still testing it to fix possible glitches. Thank you.**

Python and C++ code for realistic 3D face modeling from single image using **[our shape and detail regression networks](https://arxiv.org/abs/1712.05083)** published in CVPR 2018 [1] (follow the link to our PDF which has many, **many** more reconstruction results.)

This page contains end-to-end demo code that estimates the 3D facial shape with realistic details directly from an unconstrained 2D face image. For a given input image, it produces standard ply files of the 3D face shape. It accompanies the deep networks described in our paper [1] and [2]. The occlusion recovery code, however, will be published in a future release. We also include demo code and data presented in [1].

## Dependencies

## Data requirements

Before compiling the code, please, make sure to have all the required data in the following specific folder:
- **[Download our Bump-CNN](https://docs.google.com/forms/d/11zprdPz9DaBiOJakMixis1vylHps7yn8XcSw72fecGo)** and move the CNN model (1 file: `ckpt_109_grad.pth.tar`) into the `CNN` folder
- **[Download our PyTorch CNN model](https://docs.google.com/forms/d/e/1FAIpQLSd6cwKh-CO_8Yr-VeDi27GPswyqI9Lvub6S2UYBRsLooCq9Vw/viewform)** and move the CNN model (3 files: `shape_model.pth`,`shape_model.py`,`shape_mean.npz`) into the `CNN` folder
- **[Download the Basel Face Model](http://faces.cs.unibas.ch/bfm/main.php?nav=1-2&id=downloads)** and move `01_MorphableModel.mat` into the `3DMM_model` folder
- **[Acquire 3DDFA Expression Model](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Code/3DDFA.zip)**, run its code to generate `Model_Expression.mat` and move this file the `3DMM_model` folder
- Go into `3DMM_model` folder. Run the script `python trimBaselFace.py`. This should output 2 files `BaselFaceModel_mod.mat` and `BaselFaceModel_mod.h5`.
- **[Download dlib face prediction model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)** and move the `.dat` file into the `dlib_model` folder.

Note that we modified the model files from the 3DMM-CNN paper. Therefore, if you generated these files before, you need to re-create them for this code.

## Installation

There are 2 options below to compile our code:

### Installation with Docker (recommended)

- Install [Docker CE](https://docs.docker.com/install/)
- With Linux, [manage Docker as non-root user](https://docs.docker.com/install/linux/linux-postinstall/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Build docker image:
```
	docker build -t extreme-3dmm-docker .
```
### Installation without Docker on Linux

The steps below have been tested on Ubuntu Linux only:

- Install [Python2.7](https://www.python.org/download/releases/2.7/)
- Install the required third-party packages:
```
	sudo apt-get install -y libhdf5-serial-dev libboost-all-dev cmake libosmesa6-dev freeglut3-dev
```
- Install [Dlib C++ library](http://dlib.net/). Sample code to comiple Dlib:
```
	wget http://dlib.net/files/dlib-19.6.tar.bz2
	tar xvf dlib-19.6.tar.bz2
	cd dlib-19.6/
	mkdir build
	cd build
	cmake ..
	cmake --build . --config Release
	sudo make install
	cd ..
```
- Install [PyTorch](http://pytorch.org/)
- Install other required third-party Python packages:
```
	pip install opencv-python torchvision scikit-image cvbase pandas mmdnn dlib
```
- Config Dlib and HDF5 path in CMakefiles.txt, if needed
- Build C++ code
```
	mkdir build;
	cd build; \
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../demoCode ..;
	make;
	make install;
	cd ..
```
This code should generate `TestBump` in `demoCode` folder

## Usage

### Start docker container
If you compile our code with Docker, you need to start a Docker container to run our code. You also need to set up a shared folder to transfer input/output data between the host computer and the container.
- Prepare the shared folder on the host computer. For example, `/home/ubuntu/shared`
- Copy input data (if needed) to the shared folder
- Start container:
```
	nvidia-docker run --rm -ti --ipc=host --privileged -v /home/ubuntu/shared:/shared extreme-3dmm-docker bash
```
Now folder `/home/ubuntu/shared` on your host computer will be mounted to folder `/shared` inside the container

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
The final 3D shape has `<postfix>` as `"final_frontal"`. You can config the output models in code before compiling.

The PLY files can be displayed using standard off-the-shelf 3D (ply file) visualization software such as [MeshLab](http://meshlab.sourceforge.net).

Sample command:
```bash
	python testBatchModel.py testImages.txt /shared
```

Note that our occlusion recovery code is not included in this release.

### Demo code and data in our paper
* Go into `demoCode` folder. The demo script can be used from the command line with the following syntax:

```bash
$ Usage: ./testPaperResults.sh
```

Before exiting the docker container, remember to save your output data to the shared folder.

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

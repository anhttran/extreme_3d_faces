#include <fstream>

#include "cv.h"
#include "highgui.h"

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include "FaceServices2.h"
#include <sstream>
#include "DlibWrapper.h"
#include <string>
#include <iostream>
#include "H5Cpp.h"
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp> 

using namespace std;
using namespace cv;

// Load landmark from file (68x2)
cv::Mat loadLM(const char* LMfile){
	ifstream in_stream(LMfile);
	if (!in_stream.is_open()) {
		return cv::Mat();
	}
	std::vector<float> vals;
    	string line;
	while (!in_stream.eof())
	{
		line.clear();
		std::getline(in_stream, line);
		if (line.size() == 0 || line.at(0) == '#')
			continue;
	    	std::istringstream iss(line);
	    	float x, y;
	    	if (!(iss >> x >> y)) { continue; }
		vals.push_back(x);
		vals.push_back(y);
	}
	int N = vals.size()/2;
	cv::Mat lms(vals.size()/2,2,CV_32F);
	for (int i=0;i<N;i++){
		lms.at<float>(i,0) = vals[2*i];
		lms.at<float>(i,1) = vals[2*i+1];
	}
	return lms;
}

// Load shape parameters from file (99x1)
cv::Mat loadWeight(const char* inputFile){
	ifstream in_stream(inputFile);
	if (!in_stream.is_open()) {
		return cv::Mat();
	}
	std::vector<float> vals;
    	string line;
	while (!in_stream.eof())
	{
		line.clear();
		std::getline(in_stream, line);
	    	std::istringstream iss(line);
	    	double w;
	    	if (!(iss >> w)) { continue; }
		vals.push_back(w);
	}
	int N = vals.size();
	cv::Mat out(N,1,CV_32F);
	for (int i=0;i<N;i++){
		out.at<float>(i,0) = vals[i];
	}
	return out;
}

// Get cropped image:
//     Inputs:
//	  oriIm   : original image
//	  oriLMs  : original landmarks (68x2)
//     Outputs:
//	  newLMs  : landmarks of the cropped image
//	  return the cropped image
cv::Mat getCroppedIm(Mat& oriIm, cv::Mat &oriLMs, cv::Mat &newLMs)
{	
	float padding = 1.7;			
	newLMs = oriLMs.clone();
	Mat xs = newLMs(Rect(0, 0, 1, oriLMs.rows));
	Mat ys = newLMs(Rect(1, 0, 1, oriLMs.rows));
	
	// get LM-tight bounding box
	double min_x, max_x, min_y, max_y;
	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);
	float width = max_x - min_x;
	float height = max_y - min_y;

        if (width < 5 || height < 5 || width*height < 100) {
		std::cout << "-> Error: Input face is too small" << std::endl;
                return cv::Mat();
        }

	// expand bounding box
	int minCropX = max((int)(min_x-padding*width/3.0),0);
	int minCropY = max((int)(min_y-padding*height/3.0),0);

	int widthCrop = min((int)(width*(3+2*padding)/3.0f), oriIm.cols - minCropX - 1);
	int heightCrop = min((int)(height*(3+2*padding)/3.0f), oriIm.rows - minCropY - 1);

	if(widthCrop <= 0 || heightCrop <=0) return cv::Mat();

	// normalize image size to get a stable pose estimation, assuming focal length 1000
	double minRes = 250*250 * (3+2*padding)/5.0f;
	double maxRes = 300*300 * (3+2*padding)/5.0f;
	
	double scaling = 1;
	if (widthCrop*heightCrop < minRes) scaling = std::sqrt(widthCrop*heightCrop/minRes);
	else if (widthCrop*heightCrop > maxRes) scaling = std::sqrt(widthCrop*heightCrop/maxRes);

	// first crop the image
	cv::Mat display_image = oriIm(Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));

	// now scale it
	if (scaling != 1)
		cv::resize(display_image.clone(), display_image, Size(), 1/scaling, 1/scaling);

	int nrows = display_image.rows/4 * 4;
	int ncols = display_image.cols/4 * 4;
	if (nrows != display_image.rows || ncols != display_image.cols){
		display_image = display_image(Rect(0,0,ncols,nrows)).clone();
	}
	xs = (xs - minCropX)/scaling;
	ys = (ys - minCropY)/scaling;
	return display_image;	
}

int main(int argc, char** argv)
{
    char text[200];
    char out_prefix[200];
    cv::Mat lms;

    OutputSettings outSet;
    outSet.foundationFrontal = outSet.foundationAligned = outSet.withBumpAligned = outSet.finalFrontal = true;
    outSet.sparseFullFrontal = outSet.withBumpFrontal = false;
    // Batch mode
    if (argc > 1 && strcmp(argv[1],"-batch") == 0) {
	    if (argc < 9) {
		printf("Usage: TestBump -batch <imList> <out prefix> <input 3D folder> <input bump folder> <input segmentation folder> <BaselFace.dat path> <dlib path> [<LM folder> [<symmetry flag>]]\n");
		return -1;
	    }
	    char imDList[200], bumpDir[200], segDir[200];
	    char lmDir[200] = "";
	    strcpy(out_prefix,argv[3]);
	    strcpy(bumpDir,argv[5]);
	    strcpy(segDir,argv[6]);
	    DlibWrapper dw(argv[8]);
	    int outSize = 500;			// Size of visualized image
	    bool symFlag = true;
	    //if (argc > 10) symFlag = strcmp(argv[10],"0") == 0;	  
	    if (argc > 9){
		strcpy(lmDir,argv[9]);
	    }
	    FaceServices2 fservice(argv[7]);
	    fservice.init(500,500,1000.0f);
	    
	    ifstream in_stream(argv[2]);
	    if (!in_stream.is_open()) {
		printf("Error: Image list does not exist\n");
		return -1;
	    }
    	    string line;
	    while (!in_stream.eof())
	    {
		line.clear();
		std::getline(in_stream, line);
		printf("Process %s\n",line.c_str());
	 	if (line.size() < 3) continue;
		cv::Mat oriImg = imread(line.c_str());
	    	if( oriImg.empty() )  {	
			printf("File %s does not exist\n",line.c_str());
			continue;
		}
		int lastInd0 = line.find_last_of('/');
		int lastInd1 = line.find_last_of('\\');
		if (lastInd1 < lastInd0) lastInd1 = lastInd0;
		int firstInd = line.find('.',lastInd1+1);
		line = line.substr(lastInd1+1,firstInd-(lastInd1+1));
		// load alpha
		sprintf(text, "%s/%s.ply.alpha",argv[4],line.c_str());
	        cv::Mat alpha = loadWeight(text);
		if (alpha.rows != 99){
			printf("Error: Invalid alpha input (%s)\n",text);
			continue;
		}
		// load bump
		sprintf(text, "%s/%s_bump.png",bumpDir,line.c_str());
	    	cv::Mat bumpImg = imread(text,0);
	    	if( bumpImg.empty() )  {
			printf("Error: Invalid bump input (%s)\n",text);
			continue;
		}
		// load segmentation
		sprintf(text, "%s/%s_mask.png",segDir,line.c_str());
	        cv::Mat segImg = imread(text,0);
	        if( segImg.empty() )  segImg = 255 + cv::Mat::zeros(512,512,CV_8U);

		// Get landmarks on the original image
		if (strlen(lmDir) > 0){
			sprintf(text, "%s/%s.pts",lmDir,line.c_str());
			lms = loadLM(text);
			if (lms.rows != 68){
				printf("Bad landmark input file (%s)!\n", text);
				continue;
			}
		}
		else {
			std::vector<cv::Mat> lms0 = dw.detectLM(oriImg);
			if (lms0.size() == 0){
				printf("No face is detected (%s)!\n", line.c_str());
				continue;
			}
			lms = lms0[0].clone();
		}
		sprintf(text,"%s/%s",out_prefix, line.c_str());
                fservice.combineBump(oriImg, lms, alpha, bumpImg, segImg, std::string(text), outSet, symFlag);
	    }
    }
    else { // Image mode
	    if (argc < 8) {
		printf("Usage: TestBump <imPath> <out prefix> <input 3D alpha> <input bump map> <input segmentation map> <BaselFace.dat path> <dlib path> [<LM path> [<symmetry flag>]]\n");
		return -1;
	    }
	    char imPath[200], bumpPath[200], segPath[200];
	    char lmPath[200] = "";
	    if (argc > 8){
		strcpy(lmPath,argv[8]);
	    }

	    strcpy(imPath,argv[1]);
	    strcpy(out_prefix,argv[2]);
	    strcpy(bumpPath,argv[4]);
	    strcpy(segPath,argv[5]);
	    DlibWrapper dw(argv[7]);
	    cv::Mat alpha = loadWeight(argv[3]);
	    if (alpha.rows != 99){
			std::cout << "-> Error: Invalid alpha input!" << std::endl;
			return -1;
	    }
	    int outSize = 500;			// Size of visualized image
	    bool symFlag = true;
	    if (argc > 9) symFlag = strcmp(argv[9],"0") == 0;	    

	    // original image
	    cv::Mat oriImg = imread(imPath);
	    if( oriImg.empty() )  return 0;
	    cv::Mat bumpImg = imread(bumpPath,0);
	    if( bumpImg.empty() )  return 0;
	    cv::Mat segImg = imread(segPath,0);
	    if( segImg.empty() )  return 0;

	    FaceServices2 fservice(argv[6]);	

	    // Get landmarks on the original image
	    if (strlen(lmPath) > 0){
		lms = loadLM(lmPath);
		if (lms.rows != 68){
			printf("Bad landmark input file with %d points!\n", lms.rows);
			return -1;
		}
	    }
	    else {
		std::vector<cv::Mat> lms0 = dw.detectLM(oriImg);
		if (lms0.size() == 0){
			printf("No face is detected!\n");
			return -1;
		}
		lms = lms0[0].clone();
	    }

	    fservice.init(oriImg.cols,oriImg.rows,1000.0f);
	    fservice.combineBump(oriImg, lms, alpha, bumpImg, segImg, std::string(out_prefix), outSet, symFlag);
    }
    return 0;
}

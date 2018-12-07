/* Copyright (c) 2015 USC, IRIS, Computer vision Lab */
#include "FaceServices2.h"
#include <fstream>
#include "opencv2/contrib/contrib.hpp"
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <limits>
#include <opencv2/calib3d/calib3d.hpp> 
#include "opencv2/legacy/legacy.hpp"
#include <omp.h>

using namespace std;
using namespace cv;

FaceServices2::FaceServices2(const std::string & model_file)
{
	omp_set_num_threads(8);
	mstep = 0.0001;
	countFail = 0;
	maxVal = 4;
	im_render = nullptr;
        printf("load %s\n",model_file.c_str());
	festimator.load3DMM(model_file);
}

FaceServices2::~FaceServices2(void)
{
}

// Setup with image size (w,h) and focal length f
void FaceServices2::init(int w, int h, float f){
	// Instrinsic matrix
	memset(_k,0,9*sizeof(float));
	_k[8] = 1;
	_k[0] = -f;
	_k[4] = f;
	_k[2] = w/2.0f;
	_k[5] = h/2.0f;
	if (faces.empty())
        	faces = festimator.getFaces() - 1;
	cv::Mat shape = festimator.getShape(cv::Mat(99,1,CV_32F));
	tex = shape*0 + 128;
	
	// Initialize image renderer
	if (im_render == nullptr)
	{
		im_render = new FImRenderer(cv::Mat::zeros(h, w, CV_8UC3));
		im_render->loadMesh(shape, shape * 0, faces);
	}
	else im_render->init(cv::Mat::zeros(h, w, CV_8UC3));
}

// Estimate pose and expression from landmarks
//     Inputs:
//	   colorIm    : Input image
//         lms        : 2D landmarks (68x2)
//         alpha      : Subject-specific shape parameters (99x1)
//     Outputs:
//         vecR       : Rotation angles (3x1)
//         vecT       : Translation vector (3x1)
//         exprWeight : Expression parameters (29x1)
bool FaceServices2::estimatePoseExpr(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprW){
	char text[200];
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	BFMParams params;
	params.init();
	exprW = cv::Mat::zeros(29,1,CV_32F);

	int M = 99;
	// get subject shape
	cv::Mat shape = festimator.getShape(alpha);

	// get 3D landmarks
	Mat landModel0 = festimator.getLM(shape,0);
	int nLM = landModel0.rows;

	// compute 3D pose w/ the first 60 2D-3D correspondences
	Mat landIm = cv::Mat( 60,2,CV_32F);
	Mat landModel = cv::Mat( 60,3,CV_32F);
	for (int i=0;i<60;i++){
		landModel.at<float>(i,0) = landModel0.at<float>(i,0);
		landModel.at<float>(i,1) = landModel0.at<float>(i,1);
		landModel.at<float>(i,2) = landModel0.at<float>(i,2);
		landIm.at<float>(i,0) = lms.at<float>(i,0);
		landIm.at<float>(i,1) = lms.at<float>(i,1);
	}
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);

	// reselect 3D landmarks given estimated yaw angle
	float yaw = -vecR.at<float>(1,0);
	landModel0 = festimator.getLM(shape,yaw);
	// select landmarks to use based on estimated yaw angle
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (i > 16 || abs(yaw) <= M_PI/10 || (yaw > M_PI/10 && i > 7) || (yaw < -M_PI/10 && i < 9))
			lmVisInd.push_back(i);
	}
	landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}
	// resetimate 3D pose
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);
	
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);
	
	// add the inner mouth landmark points for expression estimation
	for (int i=60;i<68;i++) lmVisInd.push_back(i);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}

	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;

	params.weightLM = 8.0f;
	Mat alpha0;
	int iter=0;
	int badCount = 0;
	memset(params.doOptimize,true,sizeof(bool)*6);

	// optimize pose+expression from landmarks
	int EM = 29;
	float renderParams_tmp[RENDER_PARAMS_COUNT];

	for (;iter<60;iter++) {
			if (iter%20 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}
	iter = 60;

	// optimize expression only
	memset(params.doOptimize,false,sizeof(bool)*6);countFail = 0;
	for (;iter<200;iter++) {
			if (iter%60 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}
}

// Render texture-less face mesh
//     Inputs:
//	   colorIm    : Input image
//         alpha      : Subject-specific shape parameters (99x1)
//         r          : Rotation angles (3x1)
//         t          : Translation vector (3x1)
//         exprW      : Expression parameters (29x1)
//     Outputs:
//         Rendered texture-less face w/ the defined shape, expression, and pose
cv::Mat FaceServices2::renderShape(cv::Mat colorIm, cv::Mat alpha,cv::Mat vecR,cv::Mat vecT,cv::Mat exprW){
	float renderParams[RENDER_PARAMS_COUNT];
	for (int i =0;i<3;i++)
		renderParams[i] = vecR.at<float>(i,0);
	for (int i =0;i<3;i++)
		renderParams[i+3] = vecT.at<float>(i,0);
	
	// Ambient
	renderParams[RENDER_PARAMS_AMBIENT] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+1] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+2] = 0.69225;
	// Diffuse
	renderParams[RENDER_PARAMS_DIFFUSE] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+1] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+2] = 0.30754;
	// LIGHT
	renderParams[RENDER_PARAMS_LDIR] = 3.1415/4;
	renderParams[RENDER_PARAMS_LDIR+1] = 3.1415/4;
	// OTHERS
	renderParams[RENDER_PARAMS_CONTRAST] = 1;
	renderParams[RENDER_PARAMS_GAIN] = renderParams[RENDER_PARAMS_GAIN+1] = renderParams[RENDER_PARAMS_GAIN+2] = RENDER_PARAMS_GAIN_DEFAULT;
	renderParams[RENDER_PARAMS_OFFSET] = renderParams[RENDER_PARAMS_OFFSET+1] = renderParams[RENDER_PARAMS_OFFSET+2] = RENDER_PARAMS_OFFSET_DEFAULT;
		
	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;

	shape = festimator.getShape2(alpha,exprW);
	im_render->copyShape(shape);

	// estimate shaded colors
	cv::Mat colors;
	tex = shape*0 + 128;
	rs.estimateColor(shape,tex,faces,renderParams,colors);
	im_render->copyColors(colors);
	im_render->loadModel();

	// render
	cv::Mat outRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat outDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r,t,_k[4],outRGB,outDepth);
	return outRGB;
}

// Render texture-less face mesh
//     Inputs:
//	   colorIm    : Input image
//         alpha      : Subject-specific shape parameters (99x1)
//         r          : Rotation angles (3x1)
//         t          : Translation vector (3x1)
//         exprW      : Expression parameters (29x1)
//     Outputs:
//         outRGB      : Rendered texture-less face w/ the defined shape, expression, and pose
//         outDepth   : The corresponding Z-buffer
void FaceServices2::renderShape(cv::Mat colorIm, cv::Mat alpha,cv::Mat vecR,cv::Mat vecT,cv::Mat exprW, cv::Mat &outRGB, cv::Mat &outDepth){
	float renderParams[RENDER_PARAMS_COUNT];
	for (int i =0;i<3;i++)
		renderParams[i] = vecR.at<float>(i,0);
	for (int i =0;i<3;i++)
		renderParams[i+3] = vecT.at<float>(i,0);
	
	// Ambient
	renderParams[RENDER_PARAMS_AMBIENT] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+1] = 0.69225;
	renderParams[RENDER_PARAMS_AMBIENT+2] = 0.69225;
	// Diffuse
	renderParams[RENDER_PARAMS_DIFFUSE] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+1] = 0.30754;
	renderParams[RENDER_PARAMS_DIFFUSE+2] = 0.30754;
	// LIGHT
	renderParams[RENDER_PARAMS_LDIR] = 3.1415/4;
	renderParams[RENDER_PARAMS_LDIR+1] = 3.1415/4;
	// OTHERS
	renderParams[RENDER_PARAMS_CONTRAST] = 1;
	renderParams[RENDER_PARAMS_GAIN] = renderParams[RENDER_PARAMS_GAIN+1] = renderParams[RENDER_PARAMS_GAIN+2] = RENDER_PARAMS_GAIN_DEFAULT;
	renderParams[RENDER_PARAMS_OFFSET] = renderParams[RENDER_PARAMS_OFFSET+1] = renderParams[RENDER_PARAMS_OFFSET+2] = RENDER_PARAMS_OFFSET_DEFAULT;
		
	float* r = renderParams + RENDER_PARAMS_R;
	float* t = renderParams + RENDER_PARAMS_T;

	shape = festimator.getShape2(alpha,exprW);
	im_render->copyShape(shape);

	// estimate shaded colors
	cv::Mat colors;
	tex = shape*0 + 128;
	rs.estimateColor(shape,tex,faces,renderParams,colors);
	im_render->copyColors(colors);
	im_render->loadModel();

	// render
	outRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	outDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);
	im_render->render(r,t,_k[4],outRGB,outDepth);
}

// Get next motion for the animated face visualization. In this sample code, only rotation is changed
//     Inputs:
//         currFrame  : current frame index. Will be increased after this call
//     Outputs:
//         vecR       : Rotation angles (3x1)
//         vecT       : Translation vector (3x1)
//         exprWeights: Expression parameters (29x1)
void FaceServices2::nextMotion(int &currFrame, cv::Mat &vecR, cv::Mat &vecT, cv::Mat &exprWeights){
    float stepYaw = 1;
    float PPI = 3.141592;
    int maxYaw = 70/stepYaw;
    float stepPitch = 1;
    int maxPitch = 45/stepPitch;

    int totalFrames = 4 * (maxYaw + maxPitch);
    currFrame = (currFrame + 1) % totalFrames;
    vecR = vecR*0 + 0.00001;
    // Rotate left
    if (currFrame < maxYaw) vecR.at<float>(1,0) = -currFrame*stepYaw * PPI/180;
    else if (currFrame < 2*maxYaw) vecR.at<float>(1,0) = -stepYaw * (2*maxYaw - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 3*maxYaw) vecR.at<float>(1,0) = (currFrame-2*maxYaw) * stepYaw * PPI/180;
    else if (currFrame < 4*maxYaw) vecR.at<float>(1,0) = stepYaw * (4*maxYaw - currFrame) * PPI/180;
    
    // Rotate up
    else if (currFrame < 4*maxYaw + maxPitch) vecR.at<float>(0,0) = -(currFrame-4*maxYaw)*stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 2*maxPitch) vecR.at<float>(0,0) = -stepPitch * (4*maxYaw+2*maxPitch - currFrame) * PPI/180;
    // Rotate right
    else if (currFrame < 4*maxYaw + 3*maxPitch) vecR.at<float>(0,0) = (currFrame-4*maxYaw-2*maxPitch) * stepPitch * PPI/180;
    else if (currFrame < 4*maxYaw + 4*maxPitch) vecR.at<float>(0,0) = stepPitch * (4*maxPitch+4*maxYaw - currFrame) * PPI/180;
}
	
// Adding background to the rendered face image
//     Inputs:
//         bg         : Background image 
//         depth      : Z-buffer of the rendered face
//     Input & output:
//         target     : The rendered face image
void FaceServices2::mergeIm(cv::Mat* target,cv::Mat bg,cv::Mat depth){
	for (int i=0;i<bg.rows;i++){
		for (int j=0;j<bg.cols;j++){
			if (depth.at<float>(i,j) >= 0.9999)
				target->at<Vec3b>(i, j) = bg.at<Vec3b>(i,j);
		}
	}
}

//////////////////////////////////////////// Supporting functions ///////////////////////////////////////////////////
// Compute Hessian matrix diagonal
float FaceServices2::updateHessianMatrix(cv::Mat alpha, float* renderParams, cv::Mat faces, cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat exprW ){
	int M = alpha.rows;
	int EM = exprW.rows;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	params.hessDiag.release();
	params.hessDiag = cv::Mat::zeros(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);
	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];

	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(alpha, lmInds, landIm, renderParams, exprW);
	cEF = currEF;

	// expr
	step = mstep*5;
	if (params.optimizeExpr) {
		for (int i=0;i<EM; i++){
			expr2.release(); expr2 = exprW.clone();
			expr2.at<float>(i,0) += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams,expr2);
			expr2.at<float>(i,0) -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams,expr2);
			params.hessDiag.at<float>(2*M+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ params.weightRegExpr * 2/(0.25f*29) ;
		}
	}
	// r
	step = mstep*2;
	if (params.doOptimize[RENDER_PARAMS_R]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_R+i] += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams2,exprW);

			renderParams2[RENDER_PARAMS_R+i] -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) + 2.0f/params.weightReg[RENDER_PARAMS_R+i];

		}
	}
	// t
	step = mstep*10;
	if (params.doOptimize[RENDER_PARAMS_T]) {
		for (int i=0;i<3; i++){
			memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
			renderParams2[RENDER_PARAMS_T+i] += step;
			float tmpEF1 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			renderParams2[RENDER_PARAMS_T+i] -= 2*step;
			float tmpEF2 = eF(alpha, lmInds, landIm, renderParams2,exprW);
			params.hessDiag.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.weightLM * (tmpEF1 - 2*currEF + tmpEF2)/(step*step) 
				+ 2.0f/params.weightReg[RENDER_PARAMS_T+i];
		}
	}
	return 0;
}

// Compute gradient
cv::Mat FaceServices2::computeGradient(cv::Mat alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, std::vector<int> &inds, cv::Mat exprW){
	int M = alpha.rows;
	int EM = exprW.rows;
	int nTri = 40;
	float step;
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat out(2*M+EM+RENDER_PARAMS_COUNT,1,CV_32F);

	cv::Mat alpha2, expr2;
	float renderParams2[RENDER_PARAMS_COUNT];
	cv::Mat rVec(3,1,CV_32F,renderParams+RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F,renderParams+RENDER_PARAMS_T);
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	float currEF = eF(alpha, lmInds, landIm, renderParams,exprW);
	cEF = currEF;
	
	#pragma omp parallel for
	for (int target=0;target<EM+6; target++){
	  if (target < EM) {
		// expr
		float step = mstep*5;
		if (params.optimizeExpr) {
				int i = target;
				std::vector<cv::Point2f> pPoints;
				cv::Mat expr2 = exprW.clone();
				expr2.at<float>(i,0) += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams,expr2);
				out.at<float>(2*M+i,0) = params.weightLM * (tmpEF - currEF)/step
					+ params.weightRegExpr * 2*exprW.at<float>(i,0)/(0.25f*29);
		}
	   }
	   else if (target < EM+3) {
		// r
		float step = mstep*2;
		if (params.doOptimize[RENDER_PARAMS_R]) {
			int i = target-EM;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_R+i] += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+i,0) = params.weightLM * (tmpEF - currEF)/step;
				out.at<float>(2*M+EM+i,0) += 2*(renderParams[RENDER_PARAMS_R+i] - params.initR[RENDER_PARAMS_R+i])/params.weightReg[RENDER_PARAMS_R+i];
		}
	  }
	  else {
		// t
		float step = mstep*10;
		if (params.doOptimize[RENDER_PARAMS_T]) {
			int i = target-EM-3;
				float renderParams2[RENDER_PARAMS_COUNT];
				memcpy(renderParams2,renderParams,RENDER_PARAMS_COUNT*sizeof(float));
				renderParams2[RENDER_PARAMS_T+i] += step;
				float tmpEF = eF(alpha, lmInds, landIm, renderParams2,exprW);
				out.at<float>(2*M+EM+RENDER_PARAMS_T+i,0) = params.weightLM * (tmpEF - currEF)/step 
					+ 2*(renderParams[RENDER_PARAMS_T+i] - params.initR[RENDER_PARAMS_T+i])/params.weightReg[RENDER_PARAMS_T+i];
		}
	  }
	}
	return out;
}

// Compute landmark error
float FaceServices2::eF(cv::Mat alpha, std::vector<int> inds, cv::Mat landIm, float* renderParams, cv::Mat exprW){
	Mat k_m(3,3,CV_32F,_k);
	cv::Mat mLM = festimator.getLMByAlpha(alpha,-renderParams[RENDER_PARAMS_R+1], inds, exprW);
	
	cv::Mat rVec(3,1,CV_32F, renderParams + RENDER_PARAMS_R);
	cv::Mat tVec(3,1,CV_32F, renderParams + RENDER_PARAMS_T);
	std::vector<cv::Point2f> allImgPts;
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );

	cv::projectPoints( mLM, rVec, tVec, k_m, distCoef, allImgPts );
	float err = 0;
	for (int i=0;i<mLM.rows;i++){
		float val = landIm.at<float>(i,0) - allImgPts[i].x;
		err += val*val;
		val = landIm.at<float>(i,1) - allImgPts[i].y;
		err += val*val;
	}
	return sqrt(err/mLM.rows);
}

// Newton optimization step
void FaceServices2::sno_step(cv::Mat &alpha, float* renderParams, cv::Mat faces,cv::Mat colorIm, std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW){
	float lambda = 0.05;
	std::vector<int> inds;
	cv::Mat dE = computeGradient(alpha, renderParams, faces, colorIm, lmInds, landIm, params,inds, exprW);
	params.gradVec.release(); params.gradVec = dE.clone();
	cv::Mat dirMove = dE*0;

	int M = alpha.rows;
	int EM = exprW.rows;
	if (params.optimizeExpr){
		for (int i=0;i<EM;i++)
			if (abs(params.hessDiag.at<float>(2*M+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+i,0) = - lambda*dE.at<float>(2*M+i,0)/abs(params.hessDiag.at<float>(2*M+i,0));
			}
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			if (abs(params.hessDiag.at<float>(2*M+EM+i,0)) > 0.0000001) {
				dirMove.at<float>(2*M+EM+i,0) = - lambda*dE.at<float>(2*M+EM+i,0)/abs(params.hessDiag.at<float>(2*M+EM+i,0));
			}
		}
	}
	float pc = line_search(alpha, renderParams, dirMove,inds, faces, colorIm, lmInds, landIm, params, exprW, 10);
	if (pc == 0) countFail++;
	else {
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				exprW.at<float>(i,0) += pc*dirMove.at<float>(i+2*M,0);
				if (exprW.at<float>(i,0) > 3) exprW.at<float>(i,0) = 3;
				else if (exprW.at<float>(i,0) < -3) exprW.at<float>(i,0) = -3;
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				renderParams[i] += pc*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (renderParams[i] > 1.0) renderParams[i] = 1.0;
					if (renderParams[i] < 0) renderParams[i] = 0;

				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (renderParams[i] > 3.0) renderParams[i]  = 3;
					if (renderParams[i] < 0.3) renderParams[i] = 0.3;
				}
			}
		}
	}
}

// Line search optimization
float FaceServices2::line_search(cv::Mat &alpha, float* renderParams, cv::Mat &dirMove,std::vector<int> inds, cv::Mat faces,cv::Mat colorIm,std::vector<int> lmInds, cv::Mat landIm, BFMParams &params, cv::Mat &exprW, int maxIters){
	float step = 1.0f;
	float sstep = 2.0f;
	float minStep = 0.0001f;
	cv::Mat alpha2, exprW2;
	float renderParams2[RENDER_PARAMS_COUNT];
	alpha2 = alpha.clone();
	exprW2 = exprW.clone();
	memcpy(renderParams2,renderParams,sizeof(float)*RENDER_PARAMS_COUNT);

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	std::vector<cv::Point2f> pPoints;
	cv::Mat rVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_R);
	cv::Mat tVec2(3,1,CV_32F,renderParams2+RENDER_PARAMS_T);

	int M = alpha.rows;
	int EM = exprW.rows;
	float ssize = 0;
	for (int i=0;i<dirMove.rows;i++) ssize += dirMove.at<float>(i,0)*dirMove.at<float>(i,0);
	ssize = sqrt(ssize);
	if (ssize > (2*M+EM+RENDER_PARAMS_COUNT)/5.0f) {
		step = (2*M+EM+RENDER_PARAMS_COUNT)/(5.0f * ssize);
		ssize = (2*M+EM+RENDER_PARAMS_COUNT)/5.0f;
	}
	if (ssize < minStep){
		return 0;
	}
	int tstep = floor(log(ssize/minStep));
	if (tstep < maxIters) maxIters = tstep;

	float curCost = computeCost(cEF, alpha, renderParams, params, exprW );

	bool hasNoBound = false;
	int iter = 0;
	for (; iter<maxIters; iter++){
		if (params.optimizeExpr){
			for (int i=0;i<EM;i++) {
				float tmp = exprW.at<float>(i,0) + step*dirMove.at<float>(2*M+i,0);
				if (tmp >= 3) exprW2.at<float>(i,0) = 3;
				else if (tmp <= -3) exprW2.at<float>(i,0) = -3;
				else {
					exprW2.at<float>(i,0) = tmp;
					hasNoBound = true;
				}
			}
		}

		for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
			if (params.doOptimize[i]){
				float tmp = renderParams[i] + step*dirMove.at<float>(2*M+EM+i,0);
				if (i == RENDER_PARAMS_CONTRAST || (i>=RENDER_PARAMS_AMBIENT && i<RENDER_PARAMS_DIFFUSE+3) ) {
					if (tmp > 1.0) renderParams2[i] = 1.0f;
					else if (tmp < -1.0) renderParams2[i] = -1.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else if (i >= RENDER_PARAMS_GAIN && i<=RENDER_PARAMS_GAIN+3) {
					if (tmp >= 3.0) renderParams2[i] = 3.0f;
					else if (tmp <= -3.0) renderParams2[i] = -3.0f;
					else {
						renderParams2[i] = tmp;
						hasNoBound = true;
					}
				}
				else renderParams2[i] = tmp;
			}
		}
		if (!hasNoBound) {
			iter = maxIters; break;
		}
		float tmpEF = cEF;
		if (params.weightLM > 0) tmpEF = eF(alpha2, lmInds,landIm,renderParams2, exprW2);
		float tmpCost = computeCost(tmpEF, alpha2, renderParams2, params,exprW2);
		if (tmpCost < curCost) {
			break;
		}
		else {
			step = step/sstep;
		}
	}
	if (iter >= maxIters) return 0;
	else return step;
}

// Cost function
float FaceServices2::computeCost(float vEF, cv::Mat &alpha, float* renderParams, BFMParams &params, cv::Mat &exprW ){
	float val = params.weightLM*vEF;
	int M = alpha.rows;
	if (params.optimizeExpr){
		for (int i=0;i<exprW.rows;i++)
			val += params.weightRegExpr * exprW.at<float>(i,0)*exprW.at<float>(i,0)/(0.5f*29);
	}

	for (int i=0;i<RENDER_PARAMS_COUNT;i++) {
		if (params.doOptimize[i]){
			val += (renderParams[i] - params.initR[i])*(renderParams[i] - params.initR[i])/params.weightReg[i];
		}
	}
	return val;
}


bool FaceServices2::combineBump(cv::Mat colorIm, cv::Mat lms, cv::Mat alpha, cv::Mat bumpMap, cv::Mat maskMap, string out_prefix, OutputSettings outSet, bool softSym, int margin){
	char text[200];
	float renderParams[RENDER_PARAMS_COUNT];
	float renderParams2[RENDER_PARAMS_COUNT];
	Mat k_m(3,3,CV_32F,_k);
	BFMParams params;
	params.init();
	cv::Mat vecR, vecT;
	cv::Mat exprW = cv::Mat::zeros(29,1,CV_32F);
	std::vector<int> pairInd = festimator.getPair();

	int M = 99;
	// get subject shape
	cv::Mat shape = festimator.getShape(alpha);

	// get 3D landmarks
	Mat landModel0 = festimator.getLM(shape,0);
	int nLM = landModel0.rows;

	// compute 3D pose w/ the first 60 2D-3D correspondences
	Mat landIm = cv::Mat( 60,2,CV_32F);
	Mat landModel = cv::Mat( 60,3,CV_32F);
	for (int i=0;i<60;i++){
		landModel.at<float>(i,0) = landModel0.at<float>(i,0);
		landModel.at<float>(i,1) = landModel0.at<float>(i,1);
		landModel.at<float>(i,2) = landModel0.at<float>(i,2);
		landIm.at<float>(i,0) = lms.at<float>(i,0);
		landIm.at<float>(i,1) = lms.at<float>(i,1);
	}
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);

	// reselect 3D landmarks given estimated yaw angle
	float yaw = -vecR.at<float>(1,0);
	landModel0 = festimator.getLM(shape,yaw);
	// select landmarks to use based on estimated yaw angle
	std::vector<int> lmVisInd;
	for (int i=0;i<60;i++){
		if (i > 16 || abs(yaw) <= M_PI/10 || (yaw > M_PI/10 && i > 7) || (yaw < -M_PI/10 && i < 9))
			lmVisInd.push_back(i);
	}
	landModel = cv::Mat( lmVisInd.size(),3,CV_32F);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landModel.at<float>(i,0) = landModel0.at<float>(ind,0);
		landModel.at<float>(i,1) = landModel0.at<float>(ind,1);
		landModel.at<float>(i,2) = landModel0.at<float>(ind,2);
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}
	// resetimate 3D pose
	festimator.estimatePose3D(landModel,landIm,k_m,vecR,vecT);
	
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_R+i] = vecR.at<float>(i,0);
	for (int i=0;i<3;i++)
		params.initR[RENDER_PARAMS_T+i] = vecT.at<float>(i,0);
	memcpy(renderParams,params.initR,sizeof(float)*RENDER_PARAMS_COUNT);
	
	// add the inner mouth landmark points for expression estimation
	for (int i=60;i<68;i++) lmVisInd.push_back(i);
	landIm = cv::Mat::zeros( lmVisInd.size(),2,CV_32F);
	for (int i=0;i<lmVisInd.size();i++){
		int ind = lmVisInd[i];
		landIm.at<float>(i,0) = lms.at<float>(ind,0);
		landIm.at<float>(i,1) = lms.at<float>(ind,1);
	}

	float bCost, cCost, fCost;
	int bestIter = 0;
	bCost = 10000.0f;

	params.weightLM = 8.0f;
	Mat alpha0;
	int iter=0;
	int badCount = 0;
	memset(params.doOptimize,true,sizeof(bool)*6);

	// optimize pose+expression from landmarks
	int EM = 29;
	float renderParams_tmp[RENDER_PARAMS_COUNT];

	for (;iter<60;iter++) {
			if (iter%20 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}
	iter = 60;

	// optimize expression only
	memset(params.doOptimize,false,sizeof(bool)*6);countFail = 0;
	for (;iter<200;iter++) {
			if (iter%60 == 0) {
				cCost = updateHessianMatrix(alpha,renderParams,faces,colorIm,lmVisInd,landIm,params, exprW);
				if (countFail > 10) {
					countFail = 0;
					break;
				}
			}
			sno_step(alpha, renderParams, faces,colorIm,lmVisInd,landIm,params,exprW);
		}

	shape = festimator.getShape(alpha,exprW);
	cv::Mat tex = festimator.getTexture(alpha*0);
	
	// Foundation shape (s+e)
	if (outSet.foundationFrontal) {
		sprintf(text,"%s_foundation_frontal.ply",out_prefix.c_str());
		write_plyShapeFloat(text,shape,faces);
	}
	cv::Mat faces_fill = festimator.getFaces_fill() - 1;
	int NEI = 1;
	
	///////////////////////////////////////////////////////////////////////////////
	////// Estimate occlusion map
	if (im_render == 0) im_render = new FImRenderer(cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3));
	im_render->loadMesh(shape,tex,faces_fill);
	cv::Mat refRGB = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_8UC3);
	cv::Mat refDepth = cv::Mat::zeros(colorIm.rows,colorIm.cols,CV_32F);

	float r[3], t[3];
	for (int i=0;i<3;i++){
		r[i] = vecR.at<float>(i,0);
		t[i] = vecT.at<float>(i,0);
	}
	im_render->loadModel();
	im_render->render(r,t,_k[4],refRGB,refDepth);
		// Compute occlusion scores (Hassners 2014)
	cv::Mat occMat = computeOccMat(im_render, r, t);
		// Compute visibility based on projection & input segmentation map
	cv::Mat mask = refDepth < 0.9999;
	cv::Mat maskMap_crop = maskMap(cv::Rect(6,6,mask.cols,mask.rows));
	mask = mask & (maskMap_crop > 127);
	bool* visible = 0;
	vector<cv::Point2f> projInit = projectCheckVis2(im_render, shape, r, t, refDepth, visible);
	if (maskMap.rows > 0) {
	   for (int i=0;i<projInit.size();i++) {
		int x = floor(projInit[i].x);
		int y = floor(projInit[i].y);
		if (!visible[i]) {
			continue;
		}
		if (x < 0 || x > refDepth.cols-2 || y < 0 || y > refDepth.rows-2) visible[i] = false;
		else {
		   if (maskMap.at<unsigned char>(y+6,x+6) < 127) {
			visible[i] = false;
		    }
		}
	   }
	}
		// Combine
	cv::Mat occStructureWeights = computeOccScores(shape, visible, occMat, r, t);
	/////////////////////////////////////////////////////////////////////////////
	//////         Compute dense mesh
		// Rotate s+e by pose (rotShape0)
	cv::Mat matR;
	cv::Rodrigues(vecR,matR);
	cv::Mat rotShape = matR * shape.t() + cv::repeat(vecT,1,shape.rows);
	rotShape = rotShape.t();
	cv::Mat rotShape0 = rotShape.clone();

	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;
		// Compute mindepth & prepare vertex indices (vindex) for the dense mesh
	float minDepth = 0;
	std::vector<int> pointInd[2];
	cv::Mat vindex = cv::Mat::zeros(mask.rows,mask.cols,CV_32S)-1;
	int numNeighbors = 0;
	for (int x=0;x<refDepth.cols;x++){
		for (int y=0;y<refDepth.rows;y++){
                        if (maskMap.rows > 0 && maskMap.at<unsigned char>(y+6,x+6) < 127) continue; 
			float dd = refDepth.at<float>(y,x);
			if (dd<0.9999){
				vindex.at<int>(y,x) = pointInd[0].size();
				pointInd[0].push_back(y);
				pointInd[1].push_back(x);
				refDepth.at<float>(y,x) = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
				if (minDepth > refDepth.at<float>(y,x))
					minDepth = refDepth.at<float>(y,x);
			}
		}
	}
	for (int x=0;x<refDepth.cols;x++){
		for (int y=0;y<refDepth.rows;y++){
			float dd = refDepth.at<float>(y,x);
			if (dd>=0.9999){
				refDepth.at<float>(y,x) = minDepth;
			}
		}
	}
	if (outSet.foundationAligned) {
		sprintf(text,"%s_foundation_aligned.ply",out_prefix.c_str());
		write_plyShapeFloat(text,rotShape,faces);
	}
		// Compute dense vertices w/o bump (v)
	refDepth = refDepth-minDepth;
	cv::Mat v(pointInd[0].size(),3,CV_32F);
	cv::Mat v00 = v.clone();
	for (int i=0;i<pointInd[0].size();i++){
			v.at<float>(i,2) = refDepth.at<float>(pointInd[0][i],pointInd[1][i])+minDepth;
			v.at<float>(i,0) = v.at<float>(i,2)*(pointInd[1][i] - _k[2])/_k[0];
			v.at<float>(i,1) = v.at<float>(i,2)*(pointInd[0][i] - _k[5])/_k[4];
			v00.at<float>(i,2) = refDepth.at<float>(pointInd[0][i],pointInd[1][i])+minDepth;
			v00.at<float>(i,0) = pointInd[1][i];
			v00.at<float>(i,1) = pointInd[0][i];
	}
		// Compute triangles on the dense mesh (fac2)
	std::vector<Vec3i> fac;
	computeFaces(vindex,fac);
	cv::Mat fac2(fac.size(),3,CV_32S);
	for (int i=0;i<fac.size();i++){
			fac2.at<int>(i,0) = fac[i](0);
			fac2.at<int>(i,1) = fac[i](1);
			fac2.at<int>(i,2) = fac[i](2);
	}
	cv::Mat fac2_im = fac2.clone();
	fac2.col(1).copyTo(fac2_im.col(2));
	fac2.col(2).copyTo(fac2_im.col(1));

	//sprintf(text,"%s_foundation_aligned_image.ply",out_prefix.c_str());
	//write_plyShapeFloat(text,v00,fac2_im);
		// Add bump map to the dense mesh (v)
	for (int j=0;j<pointInd[0].size();j++){
		int x = pointInd[1][j];
		int y = pointInd[0][j];
		unsigned char change0 = bumpMap.at<unsigned char>(y+6,x+6);
		float change = change0;
		change = (change-127)/12 * 1.25;
		v.at<float>(j,2) += change;
		refDepth.at<float>(y,x) = v.at<float>(j,2) - minDepth;
		v00.at<float>(j,2) = v.at<float>(j,2);
	}
	if (outSet.withBumpAligned) {
		sprintf(text,"%s_withBump_aligned.ply",out_prefix.c_str());
		write_plyShapeFloat(text,v,fac2);
		//sprintf(text,"%s_withBump_aligned_image.ply",out_prefix.c_str());
		//write_plyShapeFloat(text,v00,fac2_im);
	}
	if (!softSym) return true;

	///////////////////////////////////////////////////////////////////////////////
	////////	Prepare neighborhood infor (for Poisson blending)
	int* filterInd = new int[rotShape.rows];
	vector<int> filterFaces;
	cv::Mat fFaces;
	std::vector<int>* neighbors =  new std::vector<int>[pointInd[0].size()];
	for (int i=0;i<pointInd[0].size();i++){
		// Check neighbors
		int y = pointInd[0][i];
		int x = pointInd[1][i];
		int sx = (x-NEI>=0)?(x-NEI):0;
		int ex = x<=(refDepth.cols-1-NEI)?(x+NEI):(refDepth.cols-1);
		int sy = (y-NEI>=0)?(y-NEI):0;
		int ey = y<=(refDepth.rows-1-NEI)?(y+NEI):(refDepth.rows-1);
		for (int rx=sx;rx<=ex;rx++)
			for (int ry=sy;ry<=ey;ry++)
				if ((rx != x || ry != y) && mask.at<unsigned char>(ry,rx)>0) 
			{
					numNeighbors++;
					neighbors[i].push_back(vindex.at<int>(ry,rx));
			}
	}

	mask = mask/255;

	v = matR.t() * v.t() + repeat(-matR.t() * vecT, 1, v.rows) ;
	cv::Mat best_v = v.t();
	if (outSet.withBumpFrontal) {
		sprintf(text,"%s_withBump_frontal.ply",out_prefix.c_str());
		write_plyShapeFloat(text,best_v,fac2);
	}
	
	// Add bump map to the sparse mesh (rotShape)
	cv::Mat rotShapeFlip = rotShape.clone();
	int* filterIndFlip = new int[rotShape.rows];
	for (int i=0;i<rotShape.rows;i++) filterInd[i] = filterIndFlip[i] = -1;
	int countV=0;
	for (int i=0;i<rotShape.rows;i++){
		int ind = pairInd[i]-1;
		if (ind >= 0 && pairInd[ind]-1 == i) {
			if (visible[i]){
				int y = floor(projInit[i].y);
				int x = floor(projInit[i].x);
				if (x>=0 && y>=0 && x < refDepth.cols-1 && y < refDepth.rows-1){
					if (mask.at<unsigned char>(y,x)>0 && mask.at<unsigned char>(y,x+1)>0 && mask.at<unsigned char>(y+1,x)>0 && mask.at<unsigned char>(y+1,x+1)>0){
						CvPoint2D64f tmpPt = cvPoint2D64f(projInit[i].x,projInit[i].y);
						float dd = avSubMatValue32F( &tmpPt, &refDepth );
						float change = dd + minDepth - rotShape.at<float>(i,2);
						rotShape.at<float>(i,2) += change;
						filterInd[i] = countV; filterIndFlip[ind] = countV; countV++;
					}
				}
			}
		}
	}
	cv::Mat rotShapeOri = rotShape.clone();
	//        Frontalize
	rotShape = matR.t() * (rotShape.t() - cv::repeat(vecT,1,shape.rows));
	rotShape = rotShape.t();	// sparse mesh
	rotShape0 = matR.t() * (rotShape0.t() - cv::repeat(vecT,1,shape.rows));
	rotShape0 = rotShape0.t();	// foundation mesh

	//	  Flip
	for (int i=0;i<rotShape.rows; i++){
	    if (filterInd[i] >= 0) {
		int ind = pairInd[i]-1;
		rotShapeFlip.at<float>(ind,0) = -rotShape.at<float>(i,0);
		rotShapeFlip.at<float>(ind,1) = rotShape.at<float>(i,1);
		rotShapeFlip.at<float>(ind,2) = rotShape.at<float>(i,2);
	    }
	}


	//	Refine pose by aligning the original shape & the flipped one
	std::vector<int> alignInds;
	for (int i=0;i<rotShape.rows;i++){
	   if (filterInd[i] >= 0 && abs(rotShape.at<float>(i,0)) < 15 ) {
	      int ind = pairInd[i]-1;
		if (ind >= 0 && filterInd[ind] >= 0) alignInds.push_back(i);
           }
	}
	cv::Mat alignRMat = cv::Mat::eye(3,3,CV_32F);
	if (alignInds.size() > 100){
		int AL = alignInds.size();
		cv::Mat p1(AL,3,CV_32F);
		cv::Mat p2(AL,3,CV_32F);
		for (int i=0;i<AL;i++) {
			p1.at<float>(i,0) = rotShape.at<float>(alignInds[i],0);
			p1.at<float>(i,1) = rotShape.at<float>(alignInds[i],1);
			p1.at<float>(i,2) = rotShape.at<float>(alignInds[i],2);
			int ind = pairInd[alignInds[i]]-1;
			p2.at<float>(i,0) = -rotShapeFlip.at<float>(ind,0);
			p2.at<float>(i,1) = rotShapeFlip.at<float>(ind,1);
			p2.at<float>(i,2) = rotShapeFlip.at<float>(ind,2);
		}
		cv::Mat alignT,alignR;

        	estimateRigidTranf(p1, p2, alignRMat, alignT);
		cv::Rodrigues(alignRMat,alignR);
		alignR = alignR/2;
		alignT = alignT/2;
		cv::Rodrigues(alignR,alignRMat);
		rotShape = alignRMat * rotShape.t() + cv::repeat(alignT,1,shape.rows);
		rotShape = rotShape.t();
		rotShape0 = alignRMat * rotShape0.t() + cv::repeat(alignT,1,shape.rows);
		rotShape0 = rotShape0.t();
		rotShapeFlip = alignRMat.t() * (rotShapeFlip.t() - cv::repeat(alignT,1,shape.rows));
		rotShapeFlip = rotShapeFlip.t();
	}

	///////////////////////////////////////////////////////////////////////////////
	////////	Check vertex quality (bad vertices are considered as occludded)
	std::vector<SpT> triA[3];
	float LAMBDA_OCC = 0.01;
	float LAMBDA_OCC_PAD = 0.1;
		// Check triangle area
	cv::Mat tex3 = tex * 0 + 255;
	for (int i=0;i<faces.rows;i++) {
		int v1 = faces.at<int>(i,0);
		int v2 = faces.at<int>(i,1);
		int v3 = faces.at<int>(i,2);
		cv::Vec3f newE1(rotShape.at<float>(v2,0)-rotShape.at<float>(v1,0),rotShape.at<float>(v2,1)-rotShape.at<float>(v1,1),rotShape.at<float>(v2,2)-rotShape.at<float>(v1,2));
		cv::Vec3f newE2(rotShape.at<float>(v3,0)-rotShape.at<float>(v1,0),rotShape.at<float>(v3,1)-rotShape.at<float>(v1,1),rotShape.at<float>(v3,2)-rotShape.at<float>(v1,2));
		cv::Vec3f oldE1(rotShape0.at<float>(v2,0)-rotShape0.at<float>(v1,0),rotShape0.at<float>(v2,1)-rotShape0.at<float>(v1,1),rotShape0.at<float>(v2,2)-rotShape0.at<float>(v1,2));
		cv::Vec3f oldE2(rotShape0.at<float>(v3,0)-rotShape0.at<float>(v1,0),rotShape0.at<float>(v3,1)-rotShape0.at<float>(v1,1),rotShape0.at<float>(v3,2)-rotShape0.at<float>(v1,2));

		cv::Vec3f nNew = newE1.cross(newE2);
		cv::Vec3f nOld = oldE1.cross(oldE2);
		nNew = nNew/norm(nNew);
		nOld = nOld/norm(nOld);
		if (nNew.dot(nOld) < 0.2) {
			tex3.at<float>(v1,0) = tex3.at<float>(v2,0) = tex3.at<float>(v3,0) = 0;
		}
	}
		// Check distance to center of mass
	cv::Mat tex5 = tex * 0;
	for (int i=0;i<faces.rows;i++) {
		int v1 = faces.at<int>(i,0);
		int v2 = faces.at<int>(i,1);
		int v3 = faces.at<int>(i,2);
		cv::Vec3f cenNew(rotShape.at<float>(v1,0)+rotShape.at<float>(v2,0)+rotShape.at<float>(v3,0),rotShape.at<float>(v1,1)+rotShape.at<float>(v2,1)+rotShape.at<float>(v3,1),rotShape.at<float>(v1,2)+rotShape.at<float>(v2,2)+rotShape.at<float>(v3,2));
		cenNew = cenNew/3;
		cv::Vec3f cenOld(rotShape0.at<float>(v1,0)+rotShape0.at<float>(v2,0)+rotShape0.at<float>(v3,0),rotShape0.at<float>(v1,1)+rotShape0.at<float>(v2,1)+rotShape0.at<float>(v3,1),rotShape0.at<float>(v1,2)+rotShape0.at<float>(v2,2)+rotShape0.at<float>(v3,2));
		cenOld = cenOld/3;
		for (int k=0;k<3;k++) {
			int vv = faces.at<int>(i,k);
			cv::Vec3f tmp(rotShape.at<float>(vv,0),rotShape.at<float>(vv,1),rotShape.at<float>(vv,2));
			tmp = tmp - cenNew;
			float d1 = cv::norm(tmp);
			cv::Vec3f tmp2(rotShape0.at<float>(vv,0),rotShape0.at<float>(vv,1),rotShape0.at<float>(vv,2));
			tmp2 = tmp2 - cenOld;
			float d2 = cv::norm(tmp2);
			float val = d1/d2 * 50;
			if (val > tex5.at<float>(vv,0)) tex5.at<float>(vv,0) = val;
		}
		
	}

		// Filter the vertices in distorted triangles
	for (int i=0;i<shape.rows;i++) {
		if (tex3.at<float>(i,0) < 250 || tex5.at<float>(i,0) > 80) occStructureWeights.at<float>(i,0) = 0;
	}
		// Smooth occlusion map
	int CLOSE_SIZE = 1;
	cv::Mat occStructureWeights2 = occStructureWeights.clone();
	for (int iter = 0; iter < CLOSE_SIZE; iter++) {	
		for (int i=0;i<faces.rows;i++) {
			int v1 = faces.at<int>(i,0);
			int v2 = faces.at<int>(i,1);
			int v3 = faces.at<int>(i,2);
			if (occStructureWeights.at<float>(v1,0) < 0.75 || occStructureWeights.at<float>(v2,0) < 0.75 || occStructureWeights.at<float>(v3,0) < 0.75) {
				occStructureWeights2.at<float>(v1,0) = 0;
				occStructureWeights2.at<float>(v2,0) = 0;
				occStructureWeights2.at<float>(v3,0) = 0;
			}
		}
		occStructureWeights = occStructureWeights2.clone();
	}
	for (int iter = 0; iter < CLOSE_SIZE; iter++) {	
		for (int i=0;i<faces.rows;i++) {
			int v1 = faces.at<int>(i,0);
			int v2 = faces.at<int>(i,1);
			int v3 = faces.at<int>(i,2);
			if (occStructureWeights.at<float>(v1,0) >= 0.75 || occStructureWeights.at<float>(v2,0) >= 0.75 || occStructureWeights.at<float>(v3,0) >= 0.75) {
				occStructureWeights2.at<float>(v1,0) = 1;
				occStructureWeights2.at<float>(v2,0) = 1;
				occStructureWeights2.at<float>(v3,0) = 1;
			}
		}
		occStructureWeights = occStructureWeights2.clone();
	}

		// Symmetry enforcement based on yaw angle. BETTER TO REMOVE THIS AND IMPROVE OCCLUSION MAP FROM HASSNERS 14
	float ddx = cos(5*r[1]) * 100;
	if (ddx < 0) ddx = 0;
	for (int i=0;i<shape.rows;i++) {
		if ((r[1] > 0 &&  shape.at<float>(i,0) > ddx) || (r[1] < 0 &&  -shape.at<float>(i,0) > ddx)) {
			occStructureWeights.at<float>(i,0) = occStructureWeights.at<float>(i,0)/2;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	//////    Register the dense and sparse mesh
	// Neighbor infor
	std::vector< std::vector<int> > allNeighs;
	std::vector< std::vector<int> > allNeighFaces;
	allNeighs.resize(shape.rows);
	allNeighFaces.resize(shape.rows);
	for (int i=0;i<faces.rows;i++) {
		int v1 = faces.at<int>(i,0);
		int v2 = faces.at<int>(i,1);
		int v3 = faces.at<int>(i,2);
		if (std::find(allNeighs[v1].begin(),allNeighs[v1].end(),v2) == allNeighs[v1].end()) {
			allNeighs[v1].push_back(v2);
			allNeighs[v2].push_back(v1);
		}
		if (std::find(allNeighs[v1].begin(),allNeighs[v1].end(),v3) == allNeighs[v1].end()) {
			allNeighs[v1].push_back(v3);
			allNeighs[v3].push_back(v1);
		}
		if (std::find(allNeighs[v2].begin(),allNeighs[v2].end(),v3) == allNeighs[v2].end()) {
			allNeighs[v2].push_back(v3);
			allNeighs[v3].push_back(v2);
		}
		allNeighFaces[v1].push_back(i);
		allNeighFaces[v2].push_back(i);
		allNeighFaces[v3].push_back(i);
	}
	// Find the corresponding sparse triangle for each dense vertex using subdivision
	std::map<CvSubdiv2DPoint*, unsigned> ldmkLUT;		//LUT delaunayPt -> landmarkPt (l)
	CvMemStorage *storage = cvCreateMemStorage( 0 );
	std::vector<cv::Vec3i> subdivTries;	// Triangles (called as subdiv)
	std::vector<cv::Vec3f> subdivWeights;	// Corresponding weights
	std::vector<float> subdivMods;		// Depth residuals
	CvSubdiv2D *subDiv = cvCreateSubdivDelaunay2D( cvRect( 0, 0, refDepth.cols-1, refDepth.rows-1 ), storage);
	std::vector<int> refVertInds;		// reference sparse vertex indices
	std::vector<int> subdivFromPointInd;	
	int subdivCount = 0;
	CvPoint2D32f pt;
	for( int i = 0; i < shape.rows; i++ ) {
	  if (occStructureWeights.at<float>(i,0) >= 0.75)  
	  {
		refVertInds.push_back(i);
		pt.x = projInit[i].x;
		pt.y = projInit[i].y;
		CvSubdiv2DPoint *orgPt =cvSubdivDelaunay2DInsert( subDiv, pt );
		ldmkLUT.insert( std::pair<CvSubdiv2DPoint*, unsigned>( orgPt, subdivCount ));
		subdivCount++;
	  }
	}

	cv::Vec3f tx;
	cv::Mat vindexSubDiv = cv::Mat::zeros(mask.rows,mask.cols,CV_32S)-1;	// subdiv index map
	subdivCount = 0; 
	for( int i= 0; i<pointInd[0].size();i++){
		float depth = refDepth.at<float>(pointInd[0][i],pointInd[1][i]) + minDepth;
		CvSubdiv2DEdge e0 = 0, e;
		CvSubdiv2DPoint *sp = 0;
		std::vector<unsigned> facet;
		std::map<CvSubdiv2DPoint*, unsigned>::iterator it;
		pt.x = pointInd[1][i];
		pt.y = pointInd[0][i];				
		if( pt.x < subDiv->topleft.x || pt.y < subDiv->topleft.y ||
			pt.x >= subDiv->bottomright.x || pt.y >= subDiv->bottomright.y )
		{
			continue;
		}
		switch( cvSubdiv2DLocate( subDiv, pt, &e0, &sp )) {
			case CV_PTLOC_INSIDE:	// the dense vertex is inside a spase triangle
				e = e0;
				do
				{
					CvSubdiv2DPoint *orgPt = cvSubdiv2DEdgeOrg( e );	
					if (( it=ldmkLUT.find( orgPt )) != ldmkLUT.end() ) {
						facet.push_back( it->second );
					}

					e = cvSubdiv2DGetEdge( e, CV_NEXT_AROUND_LEFT );
				}
				while( e != e0 );

				switch( facet.size() )
				{
				case 3:
					{
					if (projInit[refVertInds[facet[0]]].x == projInit[refVertInds[facet[1]]].x) {
						int k = facet[1];
						facet[1] = facet[2];
						facet[2] = k;
					}
					int v0 = refVertInds[facet[0]];
					int v1 = refVertInds[facet[1]];
					int v2 = refVertInds[facet[2]];
					// check valid triangle (not skewed)
					if (abs(rotShapeOri.at<float>(v0,0) - rotShapeOri.at<float>(v1,0)) > 4 || abs(rotShapeOri.at<float>(v0,0) - rotShapeOri.at<float>(v2,0)) > 4 || abs(rotShapeOri.at<float>(v1,0) - rotShapeOri.at<float>(v2,0)) > 4)
						break;
					if (abs(rotShapeOri.at<float>(v0,1) - rotShapeOri.at<float>(v1,1)) > 4 || abs(rotShapeOri.at<float>(v0,1) - rotShapeOri.at<float>(v2,1)) > 4 || abs(rotShapeOri.at<float>(v1,1) - rotShapeOri.at<float>(v2,1)) > 4)
						break;
					if (abs(rotShapeOri.at<float>(v0,2) - rotShapeOri.at<float>(v1,2)) > 4 || abs(rotShapeOri.at<float>(v0,2) - rotShapeOri.at<float>(v2,2)) > 4 || abs(rotShapeOri.at<float>(v1,2) - rotShapeOri.at<float>(v2,2)) > 4)
						break;

					cv::Point2f p0 = projInit[v0];
					cv::Point2f p1 = projInit[v1];
					cv::Point2f p2 = projInit[v2];

					float A = p1.x - p0.x;
					float B = p2.x - p0.x;
					float C = p1.y - p0.y;
					float D = p2.y - p0.y;
					if (A == 0)
						printf("Errorrrrrrrrrrrrrrrrrrrrrrrrrr\n");
					if (D - C*B/A == 0)
						printf("8768768Errorrrrrrrrrrrrrrrrrrrrrrrrrr\n");
					float u = ( pt.y - p0.y - C/A *( pt.x-p0.x ))/( D - C*B/A );
					float t = ( pt.x - p0.x - B*u )/ A;
					float tmp = (1-t-u) * rotShapeOri.at<float>(v0,2) + t*rotShapeOri.at<float>(v1,2) + u*rotShapeOri.at<float>(v2,2);
					if (u<0 || t < 0 || 1-u-t<0) printf("Errrr\n");
					vindexSubDiv.at<int>(pointInd[0][i],pointInd[1][i]) = subdivCount;
					subdivFromPointInd.push_back(i);
					subdivTries.push_back(Vec3i(v0,v1,v2));
					subdivWeights.push_back(Vec3f(1-t-u,t,u));
					subdivMods.push_back(depth - tmp);
					subdivCount++;
													
					}
					break;
				case 2:
				case 1:
					break;
				default:;
				}
				break;
			case CV_PTLOC_ON_EDGE:	// the dense vertex is on a spase edge
				{
				CvSubdiv2DPoint *orgPt = cvSubdiv2DEdgeOrg( e0 );	
				if (( it=ldmkLUT.find( orgPt )) != ldmkLUT.end() ) {
					facet.push_back( it->second );
					int v0 = refVertInds[it->second];
					cv::Point2f p0 = projInit[v0];
					CvSubdiv2DPoint *dstPt = cvSubdiv2DEdgeDst( e0 );	
					if (( it=ldmkLUT.find( dstPt )) != ldmkLUT.end() ) {
						facet.push_back( it->second );
						int v1 = refVertInds[it->second];
						int countValid = 0;
						for (int kk=0;kk<allNeighs[v0].size();kk++) {
							int vv = allNeighs[v0][kk];
							if (vv == v1) {
								countValid++;
								break;
							}
						}
						if (countValid < 1) 
							if (abs(rotShapeOri.at<float>(v0,2) - rotShapeOri.at<float>(v1,2)) > 4)
								break;
						cv::Point2f p1 = projInit[v1];
						cv::Point2f p0p1 ( p1.x-p0.x, p1.y-p0.y );
						cv::Point2f p0p ( pt.x-p0.x,pt.y-p0.y );

						float dp0p1 = sqrt( p0p1.x*p0p1.x + p0p1.y*p0p1.y ); 
						float scal = p0p1.x*p0p.x/dp0p1 + p0p1.y*p0p.y/dp0p1;
						float t;
						if ( scal > 0 )	t = scal / sqrt( p0p1.x*p0p1.x + p0p1.y*p0p1.y );
						else			t = -scal / ( sqrt( p0p1.x*p0p1.x + p0p1.y*p0p1.y ) - scal );

						if (t<0 || 1-t<0) printf("Errrr\n");
						float tmp = (1-t) * rotShapeOri.at<float>(v0,2) + t*rotShapeOri.at<float>(v1,2);
						vindexSubDiv.at<int>(pointInd[0][i],pointInd[1][i]) = subdivCount;
						subdivFromPointInd.push_back(i);
						subdivTries.push_back(Vec3i(v0,v1,-1));
						subdivWeights.push_back(Vec3f(1-t,t,0));
						subdivMods.push_back(depth - tmp);
						subdivCount++;
								
					}
				}
				}
				break;
			case CV_PTLOC_VERTEX:	// the dense vertex is at a spase vertex
				{
				if (( it=ldmkLUT.find( sp )) != ldmkLUT.end() ) {
					int v0 = refVertInds[it->second];
					vindexSubDiv.at<int>(pointInd[0][i],pointInd[1][i]) = subdivCount;
					subdivFromPointInd.push_back(i);
					subdivTries.push_back(Vec3i(v0,-1,-1));
					subdivWeights.push_back(Vec3f(1,0,0));
					subdivMods.push_back(depth - rotShapeOri.at<float>(v0,2));
					subdivCount++;
				}
				}
				break;
			case CV_PTLOC_OUTSIDE_RECT:
				break;
			default:;
		}	
	}
	cvReleaseMemStorage( &storage );
	// 3D residual
	cv::Mat subMod = cv::Mat::zeros(subdivTries.size(),3,CV_32F);
	for (int i=0;i<subdivTries.size();i++) {
		subMod.at<float>(i,2) = subdivMods[i];
	}
	subMod = matR.t() * subMod.t();
	subMod = alignRMat * subMod; subMod = subMod.t();

	// Dense vertices reconstruction from the sparse mesh (for test only)
	cv::Mat sub_v(subdivTries.size(),3,CV_32F);
	for (int i=0;i<subdivTries.size();i++) {
		Vec3f tmpV(0,0,0);
		for (int k=0;k<3;k++) {
			int ind = subdivTries[i](k);
			if (ind >= 0) {
				tmpV += subdivWeights[i](k) * Vec3f(rotShape.at<float>(ind,0),rotShape.at<float>(ind,1),rotShape.at<float>(ind,2));
			}
		}
		sub_v.at<float>(i,0) = tmpV(0);
		sub_v.at<float>(i,1) = tmpV(1);
		sub_v.at<float>(i,2) = tmpV(2);
	}
	sub_v = sub_v + subMod;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////// Hole filling on the spare mesh
	std::vector<int> need_compute;
	int ncount = 0;
	int* need_compute_ind = new int[shape.rows];
	for (int i=0;i<shape.rows;i++) {
		need_compute_ind[i] = -1;
		if (filterInd[i] < 0 || occStructureWeights.at<float>(i,0)<0.75) {
			if (cv::countNonZero( faces == i ) != 0) {
				need_compute.push_back(i);
				need_compute_ind[i] = ncount; ncount++;
			}
		}
	}
	
	// A * x = b
	triA[0].clear();
	Eigen::VectorXd b_occ_depth[3], x_occ_depth[3];
	for (int k=0;k<3;k++) {
		b_occ_depth[k].resize(ncount*2);
		b_occ_depth[k] = b_occ_depth[k] * 0;
	}     
	
	IndWeight* occ_neighbors =  new IndWeight[ncount];
	IndWeight* bor_neighbors =  new IndWeight[ncount];
	int* borderInd = new int[ncount];
	int borderCount = 0;
	for (int i=0;i<ncount;i++){
		int ind = need_compute[i];
		borderInd[i] = -1;
		occ_neighbors[i].push_back(std::pair<int,double>(ind,0));
	}

	for (int i=0;i<faces.rows;i++) {
	    for (int u=0; u<3; u++) {
		int vind1 = faces.at<int>(i,u);
		int nind1 = need_compute_ind[vind1];
		if (nind1 >= 0) {
		   float ow1 = occStructureWeights.at<float>(vind1,0);
		   if (filterInd[vind1] < 0) ow1 = 0;
		   int ind1 = pairInd[vind1]-1;
		   float ow1_f = 0;
		   if (ind1 >= 0 && filterIndFlip[vind1] >= 0) ow1_f = occStructureWeights.at<float>(ind1,0);

		   int bor_ind = borderInd[nind1];
		   for (int v=0; v<3; v++) {
			if (v == u) continue;
			int vind2 = faces.at<int>(i,v);
			int nind2 = need_compute_ind[vind2];
			if (nind2 >= 0) { // Neighbor
			    IndWeight::iterator it = findByKey(occ_neighbors,nind1,vind2);
			    if (it == occ_neighbors[nind1].end()) {  // new pair
			    	occ_neighbors[nind1].push_back(std::pair<int,double>(vind2,-1));
				occ_neighbors[nind1].begin()->second += 1;

				float ow2 = occStructureWeights.at<float>(vind2,0);
				if (filterInd[vind2] < 0) ow2 = 0;
		   		int ind2 = pairInd[vind2]-1;
		   		float ow2_f = 0;
		   		if (ind2 >= 0 && filterIndFlip[vind2] >= 0) ow2_f = occStructureWeights.at<float>(ind2,0);

				if (ow1 >= 0.75 && ow2 >= 0.75) { // Both visible
				      for (int k=0;k<3;k++)
					b_occ_depth[k](nind1) += rotShape.at<float>(vind1,k) - rotShape.at<float>(vind2,k);
				}

				else if (ow1_f >= 0.75 && ow2_f >= 0.75) { // Both flip visible
				      for (int k=0;k<3;k++)
					b_occ_depth[k](nind1) += rotShapeFlip.at<float>(vind1,k) - rotShapeFlip.at<float>(vind2,k);
				}
				else {
				      for (int k=0;k<3;k++)
					b_occ_depth[k](nind1) += rotShape0.at<float>(vind1,k) - rotShape0.at<float>(vind2,k);	
				}

			    }
			}
			else { // Visible edge

			    if (bor_ind < 0) { // New border point
				bor_ind = borderCount + ncount;
				bor_neighbors[nind1].push_back(std::pair<int,double>(vind1,0));
				borderInd[nind1] = bor_ind;
				borderCount++;
			    }
			    IndWeight::iterator it = findByKey(bor_neighbors,nind1,vind2);
			    if (it == bor_neighbors[nind1].end()) {  // new pair
			    	bor_neighbors[nind1].push_back(std::pair<int,double>(vind2,0));
				bor_neighbors[nind1].begin()->second += 1;

				float w2 = 1-ow1;
				if (w2 > ow1_f) w2 = ow1_f;
				if (1) { //(ow1+w2==0) {
				    for (int k=0;k<3;k++)
					b_occ_depth[k](bor_ind) +=  (rotShape0.at<float>(vind1,k) - rotShape0.at<float>(vind2,k) + rotShape.at<float>(vind2,k));
					//b_occ_depth[k](nind1) += rotShape.at<float>(vind1,k);
				}
				else {
				    for (int k=0;k<3;k++)
					b_occ_depth[k](bor_ind) += (ow1*rotShape.at<float>(vind1,k)+w2*rotShapeFlip.at<float>(vind1,k))/(ow1+w2);
				}

			    }
			}
		   }
		}
	    }
	}
	
	for (int k=0;k<3;k++) {
		b_occ_depth[k].conservativeResize(ncount+borderCount);
		x_occ_depth[k].resize(ncount+borderCount);
		x_occ_depth[k] = x_occ_depth[k] * 0;
	}     

	for (int i=0;i<ncount;i++) {
	  for (int j=0;j<occ_neighbors[i].size();j++) {
	    if (occ_neighbors[i][j].second != 0) {
		  triA[0].push_back(SpT(i,need_compute_ind[occ_neighbors[i][j].first],occ_neighbors[i][j].second));
	    }
	  }
	}
	for (int i=0;i<ncount;i++) {
	  int bor_ind = borderInd[i];
	  if (bor_ind < 0)
		continue;
	  if (bor_neighbors[i].size() > 0) {
		triA[0].push_back(SpT(bor_ind,i,bor_neighbors[i][0].second));
	  }
	}

	SpMat A_occ(ncount+borderCount,ncount);
	A_occ.setFromTriplets(triA[0].begin(),triA[0].end());
	//Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
	Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int> > solver;
	solver.compute(A_occ);
	if(solver.info()!=Eigen::Success) {
	  // decomposition failed
	  printf("Solving failed!\n");
	}
	for (int k=0;k<3;k++){
		x_occ_depth[k] = solver.solve(b_occ_depth[k]);
	}

	for (int i=0;i<ncount;i++) {
		int vind = need_compute[i];
		rotShape.at<float>(vind,0) = x_occ_depth[0][i];
		rotShape.at<float>(vind,1) = x_occ_depth[1][i];
		rotShape.at<float>(vind,2) = x_occ_depth[2][i];
	}
	if (outSet.sparseFullFrontal) {
		sprintf(text,"%s_sparse_full_frontal.ply",out_prefix.c_str());
		write_plyShapeFloat(text,rotShape,faces);
	}
	// Dense vertices from the sparse ones
	for (int i=0;i<subdivTries.size();i++) {
		Vec3f tmpV(0,0,0);
		for (int k=0;k<3;k++) {
			int ind = subdivTries[i](k);
			if (ind >= 0) {
				tmpV += subdivWeights[i](k) * Vec3f(rotShape.at<float>(ind,0),rotShape.at<float>(ind,1),rotShape.at<float>(ind,2));
			}
		}
		sub_v.at<float>(i,0) = tmpV(0);
		sub_v.at<float>(i,1) = tmpV(1);
		sub_v.at<float>(i,2) = tmpV(2); //+subdivMods[i];
	}
	sub_v = sub_v + subMod;

	// Soft-symmetry on the dense mesh
	delete filterInd;
	filterInd = new int[subdivTries.size()];
	cv::Mat vindexSubDivFlip = vindexSubDiv.clone();
	cv::Mat sub_v_flip = sub_v.clone();
	for (int i=0;i<subdivTries.size();i++) {
		filterInd[i] = 1;
		Vec3f tmpV(0,0,0);
		for (int k=0;k<3;k++) {
			int ind0 = subdivTries[i](k);
			if (ind0 >= 0) {
				int ind = pairInd[ind0]-1;
				if (ind < 0) {
					filterInd[i] = -1;
					int tmp_ind = subdivFromPointInd[i];
					vindexSubDivFlip.at<int>(pointInd[0][tmp_ind],pointInd[1][tmp_ind]) = -1;
					k = 5;
					continue;
				}
				if ( occStructureWeights.at<float>(ind,0) >= 0.75 )
				{
					int tmp_ind = subdivFromPointInd[i];
					vindexSubDivFlip.at<int>(pointInd[0][tmp_ind],pointInd[1][tmp_ind]) = -1;
					filterInd[i] = 100;
				}
				if (subdivWeights[i](k) < 0 || subdivWeights[i](k) > 1) printf("Error\n");
				tmpV += subdivWeights[i](k) * Vec3f(rotShape.at<float>(ind,0),rotShape.at<float>(ind,1),rotShape.at<float>(ind,2));
			}
		}
		sub_v_flip.at<float>(i,0) = tmpV(0);
		sub_v_flip.at<float>(i,1) = tmpV(1);
		sub_v_flip.at<float>(i,2) = tmpV(2);
	}

	for (int i=0;i<subMod.rows;i++) {
		subMod.at<float>(i,0) = -subMod.at<float>(i,0);
		for (int k=0;k<3;k++) {
			if (subMod.at<float>(i,k) > 0.2) subMod.at<float>(i,0) = 0.5;
			if (subMod.at<float>(i,k) < -0.2) subMod.at<float>(i,0) = -0.5;
		}
	}
	sub_v_flip = sub_v_flip + subMod;

	//////////////////////////////////////////////////////////////////////////////////
	/////////////////   Mesh combining and zippering
	int* keepDense = new int[sub_v.rows];
	int* keepDenseFlip = new int[sub_v_flip.rows];
	for (int i=0;i<sub_v.rows;i++) keepDense[i] = 0; //inner
	for (int i=0;i<sub_v.rows;i++) keepDenseFlip[i] = (filterInd[i] == 1) - 1; //inner

	//// Look for the border of the need-to-keep regions on the sparse mesh
	int* borderTypeToSub = new int[shape.rows];
	int* borderTypeToSubFlip = new int[shape.rows];
	for (int i=0;i<shape.rows;i++) {
		borderTypeToSub[i] = -1;
		borderTypeToSubFlip[i] = -1;
	}

	// Check border
	tex = tex * 0;
	for (int i=0;i<shape.rows;i++) {
		if (occStructureWeights.at<float>(i,0) >= 0.75) {
			int y = floor(projInit[i].y);
			int x = floor(projInit[i].x);
			int countGood = 0;
			if (vindexSubDiv.at<int>(y,x) >= 0) countGood++;
			if (vindexSubDiv.at<int>(y+1,x) >= 0) countGood++;
			if (vindexSubDiv.at<int>(y,x+1) >= 0) countGood++;
			if (vindexSubDiv.at<int>(y+1,x+1) >= 0) countGood++;
			if (countGood >= 3) {
				borderTypeToSub[i] = 0; // inner point
				tex.at<float>(i,0) = 255;
			}
		}
		else {
			int ind = pairInd[i]-1;
			if (ind >= 0  && occStructureWeights.at<float>(ind,0) >= 0.75 )
			{
				int y = floor(projInit[ind].y);
				int x = floor(projInit[ind].x);
				int countGood = 0;
				if (vindexSubDivFlip.at<int>(y,x) >= 0) countGood++;
				if (vindexSubDivFlip.at<int>(y+1,x) >= 0) countGood++;
				if (vindexSubDivFlip.at<int>(y,x+1) >= 0) countGood++;
				if (vindexSubDivFlip.at<int>(y+1,x+1) >= 0) countGood++;
				if (countGood >= 3) {
					borderTypeToSubFlip[i] = 0; // inner point
					tex.at<float>(i,0) = 127;
				}
				
			}
		}
	}
	for (int i=0;i<shape.rows;i++) {
		if (borderTypeToSub[i] == -1) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSub[ind] == 0) {
					borderTypeToSub[i] = 1; // border
					tex.at<float>(i,1) = 255; break;
				}
			}
		}
		if (borderTypeToSubFlip[i] == -1) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSubFlip[ind] == 0) {
					borderTypeToSubFlip[i] = 1; // border
					tex.at<float>(i,1) = 127; break;
				}
			}
		}
	}

	for (int i=0;i<shape.rows;i++) {
		if (borderTypeToSub[i] == 0) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSub[ind] == 1) {
					borderTypeToSub[i] = 2; // next to border
					tex.at<float>(i,2) = 255; break;
				}
			}
		}
		if (borderTypeToSubFlip[i] == 0) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSubFlip[ind] == 1) {
					borderTypeToSubFlip[i] = 2; // next to border
					tex.at<float>(i,2) = 127; break;
				}
			}
		}
	}

	for (int i=0;i<shape.rows;i++) {
		if (borderTypeToSub[i] == 0) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSub[ind] == 2) {
					borderTypeToSub[i] = 3; // next next to border
					tex.at<float>(i,1) = 255; 
					tex.at<float>(i,2) = 255; break;
				}
			}
		}
		if (borderTypeToSubFlip[i] == 0) {
			for (int k=0;k<allNeighs[i].size();k++) {
				int ind = allNeighs[i][k];
				if (borderTypeToSubFlip[ind] == 2) {
					borderTypeToSubFlip[i] = 3; // next next to border
					tex.at<float>(i,1) = 127; 
					tex.at<float>(i,2) = 127; break;
				}
			}
		}
	}

	delete filterInd;
	filterInd = new int[shape.rows];
	for (int i=0;i<shape.rows;i++) {
		if (borderTypeToSub[i] != 0 && borderTypeToSubFlip[i] != 0)
			filterInd[i] = 1;
		else
			filterInd[i] = -1; // inner point
	}
	filterFaces.clear();
	for (int i=0;i<faces.rows;i++){
		if (filterInd[faces.at<int>(i,0)]>=0 && filterInd[faces.at<int>(i,1)]>=0 && filterInd[faces.at<int>(i,2)]>=0)
			filterFaces.push_back(i);
	}
	fFaces = cv::Mat::zeros(filterFaces.size(),3,faces.type());

	for (int i=0;i<filterFaces.size();i++){
		fFaces.at<int>(i,0) = faces.at<int>(filterFaces[i],0);
		fFaces.at<int>(i,1) = faces.at<int>(filterFaces[i],1);
		fFaces.at<int>(i,2) = faces.at<int>(filterFaces[i],2);
	}


	// Update keepDense
        for (int i=0;i<faces.rows;i++) {
		if (filterInd[faces.at<int>(i,0)]>=0 && filterInd[faces.at<int>(i,1)]>=0 && filterInd[faces.at<int>(i,2)]>=0) continue;
		Vec3i inds(faces.at<int>(i,0),faces.at<int>(i,1),faces.at<int>(i,2));
		float minX, minY, maxX, maxY;
		minX = minY = 100000000000000000;
		maxX = maxY = -1;
		Vec3f a(projInit[inds[0]].x, projInit[inds[0]].y, 0);
		Vec3f b(projInit[inds[1]].x, projInit[inds[1]].y, 0);
		Vec3f c(projInit[inds[2]].x, projInit[inds[2]].y, 0);

		for (int k=0;k<3;k++){
			if (projInit[inds[k]].x <  minX) minX = projInit[inds[k]].x;
			if (projInit[inds[k]].x >  maxX) maxX = projInit[inds[k]].x;
			if (projInit[inds[k]].y <  minY) minY = projInit[inds[k]].y;
			if (projInit[inds[k]].y >  maxY) maxY = projInit[inds[k]].y;
		}
		for (int y = ceil(minY); y <= maxY; y++) {
			for (int x = ceil(minX); x <= maxX; x++) {
				Vec3f p(x, y, 0);
				if (PointInTriangle(p,a,b,c)) {
					//printf("remove %d %d\n", y, x);
					int vind = vindexSubDiv.at<int>(y,x);
					if (vind >= 0) {
						keepDense[vind] = 1;
					}
				}
			}
		}
	}


	// Update keepDenseFlip
        for (int i=0;i<faces.rows;i++) {
		int v0 = pairInd[faces.at<int>(i,0)]-1;
		int v1 = pairInd[faces.at<int>(i,1)]-1;
		int v2 = pairInd[faces.at<int>(i,2)]-1;
		if (v0>=0 && filterInd[v0]>=0 && v1>=0 && filterInd[v1]>=0 && v2>=0 && filterInd[v2]>=0) continue;
		Vec3i inds(faces.at<int>(i,0),faces.at<int>(i,1),faces.at<int>(i,2));
		float minX, minY, maxX, maxY;
		minX = minY = 100000000000000000;
		maxX = maxY = -1;
		Vec3f a(projInit[inds[0]].x, projInit[inds[0]].y, 0);
		Vec3f b(projInit[inds[1]].x, projInit[inds[1]].y, 0);
		Vec3f c(projInit[inds[2]].x, projInit[inds[2]].y, 0);

		for (int k=0;k<3;k++){
			if (projInit[inds[k]].x <  minX) minX = projInit[inds[k]].x;
			if (projInit[inds[k]].x >  maxX) maxX = projInit[inds[k]].x;
			if (projInit[inds[k]].y <  minY) minY = projInit[inds[k]].y;
			if (projInit[inds[k]].y >  maxY) maxY = projInit[inds[k]].y;
		}
		for (int y = ceil(minY); y <= maxY; y++) {
			for (int x = ceil(minX); x <= maxX; x++) {
				Vec3f p(x, y, 0);
				if (PointInTriangle(p,a,b,c)) {
					//printf("remove %d %d\n", y, x);
					int vind = vindexSubDivFlip.at<int>(y,x);
					if (vind >= 0 && keepDenseFlip[vind] >= 0) {
						keepDenseFlip[vind] = 1;
					}
				}
			}
		}
	}

	cv::Mat sub_v_tex = sub_v * 0;
	for (int i=0;i<sub_v.rows;i++) {
		if (keepDense[i] == 0) {
			int y = pointInd[0][subdivFromPointInd[i]];
			int x = pointInd[1][subdivFromPointInd[i]];
			for (int sy=-1; sy <=1;sy++) {
				for (int sx=-1; sx <=1;sx++) {
					if (sx != 0 || sy != 0) {
						int vind = vindexSubDiv.at<int>(y+sy,x+sx);
						if (vind >= 0 && keepDense[vind] == 1) {
							keepDense[i] = 2; break;
						}
			
					}
				}
			}
		}
	}
	for (int i=0;i<sub_v.rows;i++) {
		sub_v_tex.at<float>(i,0) = 100*keepDense[i];
		if (keepDense[i] == 0)  {
			int y = pointInd[0][subdivFromPointInd[i]];
			int x = pointInd[1][subdivFromPointInd[i]];
			vindexSubDiv.at<int>(y,x) = -1;
		}
	}

	std::vector<Vec3i> fac_sub;
	computeFaces(vindexSubDiv,fac_sub);
	cv::Mat fac_sub2 = cv::Mat::zeros(fac_sub.size(),3,CV_32S);
	for (int i=0;i<fac_sub.size();i++){
		fac_sub2.at<int>(i,0) = fac_sub[i](0);
		fac_sub2.at<int>(i,1) = fac_sub[i](1);
		fac_sub2.at<int>(i,2) = fac_sub[i](2);
	}


	sub_v_tex = sub_v * 0;
	for (int i=0;i<sub_v.rows;i++) {
		if (keepDenseFlip[i] == 0) {
			int y = pointInd[0][subdivFromPointInd[i]];
			int x = pointInd[1][subdivFromPointInd[i]];
			for (int sy=-1; sy <=1;sy++) {
				for (int sx=-1; sx <=1;sx++) {
					if (sx != 0 || sy != 0) {
						int vind = vindexSubDivFlip.at<int>(y+sy,x+sx);
						if (vind >= 0 && keepDenseFlip[vind] == 1) {
							keepDenseFlip[i] = 2; break;
						}
			
					}
				}
			}
		}
	}
	for (int i=0;i<sub_v.rows;i++) {
		sub_v_tex.at<float>(i,0) = 100*keepDenseFlip[i];
		if (keepDenseFlip[i] == 0)  {
			int y = pointInd[0][subdivFromPointInd[i]];
			int x = pointInd[1][subdivFromPointInd[i]];
			vindexSubDivFlip.at<int>(y,x) = -1;
		}
	}


	computeFaces(vindexSubDivFlip,fac_sub);
	cv::Mat fac_sub2_flip = cv::Mat::zeros(fac_sub.size(),3,CV_32S);
	for (int i=0;i<fac_sub.size();i++){
		fac_sub2_flip.at<int>(i,0) = fac_sub[i](1);
		fac_sub2_flip.at<int>(i,1) = fac_sub[i](0);
		fac_sub2_flip.at<int>(i,2) = fac_sub[i](2);
	}
	//sprintf(text,"%s_denseflip_full_frontal.ply",out_prefix.c_str());
	//write_plyShapeFloat(text,sub_v_flip,fac_sub2_flip);

	// Combine meshes
	cv::Mat final_v_tmp;
	fac_sub2 += shape.rows; 
	fac_sub2_flip += sub_v.rows + shape.rows; 
	if (sub_v_flip.rows > 0) cv::vconcat(sub_v,sub_v_flip,final_v_tmp);
	else final_v_tmp = sub_v.clone();
	cv::vconcat(rotShape,final_v_tmp,sub_v); 
	if (fac_sub2_flip.rows > 0) cv::vconcat(fac_sub2,fac_sub2_flip,final_v_tmp);
	else final_v_tmp = fac_sub2.clone();
	cv::vconcat(fFaces,final_v_tmp,fac_sub2);

	if (outSet.finalFrontal) {
		sprintf(text,"%s_final_frontal.ply",out_prefix.c_str());
		write_plyShapeFloat(text,sub_v,fac_sub2);
	}


	return true;
}

// Compute occlusion score for the sparse mesh inside the renderer (Hassners 14)
cv::Mat FaceServices2::computeOccMat(FImRenderer* im_render, float* r, float *t){
	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;
        int height = im_render->img_.rows;
        int width = im_render->img_.cols;

	cv::Mat refRGB1 = cv::Mat::zeros(height, width,CV_8UC3);
	cv::Mat refDepth1 = cv::Mat::zeros(height, width,CV_32F);
	cv::Mat refRGB2 = cv::Mat::zeros(height, width,CV_8UC3);
	cv::Mat refDepth2 = cv::Mat::zeros(height, width,CV_32F);
	float refR[3] = {0.000001, 0.000001, 0.000001};
	float refT[3] = {0.000001, 0.000001, 0.000001};
	refT[2] = t[2];

	im_render->render(refR,refT,_k[4],refRGB1,refDepth1);
	im_render->render(r,t,_k[4],refRGB2,refDepth2);
	cv::Mat counts = cv::Mat::zeros(height, width,CV_8U);

	std::vector<cv::Point3f> point3Ds;
	for (int r = 0; r < height; r++) {
	    for (int c = 0; c < width; c++) {
		float d1 = refDepth1.at<float>(r,c); 
		if (d1>=0.9999) continue;
		d1 = - zNear_*zFar_   / ( zFar_ - d1 * ( zFar_ - zNear_ ));
		float X1 = (c - _k[2]) * d1 / _k[0];
		float Y1 = (r - _k[5]) * d1 / _k[4];
		point3Ds.push_back(Point3f(X1, Y1, d1));
	    }
	}

	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat rVec_ref( 3, 1, CV_32F, refR );
	cv::Mat rVec( 3, 1, CV_32F, r );
	cv::Mat tVec_ref( 3, 1, CV_32F, refT );
	cv::Mat tVec( 3, 1, CV_32F, t );	
	int nV = point3Ds.size();
	cv::Mat rMat;
	cv::Rodrigues(rVec, rMat);
	cv::Mat rMat_ref;
	cv::Rodrigues(rVec_ref, rMat_ref);
	cv::Mat shape( 3, nV, CV_32F );
	for (int i=0;i<nV;i++){
		Point3f tmp = point3Ds[i];
		shape.at<float>(0,i) = tmp.x;
		shape.at<float>(1,i) = tmp.y;
		shape.at<float>(2,i) = tmp.z;
	}
	cv::Mat new3D = (rMat* rMat_ref.t())* shape + cv::repeat(-rMat* rMat_ref.t() *tVec_ref + tVec,1,nV);
		
	for (int i=0;i<nV;i++){
		float Z = new3D.at<float>(2,i);
		float x = -new3D.at<float>(0,i)/Z*_k[4] + _k[2];
		float y = new3D.at<float>(1,i)/Z*_k[4] + _k[5];
		if (x > 0 && y > 0 & x < refDepth2.cols-1 && y <refDepth2.rows-1) {
			bool visible = false;
			for (int dx =-1;dx<2;dx++){
				for (int dy =-1;dy<2;dy++){
					float dd = refDepth2.at<float>(y+dy,x+dx);
					dd = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
					if (fabs(Z - dd) < 2.5){
						visible = true;
						break;
					}
				}
				if (visible) break;
			}
			if (visible) {
				counts.at<unsigned char>((int)y,(int)x) ++;
			}
			
		}
	}
	subtract(counts, Mat::ones(counts.rows, counts.cols, counts.type()), counts);
	max(counts, Mat::zeros(counts.rows, counts.cols, counts.type()), counts);
	int guassianBlurSize = min(5, (width/24)*2 + 1);
	GaussianBlur(counts, counts, Size(guassianBlurSize, guassianBlurSize), 0); 

	cv::Mat out;
	cv::Mat out_tmp;
	counts.convertTo(out_tmp,CV_32F);
	cv::exp(-0.2*out_tmp, out);
	return out;
}

// Compute visibility based on projection
std::vector<cv::Point2f> FaceServices2::projectCheckVis2(FImRenderer* im_render, cv::Mat shape, float* r, float *t, cv::Mat refDepth, bool* &visible){
	float zNear_ = im_render->zNear;
	float zFar_ = im_render->zFar;

	cv::Mat k_m( 3, 3, CV_32F, _k );
	std::vector<cv::Point2f> out;
	int nV = shape.rows;
	if (visible == 0) visible = new bool[shape.rows];
	cv::Mat rVec( 3, 1, CV_32F, r );
	cv::Mat tVec( 3, 1, CV_32F, t );
	cv::Mat rMat;
	cv::Rodrigues(rVec, rMat);
	cv::Mat new3D = rMat * shape.t() + cv::repeat(tVec,1,nV);

	for (int i=0;i<nV;i++){
		visible[i] = false;
		float Z = new3D.at<float>(2,i);
		float x = -new3D.at<float>(0,i)/Z*_k[4] + _k[2];
		float y = new3D.at<float>(1,i)/Z*_k[4] + _k[5];
		out.push_back(cv::Point2f(x,y));
		if (x > 0 && y > 0 & x < refDepth.cols-1 && y <refDepth.rows-1) {
			for (int dx =-1;dx<2;dx++){
				for (int dy =-1;dy<2;dy++){
					float dd = refDepth.at<float>(y+dy,x+dx);
					dd = - zNear_*zFar_   / ( zFar_ - dd * ( zFar_ - zNear_ ));
					if (fabs(Z - dd) < 2.5){
						visible[i] = true;
					}
				}
			}
		}
	}

	return out;
}

// Update occlusion score
cv::Mat FaceServices2::computeOccScores(cv::Mat shape, bool* visible, cv::Mat occMat, float* r, float *t){
	cv::Mat k_m( 3, 3, CV_32F, _k );
	cv::Mat distCoef = cv::Mat::zeros( 1, 4, CV_32F );
	cv::Mat rVec(3,1,CV_32F,r);
	cv::Mat tVec(3,1,CV_32F,t);

	std::vector<Point2f> projPoints;
	projectPoints(shape,rVec,tVec,k_m,distCoef,projPoints);
	cv::Mat out = cv::Mat::zeros(shape.rows,1,CV_32F);
	for (int i=0;i<shape.rows;i++){
		if (visible[i]) {
			float val = -1;
			int y = projPoints[i].y;
			int x = projPoints[i].x;
			for (int sx = -5; sx <= 5; sx++) {
			   if (sx+x >= 0 && sx+x < occMat.cols) {
				for (int sy = -5; sy <= 5; sy++) {
				   if (sy+y >= 0 && sy+y < occMat.rows) {
					if (val < 0 || val > occMat.at<float>(sy+y, sx+x))
						val = occMat.at<float>(sy+y,sx+x);
				   }
				} 
			   }
			} 
			out.at<float>(i,0) = (val>0)?val:0;	
		}
	}
	return out;
}

// Check if p1 & p2 are on the the same side of the line ab
bool FaceServices2::SameSide(Vec3f p1,Vec3f p2,Vec3f a,Vec3f b) {
    Vec3f cp1 = (b-a).cross(p1-a);
    Vec3f cp2 = (b-a).cross(p2-a);
    return cp1.dot(cp2) >= 0;
}

// Check if p is inside the triangle abc
bool FaceServices2::PointInTriangle(Vec3f p,Vec3f a,Vec3f b,Vec3f c) {
    return SameSide(p,a,b,c) && SameSide(p,b, a,c)&& SameSide(p,c, a,b);
}

// Check if point p is in the line ab
bool FaceServices2::PointOnEdge(Vec3f p,Vec3f a,Vec3f b) {
    Vec3f cp1 = (b-a).cross(p-a);
    return norm(cp1) == 0;
}

// Estimate 3D regid transformation to convert p1 (3xN) to p2 (3xN)
bool FaceServices2::estimateRigidTranf(cv::Mat p1, cv::Mat p2, cv::Mat &matR, cv::Mat &t3D){
	// p2 = R*p1 + t
	float pP[9] = {-1, 0, 0, 0, 1, 0, 0 ,0, -1};
	cv::Mat matP(3,3,CV_32F,pP);
	cv::Mat pp1 = p1;
	cv::Mat pp2 = p2;
	if (pp1.rows != 3 && pp1.cols == 3) pp1 = p1.t();
	if (pp2.rows != 3 && pp2.cols == 3) pp2 = p2.t();
	if (pp1.rows != 3 || pp2.rows != 3 || pp1.cols != pp2.cols) return false;
	Mat p1M;
	reduce(pp1,p1M,1,CV_REDUCE_AVG);
	cv::Mat p1_c = pp1 - repeat(p1M,1,pp1.cols);
	Mat p2M;
	reduce(pp2,p2M,1,CV_REDUCE_AVG);
	cv::Mat p2_c = pp2 - repeat(p2M,1,pp2.cols);

	cv::Mat H = p1_c*p2_c.t();
	cv::Mat u,s,v;
	SVD::compute(H,s,u,v);
	v = v.t()*matP;
	u = u*matP;
	matR = v*u.t();
	if (cv::determinant(matR) < 0){
		matR.at<float>(0,2) = -matR.at<float>(0,2);
		matR.at<float>(1,2) = -matR.at<float>(1,2);
		matR.at<float>(2,2) = -matR.at<float>(2,2);
	}
	t3D = - matR*p1M + p2M;
	return true;
}

// Compute triangles from the vertex index grid
void FaceServices2::computeFaces(cv::Mat findex, vector<cv::Vec3i> &updated_faces){
	int index[3];
	updated_faces.clear();
	for( int r= 0; r< findex.rows - 1; r += 1 ) {
		for( int c= 0; c< findex.cols - 1; c += 1 ) {
			index[0] = findex.at<int>(r,c+1);
			if (index[0] >= 0) {
				index[1] = findex.at<int>(r,c);
				index[2] = findex.at<int>(r+1,c);
				if (index[1] >= 0 && index[2] >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],index[1],index[2]));

				index[1] = findex.at<int>(r+1, c);
				index[2] = findex.at<int>(r+1, c+1);
				if (index[1] >= 0 && index[2] >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],index[1],index[2]));

				if (findex.at<int>(r+1,c) < 0 && findex.at<int>(r,c) >= 0 && findex.at<int>(r+1,c+1) >= 0)
					updated_faces.push_back(cv::Vec3i(index[0],findex.at<int>(r,c),findex.at<int>(r+1,c+1)));
			}
			else
				if (findex.at<int>(r,c+1) >= 0 && findex.at<int>(r+1,c) >= 0 && findex.at<int>(r+1,c+1) >= 0)
					updated_faces.push_back(cv::Vec3i(findex.at<int>(r,c+1),findex.at<int>(r+1,c),findex.at<int>(r+1,c+1)));
		}
	}
}


IndWeight::iterator FaceServices2::findByKey(IndWeight *list, int index,int key){
	for (IndWeight::iterator it = list[index].begin(); it != list[index].end(); it++){
		if (it->first == key) {
			return it;
		}
	}
	//printf("Nokey %d %d\n",index,key); //getchar();
	return list[index].end();
}

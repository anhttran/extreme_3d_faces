#!/bin/bash
cd ../bin
ls ../data/paper/imgs/*.png > paperImages.txt
./TestBump -batch paperImages.txt ../data/paper/3D/ ../data/paper/shape ../data/paper/bump_blended ../data/paper/bump_blended ../3DMM_model/BaselFaceModel_mod.h5 ../dlib_model/shape_predictor_68_face_landmarks.dat ../data/paper/imgs
cd ../demoCode

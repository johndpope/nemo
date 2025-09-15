#!/bin/bash

# Install nemo-specific dependencies
echo "Installing nemo-specific dependencies..."
pip install gdown
pip install lpips pytorch_msssim
pip install git+https://github.com/Ahmednull/L2CS-Net.git
gdown 1TjTWn35pLNnjB8VtVpiUa6z0kI1Lg_Gs
gdown 1BLzbKD36lrBnRP4t3WDHI0_aRPJb_BFr


unzip repos.zip 
unzip logs.zip
cd losses
gdown 1YYq-YhuvJJSEVzBmwrgm7D91cDVCnLvX
gdown 12dvzVVtzwlno7kWTIz0aOZfaMsNV0R-J

unzip loss_model_weights.zip -d ./
unzip gaze_models.zip -d ./


git clone https://github.com/hhj1897/face_detection.git
cd face_detection
pip install -e .
cd ..


git clone https://github.com/ibug-group/roi_tanh_warping.git
cd roi_tanh_warping
pip install -e .

cd ..
git clone https://github.com/hhj1897/face_parsing.git
cd face_parsing
pip install -e .



cd ibug/face_parsing/rtnet/weights

echo "Downloading face parsing weights..."
# Download the actual weight files (not Git LFS pointers)
wget -O rtnet50-fcn-11.torch https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet50-fcn-11.torch
wget -O rtnet50-fcn-14.torch https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch
wget -O rtnet101-fcn-14.torch https://media.githubusercontent.com/media/hhj1897/face_parsing/master/ibug/face_parsing/rtnet/weights/rtnet101-fcn-14.torch

echo "✓ Face parsing weights downloaded successfully!"



# PFPMA
This repository contains the code for our paper:<br>

J. Sun, H. Yu, and J. Zhao, "Generating Adversarial Examples Using Parameter-
Free Penalty Method", In Proc. 24th IEEE International Conference on Software Security and Relia-
bility,    Fault Prediction, Prevention, Detection, and Reliability Enhancement Workshop, Cambridge,
United Kingdom, July 1-5, 2024


### Requirements
- Python3.11
- pytorch
- torchvision
- numpy
- matplotlib
- torchattacks https://github.com/Harry24k/adversarial-attacks-pytorch
- robustbench https://github.com/RobustBench/robustbench
- Adversarial-library https://github.com/jeromerony/adversarial-library

Results of MNIST in Table 3,4 can be reproduced by running  ```python experiment_PFPMA_MNIST.py```.<br>
Results of CIFAR10 in Table 5,6 can be reproduced by running  ```python experiment_PFPMA_CIFAR.py```.<br>
Results of ImageNet in Table 7,8 can be reproduced by running  ```python experiment_PFPMA_ImageNet.py```.<br>
Running results is in ```./result```.<br>
Three models of MNIST are in the directory ```./models/mnist```. 
Models for CIFAR and ImageNet will be downloaded automatically by robustbench package. MNIST and CIFAR datasets will also be downloaded automatically and the images used for ImageNet(ILSVRC2012) are the first 1000 images in validation set. <br>

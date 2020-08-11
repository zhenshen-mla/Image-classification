# Image-classification

  Experiments on cifar-10 with torch.         
  
## Introduction
  In this repo, we experiment on cifar-10 dataset[1] using three kinds of network structures: VGG[2], ResNet[3] and GoogLeNet[4].   

## Models
  * `/models/vgg.py`: implementation of VGG11;  
  * `/models/resnet.py`: implementation of ResNet18;  
  * `/models/googlenet.py`: implementation of GoogLeNet v1;  
  * `/load_weights.py`: load the pre-training model for testing;  
  * `/main_vgg.py`: train the VGG11 network;  
  * `/main_resnet.py`: train the ResNet18 network;  
  * `/main_google.py`: train the GoogLeNet v1 network;  
## Requirements  

  Python >= 3.6  
  numpy  
  PyTorch >= 1.0  
  torchvision  
  tensorboardX  
  sklearn  
  

## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/zhenshen-mla/Series-Photo-Selection.git   
    ```   
    ```
    cd Series-Photo-Selection  
    ```
  2. For custom dependencies:   
    ```
    pip install matplotlib tensorboardX sklearn   
    ```
## Usage   
  1. Download the dataset([Automatic triage for a photo series](https://phototriage.cs.princeton.edu/dataset.html)) and configure the data path.   
  2. Train the baseline with ResNet backbone:  
  ``` python train_resnet.py ```  
  3. Train the network with PAUnit:  
  ``` python train_pau.py ```  
  
## References
  [1] cifar-10 dataset: http://www.cs.toronto.edu/~kriz/cifar.html.  
  [2] Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition."   
  [3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition."    
  [4] Christian Szegedy, Wei Liu, Yangqing Jia. "Going deeper with convolutions."  

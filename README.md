# Image-classification

  Experiments on cifar10 with torch.         
  
## Introduction
  In this repo, we experiment on cifar10 dataset using three kinds of network structures: VGG, ResNet and GoogleNet.   

## Models
  * `/models/PAUnit.py`: implementation of Parallel Attention Unit;  
  * `/models/ResNet18.py`: baseline with resnet18 backbone;  
  * `/models/ResNet50.py`: baseline with resnet50 backbone;  
  * `/models/ResNet101.py`: baseline with resnet101 backbone;  
  * `/models/Inceptionv3.py`: baseline with Inception network;  
  * `/models/PAUnit.py`: resnet50 network with PAUnit;   
  
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

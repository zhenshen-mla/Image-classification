# Image-classification

  Experiments on cifar-10 with torch.         
  
## Introduction
  In this repo, we experiment on cifar-10 dataset [1] using three kinds of network structures: VGG [2], ResNet [3] and GoogLeNet [4].   
  VGG特点：相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少），并且加深了网络深度。  
  优点：（1）VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。（2）几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好。（3）验证了通过不断加深网络结构可以提升性能。  

## Files
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
  

## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/zhenshen-mla/Image-classification.git   
    ```   
    ```
    cd Image-classification  
    ```
  2. For custom dependencies:   
    ```
    pip install tensorboardX   
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

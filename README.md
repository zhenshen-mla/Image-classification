# Image-classification

  Experiments on cifar-10 with torch.         
  
## Introduction
  In this repo, we experiment on cifar-10 dataset [1] using three kinds of network structures: VGG [2], ResNet [3] and GoogLeNet [4].   

## Comparison of Models
  **VGG**  
  特点：相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少），并且加深了网络深度。  
  优点：（1）VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。（2）几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好。（3）验证了通过不断加深网络结构可以提升性能。  
  缺点：VGG耗费更多计算资源，并且使用了更多的参数（fc层）。  
  **ResNet**  
  特点：ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习。  
  **GoogLeNet**  
  **v1**  
  (1) 在3x3，5x5的卷积前面用1x1进行降维，可以减少参数量。（2）Inception内部是不同感受野的分支进行计算，然后对输出进行concatenate操作，从而产生稠密数据。  
  **v2**  
  （1）考虑使用小卷积替代大卷积，即使用1x1，3x3代替5x5，甚至使用1x1代替3x3。（2）考虑1xn，nx1的联合使用来代替nxn。（3）在v2前向传播中，先使用v1的inception结构，再使用（1）中思想，末端使用（2）中思想。  
  **v3**  
  v3是对v2中的nxn进一步分解，将一个卷积拆为两个，并添加relu，网络进一步加深并增加了非线性。  
  **v4**  
  v4就是利用残差连接（Residual Connection)来改进v3。  

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
  3. Users can download pre-trained models through [Baidu Netdisk](https://pan.baidu.com/s/1K0uje4jHRzlOHlgW7jSxPg)(```xsza```) to /Image-classification/weight/ directory.     
## Usage   
  1. When the following command is executed, the cifar dataset is automatically downloaded and saved in directory /Image-classification/data/, and the network weights are saved in directory /Image-classification/weight/ during training.   
  2. Train the models in lr_step strategies:  
  ``` python main_vgg.py ``` ``` python main_resnet.py ``` ``` python main_google.py ```   
  3. And you can use **poly** strategy in **/lr_scheduler.py** for training.  
  
## References
  [1] cifar-10 dataset: http://www.cs.toronto.edu/~kriz/cifar.html.  
  [2] Karen Simonyan, Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition."   
  [3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition."    
  [4] Christian Szegedy, Wei Liu, Yangqing Jia. "Going deeper with convolutions."  
  [5] VGG: https://zhuanlan.zhihu.com/p/41423739.  
  [6] ResNet: https://zhuanlan.zhihu.com/p/31852747.  
  [7] GoogLeNet: https://zhuanlan.zhihu.com/p/104671625.  
  [8] https://github.com/kuangliu/pytorch-cifar

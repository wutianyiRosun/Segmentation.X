# SemSegPaddle: A Paddle-based Framework for Deep Learning in Semantic Segmentation

This is a Paddle implementation of semantic segmentation models on muiltiple datasest, including Cityscapes, Pascal Context, and ADE20K.

## Updates

- [2020/01/08] We release ***PSPNet-ResNet101-Cityscapes*** and ***GloRe-ResNet101-PascalContext*** models.

## Highlights

Synchronized Batch Normlization is important for segmenation.
  - The implementation is easy to use as it is pure-python, no any C++ extra extension libs.
   
  - Paddle provides sync_batch_norm.
   
   
## Support models

We split our models into backbone and decoder network, where backbone network are transfered from classification networks.

Backbone:
  - ResNet
  - ResNeXt
  - HRNet
  - Xception
  - EfficientNet
  
Decoder:
  - PSPNet
  - DeepLabv3
  - GloRe
  - GINet
  


## Peformance

 - Performance of Cityscapes validation set.

Method    | Backbone   | lr     | BatchSize  | epoch    | mean IoU (SS) |  mean IoU (MS)  |   Training Time  |
----------|:----------:|:------:|:----------:|:--------:|:-------------:|:---------------:|:-----------------|
PSPNet    | resnet101  | 0.01   |   8       | 80        | 78.1          |                 |     9.5 h        |

 - Performance of Pascal-context validation set.

Method    | Backbone   | lr     | BatchSize  | epoch    | mean IoU (SS) |  mean IoU (MS)  |   Training Time  |
----------|:----------:|:------:|:----------:|:--------:|:-------------:|:---------------:|:-----------------|
GloRe     | resnet101  | 0.01   |   8        | 80        |               |                 |                 |


## Environment

This repo is developed under the following configurations:

 - Hardware: 4 GPUs for training, 1 GPU for testing
 - Software: Centos 6.10, ***CUDA>=9.2 Python>=3.6, Paddle>=1.6***


## Quick start: training and testing models

### 1. Preparing data

Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. It should have this basic structure:

      cityscapes/
      ├── cityscapes_list
      │   ├── test.lst
      │   ├── train.lst
      │   ├── train+.lst
      │   ├── train++.lst
      │   ├── trainval.lst
      │   └── val.lst
      ├── gtFine
      │   ├── test
      │   ├── train
      │   └── val
      ├── leftImg8bit
      │   ├── test
      │   ├── train
      │   └── val
      ├── license.txt
      └── README
   
 Download Pascal-Context dataset. It should have this basic structure:  

      pascalContext/
      ├── GroundTruth_trainval_mat
      ├── GroundTruth_trainval_png
      ├── JPEGImages
      ├── pascal_context_train.txt
      ├── pascal_context_val.txt
      ├── README.md
      └── VOCdevkit


### 2. Training

      CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch train.py  --use_gpu --use_mpio \
                                                 --cfg ./configs/pspnet_res101_cityscapes.yaml | tee -a train.log 2>&1


### 3. Testing 


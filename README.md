## Object Detection


[![](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/bruce1408/Pytorch_learning)
![](https://img.shields.io/badge/platform-Pytorch-brightgreen.svg)
![](https://img.shields.io/badge/python-3.6-blue.svg)


This repository provides code for deep learning researchers to learn [Object Detection](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

Object detection is the task of detecting instances of objects of a certain class within an image. The state-of-the-art methods can be categorized into two main types: one-stage methods and two stage-methods. One-stage methods prioritize inference speed, and example models include YOLO, SSD and RetinaNet. Two-stage methods prioritize detection accuracy, and example models include Faster R-CNN, Mask R-CNN and Cascade R-CNN.

This repository contains:

- **YOLO_v1**
- **YOLO_v2**
- **YOLO_v3**
- **YOLO_v4**
- **Instance Segmentation**

  - **FCN**

  - **U-Net**

## Table of Contents

- [Install](#install)
- [Dataset](#Dataset)
- [Related impacts](#Related-impacts)
- [Contributors](#Contributors)
- [Reference](#Reference)

## Install

This project uses [Pytorch](https://pytorch.org/get-started/previous-versions/). Go check them out if you don't have them locally installed and thirt-party dependencies.

```sh
CUDA 10.1+
torch >= 1.5.0
$ pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset

All data for this project can be found as follow

- data <https://pan.baidu.com/s/1mylwtZH9ydB4DBz5c833AA>  pasd: u871
- inception-2015-12-05.tgz: <https://pan.baidu.com/s/1o_BCsopsbgKMPqlNzMwTYw>  pasd: zt3t
- classify_image_graph_def.pd: <https://pan.baidu.com/s/1yMoF8ol4HemE4SnqCIDa0A>  pasd: 7a6k
- captcha/images: <https://pan.baidu.com/s/1p_ZYQyv7quiYdLydFw8SmA>  pasd:m1y4

```sh
copy all data into data directory
```
## Related Impacts

- [Aymeric Damien](https://github.com/aymericdamien)
- [Hvass-Labs](https://github.com/Hvass-Labs)

## Reference

### Online Video





## Contributors

This project exists thanks to all the people who contribute.
Everyone is welcome to submit code.

## VOC 数据集
voc 数据集的框是 xmin,ymin, xmax, ymax的格式,表示的是框的坐标

yolo 数据集表示的是框的 中心坐标位置,(归一化之后的),框的宽和高(归一化之后的)

计算方式:
((xmax + xmin)/2)/w, ((ymin + ymax)/2)/h, (xmax - xmin)/w, (ymax - ymin)/h

参考文献:
https://blog.csdn.net/weixin_41010198/article/details/106072347
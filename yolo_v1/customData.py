import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

folder = "/home/chenxi/dataset/VOCdevkit/VOC2012/"
NUM_BBOX = 2
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']


class VOC2012(Dataset):
    def __init__(self, is_train, is_aug=True):

        self.imgname = []
        # self.filenames = []

        self.is_aug = is_aug

        if is_train:
            with open(folder + "ImageSets/Main/train.txt", "r") as f:
                self.imgname = [x.strip() for x in f]
        else:
            with open(folder + "ImageSets/Main/val.txt") as f:
                self.imgname = [x.strip() for x in f]

        self.imagepath = folder + "JPEGImages/"  # img path

        self.labelpath = "./labels/"  # 对应的转化后的txt格式的文件

    def __len__(self):
        return len(self.imgname)

    def __getitem__(self, item):
        img = cv2.imread(self.imagepath + self.imgname[item] + '.jpg')

        h, w = img.shape[0:2]  # 分别是高和宽
        input_size = 448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0
        if h > w:
            padw = (h - 2) // 2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), "constant", constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), "constant", constant_values=0)

        img = cv2.resize(img, (input_size, input_size))

        # 数据增强部分, 不做过多处理, 仅仅归一化即可
        if self.is_aug:
            aug = transforms.Compose([
                transforms.ToTensor()
            ])
            img = aug(img)

        # 读取对应的bbox信息, 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls, xc, yc, w, h)
        with open(self.labelpath + self.imgname[item] + ".txt") as f:
            bbox = f.read().split("\n")
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox) % 5 != 0:
            raise ValueError("File:" + self.labelpath + self.imgname[item] + ".txt" + "——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        for i in range(len(bbox) // 5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w

        labels = convert_bbox2labels(bbox)
        labels = transforms.ToTensor()(labels)
        return img, labels


def convert_bbox2labels(bbox):
    """
    将bbox的(cls, x, y, w, h)数据转换为训练时方便计算
    Loss的数据形式(7, 7, 5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，
    转换为labels后，bbox的信息转换为了(px,py,w,h)格式
    """
    gridsize = 1.0 / 7
    labels = np.zeros((7, 7, 5 * NUM_BBOX + len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox) // 5):
        gridx = int(bbox[i * 5 + 1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i * 5 + 2] // gridsize)  # 当前bbox中心落在第gridy个网格,行

        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置, 偏移量
        relativX = bbox[i * 5 + 1] / gridsize - gridx
        relativY = bbox[i * 5 + 2] / gridsize - gridy

        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([relativX, relativY, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([relativX, relativY, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10 + int(bbox[i * 5])] = 1
    return labels


if __name__ == "__main__":
    with open(folder + "ImageSets/Main/val.txt", "r") as f:
        imgs = [x.strip() for x in f]

    data = VOC2012(is_train=True, is_aug=True)
    print(data.__len__())
    c = data[0]
    print(data[0][0].shape)
    print(data[0][1].shape)




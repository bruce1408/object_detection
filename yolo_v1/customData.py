import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

folder = "/home/bruce/bigVolumn/Datasets/VOCdevkit/VOC2012/"


class VOC2012(Dataset):
    def __init__(self, is_train, is_aug=True):

        self.imgname = []

        self.is_aug = is_aug

        if is_train:
            with open(folder + "ImageSets/Main/train.txt", "r") as f:
                self.imgname = [x.strip() for x in f]
        else:
            with open(folder + "ImageSets/Main/val.txt") as f:
                self.imgname = [x.strip() for x in f]

        self.imagepath = folder + "JPEGImages/"  # img path

        self.labelpath = "./labels/"

    def __Len__(self):
        return len(self.imgname)

    def __getitem__(self, item):
        img = cv2.imread(self.imagepath + self.imgname[item]+'.jpg')

        h, w = img.shape[0:2]
        input_size = 448
        padw, padh = 0, 0
        if h > w:
            padw = (h-2)//2
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), "constant", constant_values=0)






with open(folder + "ImageSets/Main/val.txt", "r") as f:
    imgs = [x.strip() for x in f]

print(imgs)



import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

folder = "/home/bruce/bigVolumn/Downloads/VOCdevkit/VOC2012/"


class VOC2012(Dataset):
    def __init__(self, is_train, is_aug=True):

        self.imgs = []
        if is_train:
            with open(folder + "ImageSets/Main/train.txt", "r") as f:
                self.imgs = [x.strip() for x in f]
        else:
            with open(folder + "ImageSets/Main/val.txt") as f:
                self.imgs = [x.strip() for x in f]



with open(folder + "ImageSets/Main/val.txt", "r") as f:
    imgs = [x.strip() for x in f]

print(imgs)



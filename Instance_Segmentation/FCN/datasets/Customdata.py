import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
IMAGE_SIZE = 256


class CustomData(data.Dataset):
    def __init__(self, imgdir, mode="train"):
        """
        FCN 数据结构, 训练集和验证集 返回的是当前原图像resize成256,
        mask返回的是256*256*2,
        [256, 256, 0] = real_mask,
        [256, 256, 1] = ~real_mask
        :param imgdir:
        :param mode:
        """
        self.imgdir = imgdir
        self.mode = mode
        # self.labealdir = labeldir
        self.imgName = []
        self.imgPath = []
        self.labelpth = []
        self.imgName = os.listdir(imgdir)
        self.imgPath = [os.path.join(self.imgdir, path) for path in self.imgName]

        if mode in ["train", "val"]:
            self.mask_path = self.imgdir+"Masks"
            self.labelpth = [os.path.join(self.mask_path, path) for path in os.listdir(self.mask_path)]

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 尺寸变化
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),  # 中心裁剪
            transforms.ToTensor(),  # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_label = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            img = Image.open(self.imgPath[index])
            mask = Image.open(self.labelpth[index])
            mask = np.array(mask.convert('1').resize((IMAGE_SIZE, IMAGE_SIZE)))
            masks = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
            masks[:, :, 0] = mask
            masks[:, :, 1] = ~mask

            img = self.transform(img)
            masks = self.transform_label(masks) * 255

            return img, masks
        else:
            img = Image.open(self.imgPath[index])
            img = self.transform(img)
            path = self.imgPath[index]
            return img, path

    def __len__(self):
        return len(self.imgName)


if __name__ == "__main__":
    data = CustomData("/home/chenxi/tmp/FCN/data/BagImages", mode="val")
    print(data[0][0].shape)

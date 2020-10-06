import os
import glob
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, dataloader
import torchvision.transforms as transforms


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


def pad_to_square(img, pad_value=0):
    """
    如果图像时矩形的,即 h>w 或者是 w>h ,那么对图像进行补全为正方形
    :param img:
    :param pad_value: 补全值设置为0
    :return:
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding, 要么是上下, 要么是左右两侧
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class ImageFolder(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, 'r') as f:
            self.img_files = [i.rstrip('\n') for i in f.readlines()]

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                            for path in self.img_files]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3*32
        self.max_size = self.img_size + 3*32
        self.batch_count = 0

    def __getitem__(self, index):
        """images"""
        # img_path = self.img_files[index % len(self.img_files)]  # 可以防止越界的写法
        img_path = self.img_files[index]
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))  # 归一化
        if len(img.shape) != 3:
            h, w = img.shape
            img = img.unsqueeze(0)
            img = img.expand((3, h, w))
        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        """labels"""
        label_path = self.label_files[index]
        targets = None
        if os.path.exists(label_path):
            # boxes 是归一化之后的 x, y, w, h
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3]/2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4]/2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3]/2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4]/2)
            # 针对图片填充,框也进行填充
            x1 += pad[0]
            x2 += pad[1]
            y1 += pad[2]
            y2 += pad[3]
            # 返回(x, y, w, h)
            boxes[:, 1] = ((x1 + x2)/2)/padded_w
            boxes[:, 2] = ((y1 + y2)/2)/padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # 是否进行图像增强, 图像翻转
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


if __name__ == "__main__":
    folder_path = "/home/bruce/PycharmProjects/object_detection/YOLO_v3/data/coco/trainvalno5k.txt"
    # with open(folder_path, 'r') as f:
    #     img_files = [i.rstrip('\n') for i in f.readlines()]
    #     img = Image.open(img_files[0]).convert("RGB")
    #     img = transforms.ToTensor()(img)  # 归一化
    #     print(img.shape)
    data = ImageFolder(folder_path, augment=True, multiscale=True)
    print(data[0])



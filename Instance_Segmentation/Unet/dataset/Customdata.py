import os
import torch
import logging
import numpy as np
from glob import glob
from PIL import Image
from os import listdir
from torchvision.transforms import transforms
from os.path import splitext
from torch.utils.data import Dataset

IMAGE_SIZE = 256


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mode='train', scale=1):
        """
        对img和mask进行图片归一化，除以255,在dataset里面兼顾train和test
        :param imgs_dir:
        :param masks_dir:
        :param scale:
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mode = mode
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # 类方法
    @classmethod
    def preprocess(cls, pil_img, scale):
        """
        对图片只进行尺度缩放到scale和归一化
        :param pil_img: 输入图片
        :param scale: 缩放因子
        :return:
        """
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)  # 缩放比例
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        # 如果图片通道是1,增加图片的通道数变成[h, w, c]
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        # 归一化
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir, idx + ".png")
        img = Image.open(img_file)
        if self.mode == "train":
            mask_file = os.path.join(self.masks_dir, idx+"_matte.png")
            mask = Image.open(mask_file)

            # 确保 img 和 mask 尺寸大小相同
            assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            img = self.preprocess(img, self.scale)
            mask = self.preprocess(mask, self.scale)
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            return img.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        else:
            return torch.from_numpy(img).type(torch.FloatTensor)


class CustomData(Dataset):
    def __init__(self, imgdir, mode='train'):
        self.imgdir = imgdir
        self.mode = mode
        if mode in ["train", "val"]:
            self.imgList = os.listdir(os.path.join(imgdir, "trainImg"))
            self.imgpath = [os.path.join(imgdir, "trainImg", i) for i in self.imgList]
        else:
            self.imgList = os.listdir(os.path.join(imgdir, "testImg"))
            self.imgpath = [os.path.join(imgdir, "testImg", i) for i in self.imgList]

        self.transform_img = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_label = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgpath)

    def __getitem__(self, index):
        img = Image.open(self.imgpath[index])
        img = self.transform_img(img)
        if self.mode in ["train", "val"]:
            self.labelpath = [os.path.join(self.imgdir, "trainMask", i.split(".")[0]+"_matte.png") for i in self.imgList]
            mask = Image.open(self.labelpath[index])
            mask = np.array(mask.convert('1').resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
            mask = self.transform_label(mask)*255
            return img, mask
        else:
            return img, self.imgpath[index]


if __name__ == "__main__":
    data = CustomData("/home/bruce/bigVolumn/Datasets/human_instance_segment/", mode='train')

    print(data[0][0])
    print(set(data[0][1].data.numpy().flatten().tolist()))
    print(data[0][1].shape)






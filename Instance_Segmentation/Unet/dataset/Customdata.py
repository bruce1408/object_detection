from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        """
        对img和mask进行图片归一化，除以255
        :param imgs_dir:
        :param masks_dir:
        :param scale:
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # 类方法
    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = os.path.join(self.imgs_dir, idx+".png")
        mask_file = os.path.join(self.masks_dir, idx+"_matte.png")

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}


if __name__ == "__main__":
    data = BasicDataset("/home/chenxi/dataset/human_mask/training", "/home/chenxi/dataset/human_mask/trainMask")

    print(data[0]['image'].shape)
    print(data[0]['mask'])

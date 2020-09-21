"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
import torch
import cv2
import PIL
import os
import pickle
# from get_imdb import get_imdb
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import config as cfg
import xml.etree.ElementTree as ET
from augmentation import augment_img


class RoiDataset(Dataset):

    def __init__(self, root_dir, filename, train=True):
        """
        函数构造函数是imdb, imdb 是类 pascal_voc, 把该类作为初始化参数加到构造函数里面
        :param imdb: 类名pascal_voc
        :param train:
        """
        super(RoiDataset, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.totalData = []
        self.imageName = []
        self.boxes = []  # boxes  [ [box], [[x1,y1,x2,y2], ...], ... ]
        self.labels = []  # labels [ [1], [2], ... ]
        self.mean = (123, 117, 104)  # RGB 形式的均值
        self.num_samples = 0  # 样本总数

        self.image_dir = "/home/chenxi/dataset/VOCdevkit/VOC2012/JPEGImages"

        self._class_to_ind = dict(zip(cfg.classes, range(len(cfg.classes))))  # 类别对应数字序号

        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,  # diff 图片不进行训练
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        # ==================================
        with open(filename) as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split()  # ['005246.jpg', '84', '48', '493', '387', '2'] img_name + 坐标 + 类型(labels)
            self.imageName.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            restore_dict = {}

            for i in range(num_boxes):
                x1 = float(splited[1 + 5 * i]) - 1
                y1 = float(splited[2 + 5 * i]) - 1
                x2 = float(splited[3 + 5 * i]) - 1
                y2 = float(splited[4 + 5 * i]) - 1
                c_label = splited[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c_label))
            self.boxes.append(box)
            self.labels.append(label)

            restore_dict['boxes'] = np.array(box)
            restore_dict['gt_class'] = np.array(label)

            self.totalData.append(restore_dict)
        self.num_samples = len(self.boxes)  # 数据集中包含所有Ground truth个数
        # ==================================
        # _roidb 是一个list, 存放的是每一个xml内部标签,写成字典的格式{"boxes":array([[x1, y1, x2, y2]], "gt_classes":array([[label]]}
        self._roidb = self.load_data()

        self._image_paths = [self.image_path_from_index(self.imageName[i]) for i in range(len(self.imageName))]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.image_dir, index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self.root_dir, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def __getitem__(self, i):

        # 得到的是最原始的图像, 标签信息数据
        image_path = self._image_paths[i]
        im_data = Image.open(image_path)
        boxes = self.boxes[i]
        gt_classes = self.labels[i]
        # print(self.totalData[i]['boxes'])
        # print(boxes)

        # 获得原始图像的 w, h
        im_info = torch.FloatTensor([im_data.size[0], im_data.size[1]])

        if self.train:

            # 数据增强
            im_data, boxes, gt_classes = augment_img(im_data, boxes, gt_classes)

            w, h = im_data.size[0], im_data.size[1]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            # resize image, 图像缩放到416尺寸即可
            input_h, input_w = cfg.input_size
            im_data = im_data.resize((input_w, input_h))

            # 缩放之后的尺寸
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255

            # convert [h, w, 3] -> [3, h, w]
            im_data_resize = im_data_resize.permute(2, 0, 1)

            # convert to tensor
            boxes = torch.from_numpy(boxes)

            # convert to tensor
            gt_classes = torch.from_numpy(gt_classes)
            num_obj = torch.Tensor([boxes.size(0)]).long()
            return im_data_resize, boxes, gt_classes, num_obj

        else:
            input_h, input_w = cfg.test_input_size
            im_data = im_data.resize((input_w, input_h))
            im_data_resize = torch.from_numpy(np.array(im_data)).float() / 255
            im_data_resize = im_data_resize.permute(2, 0, 1)
            return im_data_resize, im_info

    def load_data(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.root_dir, self.cache_path, 'train_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('gt roidb loaded from {}'.format(cache_file))
            return roidb

        with open(cache_file, 'wb') as fid:
            pickle.dump(self.totalData, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return self.totalData

    def __len__(self):
        return len(self._roidb)


def detection_collate(batch):
    """
    Collate data of different batch, it is because the boxes and gt_classes have changeable length.
    This function will pad the boxes and gt_classes with zero.

    Arguments:
    batch -- list of tuple (im, boxes, gt_classes)

    im_data -- tensor of shape (3, H, W)
    boxes -- tensor of shape (N, 4)
    gt_classes -- tensor of shape (N)
    num_obj -- tensor of shape (1)

    Returns:

    tuple
    1) tensor of shape (batch_size, 3, H, W)
    2) tensor of shape (batch_size, N, 4)
    3) tensor of shape (batch_size, N)
    4) tensor of shape (batch_size, 1)

    """

    # kind of hack, this will break down a list of tuple into
    # individual list
    bsize = len(batch)
    im_data, boxes, gt_classes, num_obj = zip(*batch)
    max_num_obj = max([x.item() for x in num_obj])
    padded_boxes = torch.zeros((bsize, max_num_obj, 4))
    padded_classes = torch.zeros((bsize, max_num_obj,))

    for i in range(bsize):
        padded_boxes[i, :num_obj[i], :] = boxes[i]
        padded_classes[i, :num_obj[i]] = gt_classes[i]

    return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)


if __name__ == "__main__":
    data = RoiDataset("/home/chenxi/dataset/VOCdevkit", "/home/chenxi/tempfile/YOLO_v1/utils/voc2007test.txt")
    i = 0
    print(data[i].__len__())
    print(data[i][0].shape)
    print(data[i][1].shape)
    print(data[i][1])
    print(data[i][2].shape)
    print(data[i][2])



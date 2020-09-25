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

    # _classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    #             'bottle', 'bus', 'car', 'cat', 'chair',
    #             'cow', 'diningtable', 'dog', 'horse',
    #             'motorbike', 'person', 'pottedplant',
    #             'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, filename, train=True):
        """
        函数构造函数是imdb, imdb 是类 pascal_voc, 把该类作为初始化参数加到构造函数里面
        :param imdb: 类名pascal_voc
        :param train:
        """
        super(RoiDataset, self).__init__()
        # self._imdb = imdb
        self._year = "2012"
        self._image_set = "train"
        self._devkit_path = "/home/chenxi/dataset/VOCdevkit"
        self.data_dir = "/home/chenxi/dataset/VOCdevkit"
        self.name = 'voc_' + self._year + '_' + self._image_set

        self.imageName = []
        self.boxes = []  # boxes  [ [box], [[x1,y1,x2,y2], ...], ... ]
        self.labels = []  # labels [ [1], [2], ... ]
        self.mean = (123, 117, 104)  # RGB 形式的均值
        self.num_samples = 0  # 样本总数

        self._data_path = "/home/chenxi/dataset/VOCdevkit/VOC2012/"

        # self.num_classes 是父类的方法, 在此处继承父类方法,且父类中将其变成了属性,所以直接调用属性即可.
        self._class_to_ind = dict(zip(cfg.classes, range(len(cfg.classes))))  # 类别对应数字序号

        self._image_ext = '.jpg'

        self.config = {'cleanup': False,
                       'use_salt': True,
                       'use_diff': False,  # diff 图片不进行训练
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        # list, 存放图片的路径
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        self.image_index = image_index
        # ==================================
        with open("/home/chenxi/tempfile/YOLO_v1/tools/voc2007test.txt") as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split()  # ['005246.jpg', '84', '48', '493', '387', '2'] img_name + 坐标 + 类型(labels)
            self.imageName.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(splited[1 + 5 * i]) - 1
                y1 = float(splited[2 + 5 * i]) - 1
                x2 = float(splited[3 + 5 * i]) - 1
                y2 = float(splited[4 + 5 * i]) - 1
                c_label = splited[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c_label) + 1)
            self.boxes.append(box)
            self.labels.append(label)
        self.num_samples = len(self.boxes)  # 数据集中包含所有Ground truth个数
        # ==================================
        # _roidb 是一个list, 存放的是每一个xml内部标签,写成字典的格式{"boxes":array([[x1, y1, x2, y2]], "gt_classes":array([[label]]}
        self._roidb = self.gt_roidb()

        self.train = train

        self._image_paths = [self.image_path_from_index(self.imageName[i]) for i in range(len(self.imageName))]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, "JPEGImages", index)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(self.data_dir, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def __getitem__(self, i):

        # 得到的是最原始的图像, 标签信息数据
        image_path = self._image_paths[i]
        im_data = Image.open(image_path)
        boxes = self.boxes[i]
        gt_classes = self.labels[i]

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

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = pickle.load(fid)
        #     print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb

        gt_roidb = [self._load_pascal_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        # Exclude the samples labeled as difficult
        if not self.config['use_diff']:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return {'boxes': boxes, 'gt_classes': gt_classes, }

    def __len__(self):
        return len(self._roidb)

    def __add__(self, other):
        self._roidb = self._roidb + other._roidb
        self._image_paths = self._image_paths + other._image_paths
        return self


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
    data = RoiDataset("voc_2012_train")
    i = 1
    print(data[i].__len__())
    print(data[i][0].shape)
    print(data[i][1].shape)
    print(data[i][1])
    print(data[i][2].shape)




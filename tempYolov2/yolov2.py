# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from torchsummary import summary
from config import config as cfg
from darknet import Darknet19
from darknet import conv_bn_leaky
# from loss import build_target, yolo_loss
from util.network import ReorgLayer
from tensorboardX import SummaryWriter


class Yolov2(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        """
        yolov2 模型，对输入是[batch, 3, 416, 416]的图片进行网络训练, 得到的是shape为[batch, 125, 13, 13]的结果,然后这个继续
        转化成[batch, 13*13*5, 25]的输出, 分别输出坐标:[batch, 13*13*5, 4], 置信度：[batch, 13*13*5, 1],
        label：[batch, 13*13*5, 20]
        输出坐标是x1,y1,w,h的形式, 但是经过转化之后, x,y 使用sigmoid函数进行转化, w, h 使用exp 进行转化
        :param classes:
        :param weights_file:
        """
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        darknet19 = Darknet19()

        # 模型加载部分
        if weights_file:
            print('load pretrained weight from {}'.format(weights_file))
            darknet19.load_weights(weights_file)
            print('pretrained weight loaded!')

        # darknet backbone, output = [b, 512, 26, 26]
        self.conv1 = nn.Sequential(darknet19.layer0,
                                   darknet19.layer1,
                                   darknet19.layer2,
                                   darknet19.layer3,
                                   darknet19.layer4)

        self.reorg = ReorgLayer()

        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        # output = [b, 1024, 13, 13]
        self.conv2 = darknet19.layer5

        # detection layers, 检测部分使用3个卷积层来进行,
        self.conv3 = nn.Sequential(conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
                                   conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.conv4 = nn.Sequential(conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

    def forward(self, x, training=False):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        x = self.conv1(x)
        shortcut = self.reorg(self.downsampler(x))  # [1, 256, 13, 13]
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([shortcut, x], dim=1)
        out = self.conv4(x)  # out = [batch, 125, 13, 13]

        if cfg.debug:
            print('check output', out.view(-1)[0:10])

        # out = (B, num_anchors * (5 + num_classes), H, W) = [batch, 125, 13, 13]
        bsize, _, h, w = out.size()

        """
        5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        [batch, 13*13*5, 25]
        """
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        """
        activate the output tensor, 对xy中心坐标进行sigmoid函数, 对hw进行exp函数变换
        `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        `softmax` for (class1_score, class2_score, ...)
        """
        xy_pred = torch.sigmoid(out[:, :, 0:2])  # 对x,y坐标进行sigmoid函数变换
        hw_pred = torch.exp(out[:, :, 2:4])  # 对 h,w坐标进行exp函数变换
        conf_pred = torch.sigmoid(out[:, :, 4:5])  # 置信度进行sigmoid函数

        class_pred = out[:, :, 5:]
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)  # [batch_size, h*w*5, 4]
        class_score = F.softmax(class_pred, dim=-1)  # 对label进行softmax进行转换

        # if training:
        #     output_variable = (delta_pred, conf_pred, class_pred)
        #     output_data = [v.data for v in output_variable]
        #     # print(output_data[0].shape)  # 16, 845, 4
        #     # print(output_data[1].shape)  # 16, 845, 1
        #     # print(output_data[2].shape)  # 16, 845, 20
        #
        #     gt_data = (gt_boxes, gt_classes, num_boxes)  # 真实数据
        #     # print(gt_boxes.shape)  # [batch, num_box, 4]
        #     # print(gt_classes.shape)  # [batch, num_box]
        #     # print(num_boxes.shape)  # [batch, 1]
        #     target_data = build_target(output_data, gt_data, h, w)  # 构建预测值和真实值
        #
        #     target_variable = [v for v in target_data]
        #     box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
        #
        #     return box_loss, iou_loss, class_loss
        #
        # return delta_pred, conf_pred, class_score
        if training:
            return delta_pred, conf_pred, class_pred, h, w
        else:
            return delta_pred, conf_pred, class_score


if __name__ == '__main__':
    model = Yolov2(weights_file="data/pretrained/darknet19_448.weights")
    if torch.cuda.is_available():
        model.cuda()
        summary(model, (3, 416, 416))
        x = torch.rand((1, 3, 416, 416)).to('cuda')
        out = model(x, True)
        delta_pred, conf_pred, class_pred, h, w = out
        print('delta_pred size:', delta_pred.size())
        print('conf_pred size:', conf_pred.size())
        print('class_pred size:', class_pred.size())
        # with SummaryWriter(comment="yolov2") as w:
        #     w.add_graph(model, x)
    else:
        summary(model, (3, 224, 224))





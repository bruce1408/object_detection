import torch
import numpy as np
import torch.nn as nn
from layers import ReorgLayer
from darknet import darknet19
from darknet import Conv_BN_LeakyReLU
from torchsummary import summary


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        """
        convert [b, c, h, w] -> [b, c * stride * stride, h/stride, w/stride]
        :param stride:
        """
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        # [batch, 64, 26, 26]
        B, C, H, W = x.size()
        ws = self.stride
        hs = self.stride

        # x = [b, c, h/2, w/2]=[16, 64, 13, 13, 2, 2]
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()

        # x = [b, c, (h/2 * w/2), 2*2] = [16, 64, 169, 4]->[16, 64, 4, 169]
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()

        # x = [b, c, 2*2, h/2, h/2] = [16, 64, 4, 13, 13]
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()

        # x = [b, 4*64, 13, 13]
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class Yolov2(nn.Module):
    num_classses = 20
    num_anchors = 5

    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5,
                 anchor_size=None, hr=False):
        super(Yolov2, self).__init__()
        self.device = device
        self.num_classses = num_classes
        self.trainable = trainable
        self.input_size = input_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_number = len(anchor_size)
        self.stride = 32
        self.hr = hr

        if not trainable:
            self.grid_cell, self.all_anchor_wh = self.set_init(input_size)

        # backbone darknet-19
        self.backbone = darknet19(pretrained=trainable, hr=hr)

        # detection
        self.conv1 = nn.Sequential(
            Conv_BN_LeakyReLU(1024, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 1024, 3, 1)
        )

        self.route_layer = Conv_BN_LeakyReLU(512, 64, 1)
        self.reorg = ReorgLayer(stride=2)
        self.conv2 = Conv_BN_LeakyReLU(1280, 1024, 3, 1)

        # prediction layer
        self.pred = nn.Conv2d(1280, 1024, 3, 1)





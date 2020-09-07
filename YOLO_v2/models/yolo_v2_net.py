import torch
import numpy as np
import torch.nn as nn
from layers import ReorgLayer
from darknet import darknet19
from darknet import Conv_BN_LeakyReLU

import torch.nn.functional as F
from torch.autograd import Variable

from torchsummary import summary
from config import config as cfg
from darknet import Darknet19
from darknet import conv_bn_leaky
# from loss import build_target, yolo_loss
from util.network import ReorgLayer
from tensorboardX import SummaryWriter

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

    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        darknet19 = Darknet19()

        # 模型加载部分
        if weights_file:
            print('darknet-19 load pretrained weight from {}'.format(weights_file))
            darknet19.load_weights(weights_file)
            print('pretrained weight loaded!')

        # darknet backbone, output = [b, 512, 26, 26]
        self.conv1 = nn.Sequential(darknet19.layer0,
                                   darknet19.layer1,
                                   darknet19.layer2,
                                   darknet19.layer3,
                                   darknet19.layer4)

        # output = [b, 1024, 13, 13]
        self.conv2 = darknet19.layer5

        # detection layers
        self.conv3 = nn.Sequential(conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
                                   conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True))

        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.conv4 = nn.Sequential(conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

        self.reorg = ReorgLayer()

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
        out = self.conv4(x)

        if cfg.debug:
            print('check output', out.view(-1)[0:10])

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        # [batch, 13*13*5, 25]
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        hw_pred = torch.exp(out[:, :, 2:4])
        conf_pred = torch.sigmoid(out[:, :, 4:5])

        class_pred = out[:, :, 5:]
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)  # [batch_size, h*w*5, 4]
        class_score = F.softmax(class_pred, dim=-1)

        if training:
            return delta_pred, conf_pred, class_pred, h, w
        else:
            return delta_pred, conf_pred, class_score


if __name__ == '__main__':
    model = Yolov2()
    if torch.cuda.is_available():
        model.cuda()
        x = torch.rand((1, 3, 416, 416)).to('cuda')
        out = model(x, False)
        delta_pred, conf_pred, class_pred = out
        print('delta_pred size:', delta_pred.size())
        print('conf_pred size:', conf_pred.size())
        print('class_pred size:', class_pred.size())
        summary(model, (3, 416, 416))
        with SummaryWriter(comment="yolov2") as w:
            w.add_graph(model, x)
    else:
        summary(model, (3, 224, 224))





import torch
import torch.nn as nn
from torchsummary import summary
from darknet import Darknet19, conv_bn_leaky
# from tensorboardX import SummaryWriter
from layers import ReorgLayer
import torch.nn.functional as F


class Yolov2(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classesName=None, weights_path=""):
        super(Yolov2, self).__init__()
        if classesName:
            self.num_classes = len(classesName)

        darknet19 = Darknet19()

        if weights_path.endswith('.pt'):
            darknet19.load_state_dict(torch.load(weights_path, map_location='cuda')['model'])
        elif weights_path == '':
            pass
        else:
            darknet19.load_weights(weights_path)
            print("load the pretrained weights from %s !" % weights_path)

        self.conv1 = nn.Sequential(
            darknet19.layer0,
            darknet19.layer1,
            darknet19.layer2,
            darknet19.layer3,
            darknet19.layer4
        )

        # [batch, 1024, 13, 13]
        self.conv2 = darknet19.layer5

        # add the new route layers
        self.conv3 = nn.Sequential(
            conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True),
            conv_bn_leaky(1024, 1024, kernel_size=3, return_module=True)
        )

        # downsample layer to substract the channels
        self.downsampler = conv_bn_leaky(512, 64, kernel_size=1, return_module=True)

        self.reorg = ReorgLayer()

        # the last conv layers
        self.conv4 = nn.Sequential(
            conv_bn_leaky(1280, 1024, kernel_size=3, return_module=True),
            nn.Conv2d(1024, (5+self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(self, x, training=False):

        x = self.conv1(x)
        shortcut = self.reorg(self.downsampler(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([shortcut, x], dim=1)
        out = self.conv4(x)

        # output = [batch, 125, 13, 13]
        batch_size, _, h, w = out.size()

        # output = [batch, 13 * 13 * 5, 25]
        out = out.permute(0, 2, 3, 1).contiguous().view(batch_size, h * w * self.num_anchors, 5 + self.num_classes)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        hw_pred = torch.exp(out[:, :, 2:4])
        conf_pred = torch.sigmoid(out[:, :, 4:5])

        class_pred = out[:, :, 5:]
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)
        class_score = F.softmax(class_pred, dim=-1)

        if training:
            return delta_pred, conf_pred, class_pred, h, w
        else:
            return delta_pred, conf_pred, class_score


if __name__ == "__main__":
    # net = Yolov2(weights_path="../pretrained/darknet19_448.weights")
    net = Yolov2()
    if torch.cuda.is_available():
        print("gpu")
        net.cuda()
        summary(net, (3, 416, 416))
        x = torch.rand((1, 3, 416, 416)).to("cuda")
        out = net(x)
        print(out[0].shape)
    else:
        summary(net, (3, 416, 416))


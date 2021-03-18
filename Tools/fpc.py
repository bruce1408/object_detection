import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion * planes)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        print(c5.shape)
        # Top-down
        p5 = self.toplayer(c5)
        print(p5.shape)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        p3 = self._upsample_add(p3, p2)  # 256 56 56
        p4 = self._upsample_add(p4, p2)  # 256 56 56
        p5 = self._upsample_add(p5, p2)  # 256 56 56

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101():
    return FPN(Bottleneck, [2, 2, 2, 2])


if __name__ == "__main__":
    net = FPN(Bottleneck, [2, 2, 2, 2])
    if torch.cuda.is_available():
        summary(net.cuda(), (3, 416, 416))
        print('cuda')
        print(net)
        x = torch.rand(size=(1, 3, 416, 416)).to('cuda')
        o1, o2, o3, o4 = net(x)
        print(o1.shape)
        print(o2.shape)
        print(o3.shape)
        print(o4.shape)
        result = torch.cat((o1, o2, o3, o4), dim=1)
        print(result.shape)






# '''RetinaFPN in PyTorch.
# See the paper "Focal Loss for Dense Object Detection" for more details.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch.autograd import Variable
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class RetinaFPN(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(RetinaFPN, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         # Bottom-up layers
#         self.layer2 = self._make_layer(block,  64, num_blocks[0], stride=1)
#         self.layer3 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer4 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer5 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
#         self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
#
#         # Top layer
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def _upsample_add(self, x, y):
#         '''Upsample and add two feature maps.
#         Args:
#           x: (Variable) top feature map to be upsampled.
#           y: (Variable) lateral feature map.
#         Returns:
#           (Variable) added feature map.
#         Note in PyTorch, when input size is odd, the upsampled feature map
#         with `F.upsample(..., scale_factor=2, mode='nearest')`
#         maybe not equal to the lateral feature map size.
#         e.g.
#         original input size: [N,_,15,15] ->
#         conv2d feature map size: [N,_,8,8] ->
#         upsampled feature map size: [N,_,16,16]
#         So we choose bilinear upsample which supports arbitrary output sizes.
#         '''
#         _,_,H,W = y.size()
#         return F.upsample(x, size=(H, W), mode='bilinear') + y
#
#     def forward(self, x):
#         # Bottom-up
#         c1 = F.relu(self.bn1(self.conv1(x)))
#         c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
#         c2 = self.layer2(c1)
#         c3 = self.layer3(c2)
#         c4 = self.layer4(c3)
#         c5 = self.layer5(c4)
#         p6 = self.conv6(c5)
#         p7 = self.conv7(F.relu(p6))
#         # Top-down
#         p5 = self.toplayer(c5)
#         p4 = self._upsample_add(p5, self.latlayer1(c4))
#         p3 = self._upsample_add(p4, self.latlayer2(c3))
#         # Smooth
#         p4 = self.smooth1(p4)
#         p3 = self.smooth2(p3)
#         return p3, p4, p5, p6, p7
#
#
# def RetinaFPN101():
#     # return RetinaFPN(Bottleneck, [2,4,23,3])
#     return RetinaFPN(Bottleneck, [2,2,2,2])
#
#
# def test():
#     net = RetinaFPN101()
#     fms = net(Variable(torch.randn(1, 3, 416, 416)))
#     for fm in fms:
#         print(fm.size())
#
# test()
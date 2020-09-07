import torch
import torch.nn as nn


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        _height, _width = height // self.stride, width // self.stride

        # x = [b, c, h/2, w/2]=[16, 64, 13, 13, 2, 2]
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()

        # x = [b, c, (h/2 * w/2), 2*2] = [16, 64, 169, 4]->[16, 64, 4, 169]
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()

        # x = [b, c, 2*2, h/2, h/2] = [16, 64, 4, 13, 13]
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()

        # x = [b, 4*64, 13, 13]
        x = x.view(batch_size, -1, _height, _width)

        return x


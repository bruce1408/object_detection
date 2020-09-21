import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WeightLoader(object):
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None

    def load_conv_bn(self, conv_model, bn_model):

        num_w = conv_model.weight.numel()

        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        self.start = self.start + num_b

        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        self.start = self.start + num_b

        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        self.start = self.start + num_b

        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        self.start = self.start + num_b

        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))

        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)

        # make sure the loaded weight is right
        assert size == self.start


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        """
        convert [b, c, h, w] -> [b, c * stride * stride, h/stride, w/stride]
        :param stride:
        """
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride

        # x = [b, c, h/2, w/2]=[16, 64, 13, 13, 2, 2]
        x = x.contiguous().view(B, C, int(H // hs), hs, int(W // ws), ws).transpose(3, 4)

        # x = [b, c, (h/2 * w/2), 2*2] = [16, 64, 169, 4]->[16, 64, 4, 169]
        x = x.contiguous().view(B, C, int(H // hs * W // ws), hs * ws).transpose(2, 3).contiguous()

        # x = [b, c, 2*2, h/2, h/2] = [16, 64, 4, 13, 13]
        x = x.contiguous().view(B, C, hs * ws, int(H // hs), int(W // ws)).transpose(1, 2).contiguous()

        # x = [b, 4*64, 13, 13]
        x = x.contiguous().view(B, hs * ws * C, int(H // hs), int(W // ws))
        return x
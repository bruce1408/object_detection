import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
i = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(
            f'f_conv{i}',
            nn.Conv2d(3, 2, 3, 1)
        )

        self.conv.add_module(
            f"f_bn{0}",
            nn.BatchNorm2d(2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == "__main__":
    net = Net().to('cuda')
    print(net)
    x = torch.rand((1, 3, 224, 224)).to('cuda')
    output = net(x)
    print(output)

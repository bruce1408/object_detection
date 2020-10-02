import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
# https://www.bilibili.com/read/cv5387252/


class net_modlist(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = nn.ModuleList([
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU()
        ])

    def forward(self, x):
        for m in self.modlist:
            print(m)
            x = m(x)
        return x


net = net_modlist()
x = torch.rand((1, 1, 224, 2224))
output = net(x)
print(output.shape)


# if __name__ == "__main__":
    # config = {"model_params": {"backbone_name": "darknet_53"}}
    # m = ModelMain(config)
    # x = torch.randn(1, 3, 416, 416)
    # y0, y1, y2 = m(x)
    # print(y0.size())
    # print(y1.size())
    # print(y2.size())
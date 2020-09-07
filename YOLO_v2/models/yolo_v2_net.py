import torch
import torch.nn as nn
from torchsummary import summary
from darknet import Darknet19
from tensorboardX import SummaryWriter

class Yolov2(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classesName=None, weights_path=False):
        super(Yolov2, self).__init__()
        if classesName:
            self.num_classes = len(classesName)

        darknet19 = Darknet19()

        if weights_path.endswith('.pt'):
            darknet19.load_state_dict(torch.load(weights_path, map_location='cuda')['model'])
        else:




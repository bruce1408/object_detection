import torch.nn as nn
import torch.nn.functional as F

"""
损失函数的定义部分
"""


def dice_loss(prediction, target):
    smooth = 1.0
    x1 = prediction.view(-1)
    x2 = target.view(-1)
    interArea = (x1 * x2).sum() + smooth
    unionArea = x1.sum() + x2.sum() + smooth
    return 1 - (2. * interArea) / unionArea


def calc_loss(prediction, target, bce_weight=0.7):
    bce = nn.BCEWithLogitsLoss()(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    loss = dice * (1-bce_weight) + bce * bce_weight

    return loss
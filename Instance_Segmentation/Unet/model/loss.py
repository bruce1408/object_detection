import torch.nn as nn
import torch.nn.functional as F

"""
损失函数的定义部分
使用dice_loss
它主要是用来评估样本相似性，如果是单纯的交叉熵，那么loss值只是和当前的像素点预测有点，
现在希望考虑引入某个点的loss不仅是该点的，还和其他点的label、loss相关，
那么就加入dice_loss, dice loss 处理正负样本不均衡的情况，就是当正负样本不均衡的时候，前景占比比较小，
而dice_loss可以求两个集合predict 和 label 的重叠面积，相当于突出了前景类别的情况，求交集的过程类似是mask掩码作用，
所以可以很好的处理类别不均衡问题，值越大表示评估样本相似程度越高，不论是在图像分割，还是NLP，都会用到dice-loss 的思想
参考文献：
A survey of loss functions for semanticsegmentation
Boundary loss for highly unbalanced segmentation
Dice Loss for Data-imbalanced NLP Tasks
https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b
https://zhuanlan.zhihu.com/p/101773544
https://zhuanlan.zhihu.com/p/269592183

"""


def dice_loss(prediction, target):
    """
    dice loss function:
    原始的形式为：
                   2py
    dice-loss1 = --------
                  p + y
    dice-loss1有个问题就是如果是父类，
    :param prediction:
    :param target:
    :return:
    """
    smooth = 1.0
    x1 = prediction.view(-1)
    x2 = target.view(-1)
    interArea = (x1 * x2).sum() + smooth
    unionArea = x1.sum() + x2.sum() + smooth
    return 1 - (2. * interArea) / unionArea


def calc_loss(prediction, target, bce_weight=0.3):
    bce = nn.BCEWithLogitsLoss()(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    # loss = dice * (1-bce_weight) + bce * bce_weight
    loss = dice * (1-bce_weight) + bce * 5

    return loss

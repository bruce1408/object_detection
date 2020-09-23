# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (M, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, M), ious between boxes
    """

    N = box1.size(0)
    M = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    """
    左上角是取最大，然后右下角取最小坐标范围。
    """
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, M))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, M))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, M))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, M))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    """
    判断大于tensor[0]的数字， 如果大于0保持原来的数， 小于0则直接置0
    orig a = tensor([[-0.1113, -0.2255, -0.2318,  1.0955,  0.8830, -1.6655,  0.4430,  0.7676]])
    max(a) = tensor([[0.0000, 0.0000, 0.0000, 1.0955, 0.8830, 0.0000, 0.4430, 0.7676]])
    """
    # iw = xi2-xi1
    # iw[iw < 0] = 0
    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))  # 如果相减结果是小于0的情况，那么就置为0
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 直接最后相加，因为广播效应，可以直接得到最后的向量大小的结果
    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, M)

    union_area = box1_area + box2_area - inter  # [845, 7]

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    把平面坐标转换成为中心坐标,且是绝对中心坐标,而不是相对中心坐标
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    把中心坐标转移成xy平面坐标即可；
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)
    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2  # x1 = cx1 - w/2
    y1 = box[:, 1] - (box[:, 3]) / 2  # y1 = cy1 - h/2
    x2 = box[:, 0] + (box[:, 2]) / 2  # x2 = cx1 + w/2
    y2 = box[:, 1] + (box[:, 3]) / 2  # y2 = cy2 + h/2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(anchor_max_iou, gt_box):
    """
    参考链接: https://blog.csdn.net/weixin_37721058/article/details/96485158
    box1表示iou和真实值最大的那个选出来anchor框; box2表示当前的真实标签; 最后得到的是中心的相对位置坐标
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to box2
    Arguments:
    anchor_max_iou  -- shape (N, 4) first set of boxes (c_x, c_y, w, h)
    gt_box          -- shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- shape (N, 4) delta values (t_x, t_y, t_w, t_h) used for transforming boxes to reference boxes
    """

    t_x = gt_box[:, 0] - anchor_max_iou[:, 0]  # 中心偏移位置x
    t_y = gt_box[:, 1] - anchor_max_iou[:, 1]  # 中心偏移位置y
    t_w = gt_box[:, 2] / anchor_max_iou[:, 2]  # 中心偏移位置w
    t_h = gt_box[:, 3] / anchor_max_iou[:, 3]  # 中心偏移位置h

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def box_transform_inv(anchor_box, pred_box):
    """
    yolov2 预测出来的结果的格式是σ(t_x), σ(t_y), exp(t_w), exp(t_h)),对于xywh坐标已经进行了sigmoid和exp转换
    主要是把 grid_anchor 和 预测的结果 deltas 进行合并起来，都是[cx,cy, w, h]中心坐标形式的,把两个合起来得到最后的预测结果坐标
    apply deltas to box to generate predicted boxes
    Arguments:
    box = (anchor_box) -- shape (N, 4), shape(c_x, c_y, w, h), 是 grid的中心坐标,w,h表示anchor的宽和高
    pred_box = (prediction value) -- shape (N, 4), (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    bx = anchor_box[:, 0] + pred_box[:, 0]
    by = anchor_box[:, 1] + pred_box[:, 1]
    bw = anchor_box[:, 2] * pred_box[:, 2]
    bh = anchor_box[:, 3] * pred_box[:, 3]

    bx = bx.view(-1, 1)
    by = by.view(-1, 1)
    bw = bw.view(-1, 1)
    bh = bh.view(-1, 1)

    pred_box = torch.cat([bx, by, bw, bh], dim=-1)
    return pred_box


def generate_all_anchors(anchors, H, W):
    """
    把 anchor 变成 xywh 的格式的数据
    anchors generalize to H * W
    Generate dense anchors given grid defined by (H,W)
    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    # [169, 2]
    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers, [169, 2]->[169, 1, 2]->[169, 5, 2] 和[169, 5, 2]合并
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2), anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)  # [845, 4]

    return all_anchors


if __name__ == "__main__":
    a = torch.tensor([[10, 20, 30, 40], [30, 45, 120, 230]], dtype=torch.float)
    b = torch.tensor([[20, 30.0, 56, 78], [31, 44, 50, 89], [77, 80, 97, 220]], dtype=torch.float)
    c = box_ious(a, b)
    print(c)


# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/12 18:30
@Author        : FelixFu / Bryce
@File          : yoloLoss.py
@Noice         :
@Modificattion :
    @Detail    : a little dufficult in builting yoloLoss funcion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class yoloLoss(nn.Module):
    """
    s = 7, b = 2, l_coord = 5, l_noobj = 0.5
    """
    def __init__(self, S, B, l_coord, l_noobj):
        """
        损失函数部分计算，
        :param S:
        :param B:
        :param l_coord:
        :param l_noobj:
        """
        super(yoloLoss, self).__init__()
        self.S = S  # 7
        self.B = B  # 2
        self.l_coord = l_coord  # 5
        self.l_noobj = l_noobj  # 0.5

    def compute_iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        # 首先计算两个box左上角点坐标的最大值和右下角坐标的最小值，然后计算交集面积，最后把交集面积除以对应的并集面积
        N = box1.size(0)
        M = box2.size(0)

        # 左上角的点, shape是[N, M, 2]
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2] 重复行
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2] 重复batch
        )

        # 右下角的点,大小还是[N, M, 2]
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # wh1 = torch.max(rb-lt, box1.new(1).fill_(0))
        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 指两个box没有重叠区域, 小于0的情况置0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M], 交集部分

        area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
        area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
        area1 = area1.view(N, 1)
        area2 = area2.view(1, M)
        # area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        # area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, pred_tensor, target_tensor):
        """
        pred_tensor: size(batchsize, S, S, B x 5 + 20 = 30) [x,y,w,h,c]
        target_tensor: size(batchsize, S, S, 30)
        首先得到 N 是 batch_size,然后筛选出有目标框的部分，因为真实数据[:, :, :, 4]和[:, :, :, 9]都是相同的，所以只把4拿出来。
        来筛选出是否是含有目标框的coo_mask 是有目标框的索引，noo_mask 是没有目标框的索引；
        然后对有目标框的索引 coo_mask 和 没有目标框的索引 noo_mask 坐标扩大原始数据尺寸。

        """
        N = pred_tensor.size()[0]

        # 具有目标标签的索引(bs, 7, 7, 30)中7*7方格中的哪个方格包含目标
        coo_obj_index = target_tensor[:, :, :, 4] > 0  # coo_mask.shape = (bs, 7, 7)
        noo_obj_index = target_tensor[:, :, :, 4] == 0  # 不具有目标的标签索引

        """
        上面的coo_obj_index只是在原来的数据上最后一个维度进行筛选，判断置信度有没有大于0，返回的是这个30维度的整个判断
        如果大于0，那么需要得到置信度大于0的整个30个维度的信息。所以，这里采用和原来的维度保持一致，即可取到置信度大于0的整个数组的信息。
        """
        # (coo_mask扩充到与target_tensor一样形状, 沿最后一维扩充), 然后用来在真实的数据上进行筛选
        coo_obj_index = coo_obj_index.unsqueeze(-1).expand_as(target_tensor)  # [batch, 7, 7, 30]

        # 不含有目标框的bool索引
        noo_obj_index = noo_obj_index.unsqueeze(-1).expand_as(target_tensor)  # [batch, 7, 7, 30]

        """
        _pred表示预测结果， _target表示真实结果，含有目标框检测的部分
        """
        #  置信度大于0的有目标框的预测值， 不一定是[14*14×batch, 30]，包含所有部分，坐标和置信度
        coo_pred = pred_tensor[coo_obj_index].view(-1, 30)

        # 置信度大于0的有目标框的真实值，包含所有部分，坐标和置信度
        coo_target = target_tensor[coo_obj_index].view(-1, 30)

        # 含有目标框的预测值的坐标部分
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)  # box[x1,y1,w1,h1,c1], [x2,y2,w2,h2,c2]

        # 含有目标框的预测值的label
        class_pred = coo_pred[:, 10:]

        # 真实结果目标框的坐标部分
        box_target = coo_target[:, :10].contiguous().view(-1, 5)

        # 真实结果目标框的label
        class_target = coo_target[:, 10:]

        """
        不含有目标框的预测值和真实数据部分
        """
        # 不含有目标框的预测值
        noo_obj_predValue = pred_tensor[noo_obj_index].view(-1, 30)

        # 不含有目标框的真实值
        noo_obj_targetValue = target_tensor[noo_obj_index].view(-1, 30)

        # 不含目标框的预测部分重新生成一个布尔值矩阵。
        noo_obj_pred_index = torch.cuda.ByteTensor(noo_obj_predValue.size()).bool()
        noo_obj_pred_index.zero_()  # 全部置0

        # 对不含目标框的预测部分置信度置1
        noo_obj_pred_index[:, 4] = 1
        noo_obj_pred_index[:, 9] = 1

        # 不含目标框的预测部分×2， 如果含有目标的是185, 不含目标的是11
        noo_pred_c = noo_obj_predValue[noo_obj_pred_index]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_obj_targetValue[noo_obj_pred_index]

        # todo 不含有目标框的置信度损失函数，就这一个
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # 计算含有目标框的损失的部分, 设置两个索引判断iou是否最大与否，然后进行判断
        coo_iou_index = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_iou_index.zero_()

        coo_not_iou_index = torch.cuda.ByteTensor(box_target.size()).bool()
        coo_not_iou_index.zero_()

        box_target_iou = torch.zeros(box_target.size()).cuda()

        """
        坐标部分参考此链接https://www.jianshu.com/p/13ec2aa50c12，认为xy是以及转换到S的格子所在的尺度，wh是归一化以后的尺度，
        所以这里xy只要进行除以S即可和wh进行加减操作;
        """
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            """
            提取2个预测值box1和1个真实值box2的坐标，然后进行变换之后求iou值，iou最大的那个提取出来
            """
            # 获取当前预测格点的2个box
            box1 = box_pred[i:i+2]

            # 随机生成Box1大小的矩阵
            box1_xyxy = torch.FloatTensor(box1.size())

            # (x, y, w, h)->(x1, y1, x2, y2)
            box1_xyxy[:, :2] = box1[:, :2]/14. - 0.5 * box1[:, 2:4]  # x1, y1
            box1_xyxy[:, 2:4] = box1[:, :2]/14. + 0.5 * box1[:, 2:4]  # x2, y2

            # 真实数据是xywh格式
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:, :2] = box2[:, :2]/14. - 0.5*box2[:, 2:4]  # x1, y1
            box2_xyxy[:, 2:4] = box2[:, :2]/14. + 0.5*box2[:, 2:4]  # x2, y2

            # 计算2个预测框box1 和 1个真实框box2的iou值
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            # 最终iou最大的框对应的index
            coo_iou_index[i+max_index] = 1
            coo_not_iou_index[i+1-max_index] = 1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            # iou value 作为box包含目标的confidence(赋值在向量的第五个位置)
            box_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        box_target_iou = box_target_iou.cuda()

        # 1.response loss
        """
        因为box_pred是[54, 5]，预测的是两个bounding_box,通过coo_iou_index可以筛掉其中一个框，然后留下的是iou最大的那个选择框，
        box_pred_response表示最大iou对应的预测数据；
        box_target_response表示最大iou对应的真实数据；
        """
        # 预测值：计算含有box的最大iou所在的框的所有信息，包括坐标和置信度筛选出来
        box_pred_response = box_pred[coo_iou_index].view(-1, 5)

        # 真实框：计算含有box的最大iou所在的框的所有信息，包括坐标和置信度筛选出来
        box_target_response = box_target[coo_iou_index].view(-1, 5)

        # 最大iou的值筛选出来
        box_target_response_iou = box_target_iou[coo_iou_index].view(-1, 5)

        # todo 包含目标框的置信度损失
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)

        # todo 坐标损失部分计算
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + \
                   F.mse_loss(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]),
                              size_average=False)

        # 2.not response loss
        # todo iou最小的那个也没有丢掉，而是加到了这部分来计算置信度损失
        box_pred_not_response = box_pred[coo_not_iou_index].view(-1, 5)
        box_target_not_response = box_target[coo_not_iou_index].view(-1, 5)
        box_target_not_response[:, 4] = 0  # 置信度置位0
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)

        # todo 类别损失部分
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)

        # todo 分别对应的是坐标损失(xy + wh),　包含物体置信度损失,　不包含物体的置信度损失, 类别损失
        return (self.l_coord*loc_loss + self.B*contain_loss + self.l_noobj*nooobj_loss + class_loss + not_contain_loss)/N


def iou_comput(box1, box2):

    N = box1.size(0)
    M = box2.size(0)
    print(N, M)

    leftTop = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        box2[:, :2].unsqueeze(0).expand(N, M, 2))

    rightBottom = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2)
    )

    wh = torch.max(rightBottom - leftTop, box1.new(1).fill_(0))

    inter = wh[:, :, 0] * wh[:, :, 1]

    box1_area = (box1[:, 2] - box1[:, 0])*(box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0])*(box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, M)

    iou = inter / (box1_area + box2_area - inter)
    return iou


if __name__ == "__main__":
    loss = yoloLoss(7, 2, 5, 0.5)
    a = torch.tensor([[10, 20, 30, 40], [30, 45, 120, 230]], dtype=torch.float)
    b = torch.tensor([[20, 30, 56, 78], [31, 44, 50, 89], [77, 80, 97, 220]], dtype=torch.float)
    c = loss.compute_iou(a, b)
    d = iou_comput(a, b)
    box1 = torch.range(0, 5879).view(-1, 14, 14, 30)
    box1[:, 12:, 13:, 4] = 1
    box1[:, :12, :13, 4] = 0
    box1 = box1.to('cuda')
    box2 = torch.range(2, 5881).view(-1, 14, 14, 30)
    box2[:, :13, :13, 4] = 0
    box2[:, 13:, 13:, 4] = 1
    box2 = box2.to('cuda')
    loss_ = loss(box1, box2)
    print(c)
    print(d)
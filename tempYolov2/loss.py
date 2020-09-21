# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from config import config as cfg
from util.bbox import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform
import torch.nn.functional as F

#
# def build_target(output, gt_data, H, W):
#     """
#     Build the training target for output tensor
#
#     Arguments:
#
#     output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
#     gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data
#
#     delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
#     conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
#     class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)
#
#     gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values (x1, y1, x2, y2) range 0~1
#     gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
#     num_obj_batch -- tensor of shape (B, 1). number of objects
#
#
#     Returns:
#     iou_target -- tensor of shape (B, H * W * num_anchors, 1)
#     iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
#     box_target -- tensor of shape (B, H * W * num_anchors, 4)
#     box_mask -- tensor of shape (B, H * W * num_anchors, 1)
#     class_target -- tensor of shape (B, H * W * num_anchors, 1)
#     class_mask -- tensor of shape (B, H * W * num_anchors, 1)
#
#     """
#     delta_pred_batch = output[0]
#     conf_pred_batch = output[1]
#     class_score_batch = output[2]
#
#     gt_boxes_batch = gt_data[0]
#     gt_classes_batch = gt_data[1]
#     num_boxes_batch = gt_data[2]
#
#     bsize = delta_pred_batch.size(0)
#
#     num_anchors = 5  # hard code for now
#
#     # initial the output tensor
#     # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
#     # what tensor is used doesn't matter
#     iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
#     iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale
#
#     box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
#     box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
#
#     class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
#     class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
#
#     # get all the anchors
#
#     anchors = torch.FloatTensor(cfg.anchors)
#
#     # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
#     # this is very crucial because the predict output is normalized to 0~1, which is also
#     # normalized by the grid width and height
#     all_grid_xywh = generate_all_anchors(anchors, H, W)  # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
#     all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
#     all_anchors_xywh = all_grid_xywh.clone()
#     all_anchors_xywh[:, 0:2] += 0.5
#     if cfg.debug:
#         print('all grid: ', all_grid_xywh[:12, :])
#         print('all anchor: ', all_anchors_xywh[:12, :])
#     all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)
#
#     # process over batches
#     for b in range(bsize):
#         num_obj = num_boxes_batch[b].item()
#         delta_pred = delta_pred_batch[b]
#         gt_boxes = gt_boxes_batch[b][:num_obj, :]
#         gt_classes = gt_classes_batch[b][:num_obj]
#
#         # rescale ground truth boxes
#         gt_boxes[:, 0::2] *= W
#         gt_boxes[:, 1::2] *= H
#
#         # step 1: process IoU target
#         # apply delta_pred to pre-defined anchors
#         all_anchors_xywh = all_anchors_xywh.view(-1, 4)
#         box_pred = box_transform_inv(all_grid_xywh, delta_pred)
#         box_pred = xywh2xxyy(box_pred)
#
#         # for each anchor, its iou target is corresponded to the max iou with any gt boxes
#         ious = box_ious(box_pred, gt_boxes)  # shape: (H * W * num_anchors, num_obj)
#         ious = ious.view(-1, num_anchors, num_obj)
#         max_iou, _ = torch.max(ious, dim=-1, keepdim=True)  # shape: (H * W, num_anchors, 1)
#         if cfg.debug:
#             print('ious', ious)
#
#         # iou_target[b] = max_iou
#
#         # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
#         iou_thresh_filter = max_iou.view(-1) > cfg.thresh
#         n_pos = torch.nonzero(iou_thresh_filter).numel()
#
#         if n_pos > 0:
#             iou_mask[b][max_iou >= cfg.thresh] = 0
#
#         # step 2: process box target and class target
#         # calculate overlaps between anchors and gt boxes
#         overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
#         gt_boxes_xywh = xxyy2xywh(gt_boxes)
#
#         # iterate over all objects
#
#         for t in range(gt_boxes.size(0)):
#             # compute the center of each gt box to determine which cell it falls on
#             # assign it to a specific anchor by choosing max IoU
#
#             gt_box_xywh = gt_boxes_xywh[t]
#             gt_class = gt_classes[t]
#             cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
#             cell_idx = cell_idx_y * W + cell_idx_x
#             cell_idx = cell_idx.long()
#
#             # update box_target, box_mask
#             overlaps_in_cell = overlaps[cell_idx, :, t]
#             argmax_anchor_idx = torch.argmax(overlaps_in_cell)
#
#             assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
#             gt_box = gt_box_xywh.unsqueeze(0)
#             target_t = box_transform(assigned_grid, gt_box)
#             if cfg.debug:
#                 print('assigned_grid, ', assigned_grid)
#                 print('gt: ', gt_box)
#                 print('target_t, ', target_t)
#             box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
#             box_mask[b, cell_idx, argmax_anchor_idx, :] = 1
#
#             # update cls_target, cls_mask
#             class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
#             class_mask[b, cell_idx, argmax_anchor_idx, :] = 1
#
#             # update iou target and iou mask
#             iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
#             if cfg.debug:
#                 print(max_iou[cell_idx, argmax_anchor_idx, :])
#             iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale
#
#     return iou_target.view(bsize, -1, 1), \
#            iou_mask.view(bsize, -1, 1), \
#            box_target.view(bsize, -1, 4),\
#            box_mask.view(bsize, -1, 1), \
#            class_target.view(bsize, -1, 1).long(), \
#            class_mask.view(bsize, -1, 1)
#
#
# def yolo_loss(output, target):
#     """
#     Build yolo loss
#
#     Arguments:
#     output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
#     target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data
#
#     delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
#     conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
#     class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)
#
#     iou_target -- Variable of shape (B, H * W * num_anchors, 1)
#     iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
#     box_target -- Variable of shape (B, H * W * num_anchors, 4)
#     box_mask -- Variable of shape (B, H * W * num_anchors, 1)
#     class_target -- Variable of shape (B, H * W * num_anchors, 1)
#     class_mask -- Variable of shape (B, H * W * num_anchors, 1)
#
#     Return:
#     loss -- yolo overall multi-task loss
#     """
#
#     delta_pred_batch = output[0]
#     conf_pred_batch = output[1]
#     class_score_batch = output[2]
#
#     iou_target = target[0]
#     iou_mask = target[1]
#     box_target = target[2]
#     box_mask = target[3]
#     class_target = target[4]
#     class_mask = target[5]
#
#     b, _, num_classes = class_score_batch.size()
#     class_score_batch = class_score_batch.view(-1, num_classes)
#     class_target = class_target.view(-1)
#     class_mask = class_mask.view(-1)
#
#     # ignore the gradient of noobject's target
#     class_keep = class_mask.nonzero().squeeze(1)
#     class_score_batch_keep = class_score_batch[class_keep, :]
#     class_target_keep = class_target[class_keep]
#
#     # if cfg.debug:
#     #     print(class_score_batch_keep)
#     #     print(class_target_keep)
#
#     # calculate the loss, normalized by batch size.
#     box_loss = 1 / b * cfg.coord_scale *\
#                F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
#
#     iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
#
#     class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')
#
#     return box_loss, iou_loss, class_loss


class Yolo_loss(nn.Module):
    def __init__(self):
        """
        Yolo_Loss 专门定义成一个类, 输入output为预测结果, gt表示真实数据结果, h, w,当前图片的尺寸
        """
        super(Yolo_loss, self).__init__()

    def forward(self, output_variable, gt_data, h, w):
        output_data = [v.data for v in output_variable]
        # print(output_data[0].shape)  # 16, 845, 4
        # print(output_data[1].shape)  # 16, 845, 1
        # print(output_data[2].shape)  # 16, 845, 20

        # print(gt_boxes.shape)  # [batch, num_box, 4]
        # print(gt_classes.shape)  # [batch, num_box]
        # print(num_boxes.shape)  # [batch, 1]

        # build_target 是对预测和真实部分进行编码
        target_data = self.build_target(output_data, gt_data, h, w)

        target_variable = [v for v in target_data]
        box_loss, iou_loss, class_loss = self.yolo_loss(output_variable, target_variable)

        return box_loss, iou_loss, class_loss

    def yolo_loss(self, output, target):

        """
        Build yolo loss

        Arguments:
        output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
        长度是3
        target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data
        长度是6

        delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
        conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
        class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

        iou_target -- Variable of shape (B, H * W * num_anchors, 1)
        iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
        box_target -- Variable of shape (B, H * W * num_anchors, 4)
        box_mask -- Variable of shape (B, H * W * num_anchors, 1)
        class_target -- Variable of shape (B, H * W * num_anchors, 1)
        class_mask -- Variable of shape (B, H * W * num_anchors, 1)

        Return:
        loss -- yolo overall multi-task loss
        """

        delta_pred_batch = output[0]
        conf_pred_batch = output[1]
        class_score_batch = output[2]

        iou_target = target[0]
        iou_mask = target[1]
        box_target = target[2]
        box_mask = target[3]
        class_target = target[4]
        class_mask = target[5]

        b, _, num_classes = class_score_batch.size()
        class_score_batch = class_score_batch.view(-1, num_classes)
        class_target = class_target.view(-1)
        class_mask = class_mask.view(-1)

        # ignore the gradient of noobject's target
        class_keep = class_mask.nonzero().squeeze(1)
        class_score_batch_keep = class_score_batch[class_keep, :]
        class_target_keep = class_target[class_keep]

        # if cfg.debug:
        #     print(class_score_batch_keep)
        #     print(class_target_keep)

        # calculate the loss, normalized by batch size.
        box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask,
                                                        box_target * box_mask, reduction='sum') / 2.0

        iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask,
                                      iou_target * iou_mask, reduction='sum') / 2.0

        class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep,
                                                               class_target_keep, reduction='sum')

        return box_loss, iou_loss, class_loss

    def build_target(self, output, gt_data, H, W):
        """
        Build the training target for output tensor

        Arguments:

        output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
        gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

        ===========================================================================
        delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), size = [batch, 13*13*5, 4] = [batch, 845, 4]
        predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)

        conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), size = [batch, 13*13*5, 1] = [batch, 845, 1]
        prediction of IoU score σ(t_c)

        class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes),size = [batch, 13*13*5, 20] = [batch, 845, 20]
         prediction of class scores (cls1, cls2, ..)
        ==========================================================================

        gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values (x1, y1, x2, y2) range 0~1
        gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
        num_obj_batch -- tensor of shape (B, 1). number of objects


        Returns:
        iou_target -- tensor of shape (B, H * W * num_anchors, 1)
        iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
        box_target -- tensor of shape (B, H * W * num_anchors, 4)
        box_mask -- tensor of shape (B, H * W * num_anchors, 1)
        class_target -- tensor of shape (B, H * W * num_anchors, 1)
        class_mask -- tensor of shape (B, H * W * num_anchors, 1)

        """
        # 预测值
        delta_pred_batch = output[0]  # [batch, 13*13*5, 4] = [16, 845, 4]
        conf_pred_batch = output[1]  # [batch, 13*13*5, 1]
        class_score_batch = output[2]  # [batch, 13*13*5, 20]

        # 真实值
        gt_boxes_batch = gt_data[0]  # [batch, num_obj, 4] = [16, N, 4]
        gt_classes_batch = gt_data[1]  # [batch, num_obj]
        num_boxes_batch = gt_data[2]  # [batch, 1]

        # num of batch
        bsize = delta_pred_batch.size(0)

        num_anchors = 5  # hard code for now

        """
        initial the output tensor
        we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
        what tensor is used doesn't matter
        """
        # [16, 169, 5, 4]
        box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))

        # [16, 169, 5, 1]
        box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # [16, 169, 5, 1]
        iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # [16, 169, 5, 1]全是1的矩阵
        iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

        # [16, 169, 5, 1]
        class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # [16, 169, 5, 1]
        class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

        # get all the anchors
        anchors = torch.FloatTensor(cfg.anchors)

        """
        note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
        this is very crucial because the predict output is normalized to 0~1, which is also
        normalized by the grid width and height
        """
        # all_grid_xywh shape是[845, 4], [[0,0,1,1],[1,0,2,2]...]坐标加anchor的wh
        all_grid_xywh = generate_all_anchors(anchors, H, W)  # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
        all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
        all_anchors_xywh = all_grid_xywh.clone()
        all_anchors_xywh[:, 0:2] += 0.5  # 变成中心点的坐标
        if cfg.debug:
            print('all grid: ', all_grid_xywh[:12, :])
            print('all anchor: ', all_anchors_xywh[:12, :])

        # 把中心坐标改成正常二位平面坐标 [845, 4]
        all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

        # 每一个batch_size进行迭代
        for b in range(bsize):
            # 真实的框的box数目
            num_obj = num_boxes_batch[b].item()
            # 预测框坐标 [845, 4]
            delta_pred = delta_pred_batch[b]
            # 真实值box[num, 4]
            gt_boxes = gt_boxes_batch[b][:num_obj, :]
            # 真实的类别torch.size([1])
            gt_classes = gt_classes_batch[b][:num_obj]

            # rescale ground truth boxes, 扩展成13尺度的box
            gt_boxes[:, 0::2] *= W
            gt_boxes[:, 1::2] *= H

            # todo 1: process IoU target
            # apply delta_pred to pre-defined anchors
            all_anchors_xywh = all_anchors_xywh.view(-1, 4)

            # box_pred是all_grid_xywh和delta合并而来, all_grid_xywh是中心坐标,不是偏移量
            box_pred = box_transform_inv(all_grid_xywh, delta_pred)  # [845， 4]

            # 预测结果的中心坐标变成了xy平面坐标
            box_pred = xywh2xxyy(box_pred)

            # 计算预测值框和实际框的iou数值shape=[N, M] N = box_pred.size()[0] = H * W * num_anchors, M = gt_boxex.size()[0]
            ious = box_ious(box_pred, gt_boxes)

            # shape: (H * W, num_anchors, num_obj) = [169, 5, num]
            ious = ious.view(-1, num_anchors, num_obj)
            # 求最大的iou数值,然后开始进行处理
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)  # shape: (H * W, num_anchors, 1)
            if cfg.debug:
                print('ious', ious)

            # iou_target[b] = max_iou

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 满足iou值大于阈值的部分索引,变成一行[845]的True,False的一维数组
            iou_thresh_filter = max_iou.view(-1) > cfg.thresh

            # 找出所有True数值的个数; numel()表示个数(有多少是慢去iou值大于thresh的)
            n_pos = torch.nonzero(iou_thresh_filter).numel()

            if n_pos > 0:
                iou_mask[b][max_iou >= cfg.thresh] = 0  # 大于阈值则设置为0, 否则设置为1

            # todo 2: process box target and class target
            # 计算锚框和当前batch的真实框的iou
            overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)

            # 真实框从平面坐标变换成中心坐标,不是相对位置,而是绝对位置
            gt_boxes_xywh = xxyy2xywh(gt_boxes)

            """
            iterate over all objects
            compute the center of each gt box to determine which cell it falls on
            assign it to a specific anchor by choosing max IoU
            迭代每个真实框来进行计算
            """
            for t in range(gt_boxes.size(0)):
                gt_box_xywh = gt_boxes_xywh[t]
                gt_class = gt_classes[t]

                # cell_idx 不超过最大整数
                cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
                cell_idx = cell_idx_y * W + cell_idx_x
                cell_idx = cell_idx.long()

                # update box_target, box_mask
                overlaps_in_cell = overlaps[cell_idx, :, t]
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # 真实标签和对应anchor计算iou得到最大的那个anchor框
                assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
                gt_box = gt_box_xywh.unsqueeze(0)

                # 真实框和anchor框来构成 target_t, 变成中心相对坐标
                target_t = box_transform(assigned_grid, gt_box)
                if cfg.debug:
                    print('assigned_grid, ', assigned_grid)
                    print('gt: ', gt_box)
                    print('target_t, ', target_t)
                # assign the box_target and box_mask with max iou num
                box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
                box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

                # update cls_target, cls_mask
                class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
                class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

                # update iou target and iou mask
                iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]

                if cfg.debug:
                    print(max_iou[cell_idx, argmax_anchor_idx, :])
                # iou这个位置设为5
                iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

        return iou_target.view(bsize, -1, 1), iou_mask.view(bsize, -1, 1), \
               box_target.view(bsize, -1, 4), box_mask.view(bsize, -1, 1),\
               class_target.view(bsize, -1, 1).long(), class_mask.view(bsize, -1, 1)


































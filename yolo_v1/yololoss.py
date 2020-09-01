import torch
import torch.nn as nn
import numpy as np

class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1, self).__init__()

    def forward(self, pred, labels):
        """
        pred 预测结果 [30, 7, 7]
        labels 真实结果[30, 7, 7]
        """
        num_gridx, num_gridy = labels.size()[-2:]
        num_b = 2
        num_cls = 20
        noobj_confi_loss = 0.  # 不含目标网格损失(置信度损失)
        coor_loss = 0.  # 含有目标网格损失
        obj_confi_loss = 0.  # 含有目标的置信度损失
        class_loss = 0.  # 含有目标的类别损失
        n_batch = labels.size()[0]  # bs 大小

        for i in range(n_batch):
            for n in range(num_gridx):
                for m in range(num_gridy):
                    if labels[i, 4, m, n] == 1:  # 是否包含物体
                        bbox1_pred_xyxy = ((pred[i, 0, m, n] + m) / num_gridx - pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy - pred[i, 3, m, n] / 2,
                                           (pred[i, 0, m, n] + m) / num_gridx + pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy + pred[i, 3, m, n] / 2)






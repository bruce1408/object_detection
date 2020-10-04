# import torch
# import torch.nn as nn
#
# from collections import defaultdict
# from yolo_layer import YOLOLayer
#
#
# def add_conv(in_ch, out_ch, ksize, stride):
#     """
#     Add a conv2d / batchnorm / leaky ReLU block.
#     Args:
#         in_ch (int): number of input channels of the convolution layer.
#         out_ch (int): number of output channels of the convolution layer.
#         ksize (int): kernel size of the convolution layer.
#         stride (int): stride of the convolution layer.
#     Returns:
#         stage (Sequential) : Sequential layers composing a convolution block.
#     """
#     stage = nn.Sequential()
#     pad = (ksize - 1) // 2
#     stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
#                                        out_channels=out_ch, kernel_size=ksize, stride=stride,
#                                        padding=pad, bias=False))
#     stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
#     stage.add_module('leaky', nn.LeakyReLU(0.1))
#     return stage
#
#
# class resblock(nn.Module):
#     """
#     Sequential residual blocks each of which consists of \
#     two convolution layers.
#     Args:
#         ch (int): number of input and output channels.
#         nblocks (int): number of residual blocks.
#         shortcut (bool): if True, residual tensor addition is enabled.
#     """
#     def __init__(self, ch, nblocks=1, shortcut=True):
#
#         super().__init__()
#         self.shortcut = shortcut
#         self.module_list = nn.ModuleList()
#         for i in range(nblocks):
#             resblock_one = nn.ModuleList()
#             resblock_one.append(add_conv(ch, ch//2, 1, 1))
#             resblock_one.append(add_conv(ch//2, ch, 3, 1))
#             self.module_list.append(resblock_one)
#
#     def forward(self, x):
#         for module in self.module_list:
#             h = x
#             for res in module:
#                 h = res(h)
#             x = x + h if self.shortcut else h
#         return x
#
#
# def create_yolov3_modules(config_model, ignore_thre):
#     """
#     Build yolov3 layer modules.
#     Args:
#         config_model (dict): model configuration.
#             See YOLOLayer class for details.
#         ignore_thre (float): used in YOLOLayer.
#     Returns:
#         mlist (ModuleList): YOLOv3 module list.
#     """
#
#     # DarkNet53
#     mlist = nn.ModuleList()
#     mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
#     mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
#     mlist.append(resblock(ch=64))
#     mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
#     mlist.append(resblock(ch=128, nblocks=2))
#     mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
#     mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here
#     mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
#     mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here
#     mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
#     mlist.append(resblock(ch=1024, nblocks=4))
#
#     # YOLOv3
#     mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))
#     mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
#     # 1st yolo branch
#     mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
#     mlist.append(
#          YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre))
#
#     mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
#     mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
#     mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
#     mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
#     mlist.append(resblock(ch=512, nblocks=1, shortcut=False))
#     mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
#     # 2nd yolo branch
#     mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
#     mlist.append(
#         YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre))
#
#     mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
#     mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
#     mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
#     mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
#     mlist.append(resblock(ch=256, nblocks=2, shortcut=False))
#     mlist.append(
#          YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre))
#
#     return mlist
#
#
# class YOLOv3(nn.Module):
#     """
#     YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
#     The network returns loss values from three YOLO layers during training \
#     and detection results during test.
#     """
#     def __init__(self, config_model, ignore_thre=0.7):
#         """
#         Initialization of YOLOv3 class.
#         Args:
#             config_model (dict): used in YOLOLayer.
#             ignore_thre (float): used in YOLOLayer.
#         """
#         super(YOLOv3, self).__init__()
#
#         if config_model['TYPE'] == 'YOLOv3':
#             self.module_list = create_yolov3_modules(config_model, ignore_thre)
#         else:
#             raise Exception('Model name {} is not available'.format(config_model['TYPE']))
#
#     def forward(self, x, targets=None):
#         """
#         Forward path of YOLOv3.
#         Args:
#             x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
#                 where N, C are batchsize and num. of channels.
#             targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`
#         Returns:
#             training:
#                 output (torch.Tensor): loss tensor for backpropagation.
#             test:
#                 output (torch.Tensor): concatenated detection results.
#         """
#         train = targets is not None
#         output = []
#         self.loss_dict = defaultdict(float)
#         route_layers = []
#         for i, module in enumerate(self.module_list):
#             # yolo layers
#             if i in [14, 22, 28]:
#                 if train:
#                     x, *loss_dict = module(x, targets)
#                     for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
#                         self.loss_dict[name] += loss
#                 else:
#                     x = module(x)
#                 output.append(x)
#             else:
#                 x = module(x)
#
#             # route layers
#             if i in [6, 8, 12, 20]:
#                 route_layers.append(x)
#             if i == 14:
#                 x = route_layers[2]
#             if i == 22:  # yolo 2nd
#                 x = route_layers[3]
#             if i == 16:
#                 x = torch.cat((x, route_layers[1]), 1)
#             if i == 24:
#                 x = torch.cat((x, route_layers[0]), 1)
#         if train:
#             return sum(output)
#         else:
#             return torch.cat(output, 1)
#
#
# if __name__ == "__main__":
#     x = torch.rand((1, 3, 224, 224))
#     net = YOLOv3()
#     output = net(x)
#     print(output.shape)
#
#

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils.parse_config import *
from utils.utils import build_targets

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    # 判断是否含有卷积，batchNorma，leakyReLU层
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        # 卷积层判断
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2

            # 添加卷积层
            modules.add_module(f"conv_{module_i}",
                               nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size,
                                         stride=int(module_def["stride"]), padding=pad, bias=not bn,),)
            if bn:
                # 添加batchNorma层
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                # 添加leakyReLU层
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        # 最大池化层判断
        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        # 上采样层判断
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        # route 层判断
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        # shortcut 层判断
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        # yolo 层判断
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size

        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ), -1,)

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": total_loss.detach().cpu().item(),
                "x": loss_x.detach().cpu().item(),
                "y": loss_y.detach().cpu().item(),
                "w": loss_w.detach().cpu().item(),
                "h": loss_h.detach().cpu().item(),
                "conf": loss_conf.detach().cpu().item(),
                "cls": loss_cls.detach().cpu().item(),
                "cls_acc": cls_acc.detach().cpu().item(),
                "recall50": recall50.detach().cpu().item(),
                "recall75": recall75.detach().cpu().item(),
                "precision": precision.detach().cpu().item(),
                "conf_obj": conf_obj.detach().cpu().item(),
                "conf_noobj": conf_noobj.detach().cpu().item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases

                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b

                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b

                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b

                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet("/home/chenxi/tmp/YOLO_v3/config/yolov3.cfg").to(device)
    x = torch.rand((1, 3, 416, 416)).to(device)
    output = model(x)
    print(output.shape)


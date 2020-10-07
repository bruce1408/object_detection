import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.yolov3 import Darknet
from tensorboardX import SummaryWriter
from utils.parse_config import parse_model_config
from datasets.Customdata import ImageFolder


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v3')
    parser.add_argument('--max_epochs', dest='max_epochs', help='number of epochs to train', default=200, type=int)

    parser.add_argument('--names', dest='names', default='./data/coco.names', type=str)

    parser.add_argument('--ckpt', dest='ckpt', default="", help="resume model to load", type=str)

    parser.add_argument('--traindata', dest='traindata', default='./data/coco/trainvalno5k.txt', type=str)

    parser.add_argument("--valdata", dest="valdata", default="./data/coco/5k.txt", type=str)

    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=8, type=int)

    parser.add_argument("--model_def", default="config/yolov3.cfg", help="path to model definition file", type=str)

    parser.add_argument('--output_dir', dest='output_dir', default='./outputs', type=str)

    parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints', type=str)

    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool)

    parser.add_argument('--display_interval', dest='display_interval', default=10, type=int)

    parser.add_argument('--mGPUs', dest='mGPUs', default=False, type=bool)

    parser.add_argument("--img_size", dest="img_size", default=416, help="size of each image dimension", type=int)

    parser.add_argument('--save_interval', dest='save_interval', default=20, type=int)

    parser.add_argument('--resume', dest='resume', default=False, type=bool)

    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch', default=180, type=int)

    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument("--pretrained_weights", default="./pretrained/darknet53.conv.74",
                        type=str, help="if specified starts from checkpoint model")

    args = parser.parse_args()
    return args


CUDA = torch.cuda.is_available()


def main():

    # 网络参数初始化
    args = parse_args()
    net = Darknet(args.model_def)
    start_epoch = 0
    parameters = parse_model_config(args.model_def).pop(0)
    batch_size = int(parameters['batch'])
    lr = float(parameters['learning_rate'])
    mom = float(parameters['momentum'])
    decay = float(parameters['decay'])

    # 类别名称
    namespath = args.names
    with open(namespath, 'r') as f:
        names = f.read().split("\n")[:-1]

    # 是否使用GPU多卡训练
    if CUDA:
        net.to(torch.device("cuda"))
        # net = nn.DataParallel(net)

    if args.pretrained_weights:
        if args.pretrained_weights.endswith(".pth"):
            net.load_state_dict(torch.load(args.pretrained_weights))
        else:
            net.load_darknet_weights(args.pretrained_weights)
        net = nn.DataParallel(net)

    # 断点加载
    if args.resume:
        ckpt = args.ckpt
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print("load the model from {} and start epoch is: {}".format(ckpt, start_epoch))

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    # 生成模型输出路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载数据集
    print('loading datasets....')
    trainpath = args.traindata
    valpath = args.valdata

    train_dataset = ImageFolder(trainpath, augment=True, multiscale=args.multiscale_training)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=train_dataset.collate_fn)

    val_dataset = ImageFolder(valpath, img_size=args.img_size, augment=False, multiscale=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=val_dataset.collate_fn)

    print('training data number: {}'.format(len(train_dataset)), "val data number: {}".format(len(val_dataset)))

    # 优化器初始化
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=decay)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(args.max_epoch):
        train()
        val()
    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    # net.train()

    # iters_per_epoch = int(len(train_dataset) / args.batch_size)  # 1383

    # start training
    # for epoch in range(args.start_epoch, args.max_epochs+1):
    #     loss_temp = 0
    #     train_data_iter = iter(train_dataloader)
    #
    #     if epoch in args.decay_lrs:
    #         lr = args.decay_lrs[epoch]
    #         adjust_learning_rate(optimizer, lr)
    #         print('adjust learning rate to {}'.format(lr))
    #
    #     if cfg.multi_scale and epoch in cfg.epoch_scale:
    #         cfg.scale_range = cfg.epoch_scale[epoch]
    #         print('change scale range to {}'.format(cfg.scale_range))
    #
    #     print('change input size {}'.format(cfg.input_size))
    #
    #     for step in range(iters_per_epoch):
    #         if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
    #             scale_index = np.random.randint(*cfg.scale_range)
    #             cfg.input_size = cfg.input_sizes[scale_index]
    #             # print('change input size {}'.format(cfg.input_size))
    #         if step+1 == iters_per_epoch:
    #             print("inner loop change input size is: ", cfg.input_size)
    #
    #         im_data, boxes, gt_classes, num_obj = next(train_data_iter)
    #         if args.use_cuda:
    #             im_data = im_data.cuda()
    #             boxes = boxes.cuda()
    #             gt_classes = gt_classes.cuda()
    #             num_obj = num_obj.cuda()
    #
    #         im_data_variable = Variable(im_data)
    #         # todo
    #         # outPut是预测结果, 为list, 分别有 box_loss, iou_loss, class_loss, h, w 如果是 false 只有三项
    #         # box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True)
    #         output = model(im_data_variable, training=True)
    #         output_data = output[0:3]
    #         h, w = output[-2:]
    #         """
    #         # 真实标签
    #         # print('boxes is:', boxes.shape)  # [16, 20, 4]
    #         # print("gt_class is:", gt_classes.shape)  # [16, 20]
    #         # print(num_obj.shape)  # [16, 1]
    #         # print("pred boxes is:", output_data[0].shape)  # [16, 845, 4]
    #         # print("pred conf is:", output_data[1].shape)  # [16, 845, 1]
    #         # print("pred labels is:", output_data[2].shape)  # [16, 845, 20]
    #         """
    #         target_data = [boxes, gt_classes, num_obj]
    #
    #         box_loss, iou_loss, class_loss = loss_(output_data, target_data, h, w)
    #
    #         loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()
    #
    #         optimizer.zero_grad()
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         loss_temp += loss.item()
    #
    #         if (step + 1) % args.display_interval == 0:
    #             toc = time.time()
    #             loss_temp /= args.display_interval
    #
    #             iou_loss_v = iou_loss.mean().item()
    #             box_loss_v = box_loss.mean().item()
    #             class_loss_v = class_loss.mean().item()
    #
    #             print("[epoch: %d][step: %2d/%4d] loss: %.4f, lr: %.2e, "
    #                   % (epoch, step+1, iters_per_epoch, loss_temp, lr), end='')
    #             print(" iou_loss: %.4f, box_loss: %.4f, cls_loss: %.4f"
    #                   % (iou_loss_v, box_loss_v, class_loss_v))
    #
    #             if args.use_tfboard:
    #
    #                 n_iter = (epoch - 1) * iters_per_epoch + step + 1
    #                 writer.add_scalar('losses/loss', loss_temp, n_iter)
    #                 writer.add_scalar('losses/iou_loss', iou_loss_v, n_iter)
    #                 writer.add_scalar('losses/box_loss', box_loss_v, n_iter)
    #                 writer.add_scalar('losses/cls_loss', class_loss_v, n_iter)
    #
    #             loss_temp = 0
    #             tic = time.time()
    #
    #     if epoch % args.save_interval == 0:
    #         save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
    #
    #         torch.save({
    #             'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
    #             'epoch': epoch,
    #             'lr': lr},
    #             save_name)
def train():
    pass
def val():
    pass

if __name__ == '__main__':
    main()










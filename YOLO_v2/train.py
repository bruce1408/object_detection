from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import numpy as np
import argparse
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.yolov2 import Yolov2
from torch import optim
from utils.network import adjust_learning_rate
from tensorboardX import SummaryWriter
from config import config as cfg
from models.loss import Yolo_loss
from datasets.customdata import RoiDataset
from datasets.customdata import detection_collate


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=200, type=int)

    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)

    parser.add_argument('--datasets', dest='datasets',
                        default='voc0712trainval', type=str)

    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)

    parser.add_argument('--output_dir', dest='output_dir',
                        default='/home/chenxi/object_detection/YOLO_v2/output/', type=str)

    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=False, type=bool)

    parser.add_argument('--display_interval', dest='display_interval',
                        default=10, type=int)

    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=True, type=bool)

    parser.add_argument('--save_interval', dest='save_interval',
                        default=20, type=int)

    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)

    parser.add_argument('--resume', dest='resume',
                        default=False, type=bool)

    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=180, type=int)

    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)

    args = parser.parse_args()
    return args


def train():

    # define the hyper parameters first
    args = parse_args()
    args.lr = cfg.lr
    args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    args.batch_size = cfg.batch_size
    args.pretrained_model = os.path.join(
        '/home/chenxi/object_detection/data/', 'pretrained', 'darknet19_448.weights')  # 官方训练的模型

    print('Called with args:')

    lr = args.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load datasets
    print('loading datasets....')
    train_dataset = RoiDataset("/home/chenxi/object_detection/data/yolo_v1_datasets/",
                               "/home/chenxi/object_detection/data/yolo_v1_datasets/images.txt",
                               "/home/chenxi/object_detection/data/yolo_v1_datasets/images")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=detection_collate, drop_last=True)

    print('training rois number: {}'.format(len(train_dataset)))

    # initialize the model
    tic = time.time()
    model = Yolov2(device, args.mGPUs, weights_file=args.pretrained_model)
    toc = time.time()
    print('model loaded! : cost time {:.2f}s'.format(toc - tic))

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        print('resume training enable')
        resume_checkpoint_name = 'yolov2_epoch_{}.pth'.format(
            args.checkpoint_epoch)
        resume_checkpoint_path = os.path.join(
            output_dir, resume_checkpoint_name)
        print('resume from {}'.format(resume_checkpoint_path))

        # 模型加载
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        print('learning rate is {}'.format(lr))
        adjust_learning_rate(optimizer, lr)

    if args.use_cuda:
        model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)
        model = model.to(device)

    loss_ = Yolo_loss(args.mGPUs)
    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)  # 1383

    # start training
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        loss_temp = 0
        train_data_iter = iter(train_dataloader)

        if epoch in args.decay_lrs:
            lr = args.decay_lrs[epoch]
            adjust_learning_rate(optimizer, lr)
            print('adjust learning rate to {}'.format(lr))

        if cfg.multi_scale and epoch in cfg.epoch_scale:
            cfg.scale_range = cfg.epoch_scale[epoch]
            print('change scale range to {}'.format(cfg.scale_range))

        print('change input size {}'.format(cfg.input_size))

        for step in range(iters_per_epoch):
            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                cfg.input_size = cfg.input_sizes[scale_index]
                # print('change input size {}'.format(cfg.input_size))
            if step + 1 == iters_per_epoch:
                print("inner loop change input size is: ", cfg.input_size)

            im_data, boxes, gt_classes, num_obj = next(train_data_iter)
            if args.use_cuda:
                im_data = im_data.to(device)
                boxes = boxes.to(device)
                gt_classes = gt_classes.to(device)
                num_obj = num_obj.to(device)
                # print("im_data is: ", type(im_data))

            im_data_variable = Variable(im_data)
            # im_data_variable = im_data
            # print("im_data_variable", type(im_data_variable))
            # todo
            # outPut是预测结果, 为list, 分别有 box_loss, iou_loss, class_loss, h, w 如果是 false 只有三项
            # box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True)
            output = model(im_data_variable, training=True)
            # output = torch.Tensor(output)
            # print("output done")
            output_data = output[0:3]
            # print("output_data", type(output_data[0]))
            h, w = output[-2:]
            # print("h, w", type(h))
            # print(type(w))
            """
            # 真实标签
            # print('boxes is:', boxes.shape)  # [16, 20, 4]
            # print("gt_class is:", gt_classes.shape)  # [16, 20]
            # print(num_obj.shape)  # [16, 1]
            # print("pred boxes is:", output_data[0].shape)  # [16, 845, 4]
            # print("pred conf is:", output_data[1].shape)  # [16, 845, 1]
            # print("pred labels is:", output_data[2].shape)  # [16, 845, 20]
            """
            target_data = [boxes, gt_classes, num_obj]
            # print("target_data done")
            # print('output_data: ',type(output_data))
            # print("target_data", type(target_data))
            # print('h: ', type(h))
            # print('w: ', type(w))

            box_loss, iou_loss, class_loss = loss_(
                output_data, target_data, h, w)

            loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

            if (step + 1) % args.display_interval == 0:
                toc = time.time()
                loss_temp /= args.display_interval

                iou_loss_v = iou_loss.mean().item()
                box_loss_v = box_loss.mean().item()
                class_loss_v = class_loss.mean().item()

                print("[epoch: %d][step: %2d/%4d] loss: %.4f, lr: %.2e, "
                      % (epoch, step + 1, iters_per_epoch, loss_temp, lr), end='')
                print(" iou_loss: %.4f, box_loss: %.4f, cls_loss: %.4f"
                      % (iou_loss_v, box_loss_v, class_loss_v))

                if args.use_tfboard:

                    n_iter = (epoch - 1) * iters_per_epoch + step + 1
                    writer.add_scalar('losses/loss', loss_temp, n_iter)
                    writer.add_scalar('losses/iou_loss', iou_loss_v, n_iter)
                    writer.add_scalar('losses/box_loss', box_loss_v, n_iter)
                    writer.add_scalar('losses/cls_loss', class_loss_v, n_iter)

                loss_temp = 0
                tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(
                output_dir, 'yolov2_epoch_{}.pth'.format(epoch))

            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr},
                save_name)


if __name__ == '__main__':
    train()

    price = str(u'12.3456')
    bids = ['1.0', '2.0']
    zip(float(price), bids)

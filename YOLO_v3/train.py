import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.yolov3 import Darknet
from torch import optim
from tensorboardX import SummaryWriter
from datasets.Customdata import ImageFolder, collate_fn


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v3')
    parser.add_argument('--max_epochs', dest='max_epochs', help='number of epochs to train', default=200, type=int)

    parser.add_argument('--names', dest='names', default='./data/coco.names', type=str)

    parser.add_argument('--start_epoch', dest='start_epoch', default=1, type=int)

    parser.add_argument('--traindata', dest='traindata', default='./data/coco/trainvalno5k.txt', type=str)

    parser.add_argument("--valdata", dest="valdata", default="./data/coco/5k.txt", type=str)

    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=8, type=int)

    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")

    parser.add_argument('--output_dir', dest='output_dir', default='./outputs', type=str)

    parser.add_argument('--checkpoints', dest='checkpoints', default='./checkpoints', type=str)

    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool)

    parser.add_argument('--display_interval', dest='display_interval', default=10, type=int)

    parser.add_argument('--mGPUs', dest='mGPUs', default=False, type=bool)

    parser.add_argument('--save_interval', dest='save_interval', default=20, type=int)

    parser.add_argument('--cuda', dest='use_cuda', default=True, type=bool)

    parser.add_argument('--resume', dest='resume', default=True, type=bool)

    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch', default=180, type=int)

    parser.add_argument('--exp_name', dest='exp_name', default='default', type=str)

    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    args = parser.parse_args()
    return args


def train():
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the hyper parameters first
    args = parse_args()
    namespath = args.names
    with open(namespath, 'r') as f:
        names = f.read().split("\n")[:-1]
    trainpath = args.traindata
    valpath = args.valdata

    # args.lr = cfg.lr
    # args.decay_lrs = cfg.decay_lrs
    # args.weight_decay = cfg.weight_decay
    # args.momentum = cfg.momentum
    # args.batch_size = cfg.batch_size
    print('Called with args:')

    lr = args.lr

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # load datasets
    print('loading datasets....')
    # imdb_name = "voc_2007_trainval+voc_2012_trainval"
    # train_dataset = get_dataset(args.imdb_name)
    train_dataset = ImageFolder(trainpath, augment=True, multiscale=args.multiscale_training)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn, drop_last=True)

    print('training rois number: {}'.format(len(train_dataset)))

    # initialize the model
    tic = time.time()
    model = Darknet()
    toc = time.time()
    print('model loaded! : cost time {:.2f}s'.format(toc-tic))

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        print('resume training enable')
        resume_checkpoint_name = 'yolov2_epoch_{}.pth'.format(args.checkpoint_epoch)
        resume_checkpoint_path = os.path.join(output_dir, resume_checkpoint_name)
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

    loss_ = Yolo_loss()
    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)  # 1383

    # start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
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
            if step+1 == iters_per_epoch:
                print("inner loop change input size is: ", cfg.input_size)

            im_data, boxes, gt_classes, num_obj = next(train_data_iter)
            if args.use_cuda:
                im_data = im_data.cuda()
                boxes = boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            im_data_variable = Variable(im_data)
            # todo
            # outPut是预测结果, 为list, 分别有 box_loss, iou_loss, class_loss, h, w 如果是 false 只有三项
            # box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True)
            output = model(im_data_variable, training=True)
            output_data = output[0:3]
            h, w = output[-2:]
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

            box_loss, iou_loss, class_loss = loss_(output_data, target_data, h, w)

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
                      % (epoch, step+1, iters_per_epoch, loss_temp, lr), end='')
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
            save_name = os.path.join(output_dir, 'yolov2_epoch_{}.pth'.format(epoch))

            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr},
                save_name)


if __name__ == '__main__':
    train()









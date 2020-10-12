import os
import numpy as np
import argparse
import time
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from terminaltables import AsciiTable
from models.yolov3 import Darknet
from tensorboardX import SummaryWriter
from utils.parse_config import parse_model_config
from utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from datasets.Customdata import ImageFolder


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v3')
    parser.add_argument('--max_epochs', dest='max_epochs', help='number of epochs to train', default=200, type=int)

    parser.add_argument('--names', dest='names', default='./data/coco.names', type=str)

    parser.add_argument('--ckpt', dest='ckpt', default="./checkpoints/best_model_0.128224.pth",
                        help="resume model to load", type=str)

    parser.add_argument('--traindata', dest='traindata', default='./data/coco/trainvalno5k.txt', type=str)

    parser.add_argument("--valdata", dest="valdata", default="./data/coco/5k.txt", type=str)

    parser.add_argument('--nw', dest='num_workers', help='number of workers to load training data', default=8, type=int)

    parser.add_argument("--model_def", default="config/yolov3.cfg", help="path to model definition file", type=str)

    parser.add_argument('--output_dir', dest='output_dir', default='./outputs', type=str)

    parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints', type=str)

    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool)

    parser.add_argument('--display_interval', dest='display_interval', default=50, type=int)

    parser.add_argument('--mGPUs', dest='mGPUs', default=False, type=bool)

    parser.add_argument("--img_size", dest="img_size", default=416, help="size of each image dimension", type=int)

    parser.add_argument('--resume', dest='resume', default=True, type=bool)

    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    parser.add_argument("--pretrained_weights", default="./pretrained/darknet53.conv.74",
                        type=str, help="if specified starts from checkpoint model")

    args = parser.parse_args()
    return args


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_mAP = -np.inf

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
        # net = nn.DataParallel(net)

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
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=decay)
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(start_epoch, args.max_epochs):
        train(epoch, net, train_dataloader, optimizer, args)
        val(epoch, args, net, val_dataloader, 0.5, conf_thresh=0.5, nms_thresh=0.5, img_size=args.img_size)
        torch.cuda.empty_cache()


def train(epoch, model, train_dataloader, optimizer, args):
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
    model.train()
    # start_time = time.time()
    for batch_i, (_, imgs, targets) in enumerate(train_dataloader):
        if CUDA:
            imgs = imgs.cuda()
            targets = targets.cuda()

        loss, outputs = model(imgs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % args.display_interval == 0:
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.max_epochs, batch_i, len(train_dataloader))
            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats['grid_size'] = "%2d"
                formats['cls_acc'] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

            log_str += AsciiTable(metric_table).table
            log_str += f"\n Total loss {loss.item()}"
            print(log_str)
            model.seen += imgs.size(0)
        torch.cuda.empty_cache()


def val(epoch, args, model, val_dataloader, iou_thresh, conf_thresh, nms_thresh, img_size, batch_size=8):
    global best_mAP
    print("begin to val the datasets...")
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels = []
    sample_metrics = []

    for batch_i, (_, imgs, targets) in enumerate(tqdm(val_dataloader, desc="detection the objections:")):
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thresh, nms_thres=nms_thresh)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thresh)

    tp, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(tp, pred_scores, pred_labels, labels)
    val_precision = precision.mean()
    val_recall = recall.mean()
    val_f1 = f1.mean()
    val_mAP = AP.mean()
    print("precision: %.3f, recall: %.3f, f1: %.3f, mAP: %.3f" % (val_precision, val_recall, val_f1, val_mAP))
    if val_mAP > best_mAP:
        best_mAP = val_mAP
        save_name = os.path.join(args.save_dir, "best_model_%.6f.pth" % best_mAP)

        state_dict = model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            "model": state_dict,
            "epoch": epoch + 1
        }, save_name)
        print("model has been saved in %s" % save_name, end="")

    return precision, recall, AP, f1, ap_class


if __name__ == '__main__':
    main()










import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import random_split, DataLoader
from datasets.Customdata import CustomData
from PIL import Image
from models.fcn import FCNs
import torch.optim as optim
import argparse


CUDA = torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser(description="FCNs")

    parser.add_argument("--start_epoch", dest='start_epoch', default=1, type=int)

    parser.add_argument("--num_works", dest="num_works", help="num of data loading workers ", default=1, type=int)

    parser.add_argument("--epochs", dest="epochs", help="number of epochs (default: 80)", default=80, type=int)

    parser.add_argument("--batch_size", dest="batch_size", help="number of batch (default: 32)", default=16, type=int)

    parser.add_argument("--resume", dest="resume", help="resume training(default: False)", default=False, type=bool)

    parser.add_argument("--ckpt", dest="ckpt", help="load checkpoint model ", default="./checkpoints/053.pth")

    parser.add_argument("--num_classes", dest="num_classes", help="number of classes", default=2, type=int)

    parser.add_argument("--back_bone", dest="back_bone", help="backbone network to extract feature", default='vgg')

    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-2, type=float)

    parser.add_argument("--verbose", dest='verbose', help="per verbose display the result", default=10, type=int)

    parser.add_argument("--save_dir", dest="save_dir", help="save directory", default="./checkpoints", type=str)

    parser.add_argument("--num_save", dest="num_save", help="save models at intervals (default: 1)", default=5, type=int)

    parser.add_argument("--mode", dest='mode', help="mode to run the code (default train)", default="train", type=str)

    parser.add_argument("--multiGPU", dest='mGPUs', default=False, type=bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    assert args.mode in ['train', 'test']
    net = FCNs(args.num_classes, args.back_bone)
    start_epoch = 0

    # 加载之前的模型继续训练
    if args.resume:
        ckpt = args.ckpt
        # resume the model
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch']
        start_epoch = args.start_epoch
        print('load the model from {} and start epoch is: {}'.format(args.ckpt, args.start_epoch))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    """
    第一种模型保存方法
    如果用这种保存模型会比较麻烦,因为要考虑到多卡的情况或者是单卡的情况,在后面的模型保存那里
    也需要进行判断是否使用了多卡
    """
    # if CUDA and args.mGPUs:
    #     net.to(torch.device("cuda"))
    #     net = nn.DataParallel(net)
    # elif CUDA:
    #     net.to(torch.device("cuda"))
    """
    第二种模型保存的方法
    最常用的和方便的写法,保存模型可以适应单卡和多卡的情况(推荐使用)
    """
    if CUDA:
        net.to(torch.device("cuda"))
        net = nn.DataParallel(net)

    # loading the datasets
    customdata = CustomData("/home/bruce/PycharmProjects/CV-Papers-Codes/FCN/data/BagImages", mode="train")
    train_size = int(0.9 * len(customdata))
    val_size = len(customdata) - train_size
    train_set, val_set = random_split(customdata, [train_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # 二分类

    print('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(args.epochs, args.batch_size, len(train_set), len(val_set)))

    for epoch in range(start_epoch, args.epochs):
        train(net, train_dataloader, epoch, criterion, optimizer)
        validate(net, val_dataloader, criterion)


def train(net, train_dataloader, epoch, criterion, optimizer):
    start_time = time.time()
    net.train()
    args = parse_args()
    epoch_loss = 0.
    batches = 0
    for i, samples in enumerate(train_dataloader):
        image, target = samples
        if CUDA:
            image = image.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = net(image)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

        if (i + 1) % args.verbose == 0:
            print("Epoch %03d, Learning Rate %g , Training Loss: %.6f" % (
            epoch + 1, optimizer.param_groups[0]["lr"], epoch_loss / batches))

    if epoch % args.num_save == 0:
        """
        第一种保存模型的方式
        """
        save_name = os.path.join(args.save_dir, "fcn_epoch_%03d_loss_%.6f.pth" % (epoch + 1, epoch_loss / batches))
        print(save_name)
        # torch.save({
        #     "model": net.module.state_dict() if args.mGPUs else net.state_dict(),
        #     "epoch": epoch+1,
        #     "lr": optimizer.param_groups[0]["lr"]
        # }, save_name)

        """
        第二种模型保存的方法
        """
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            "epoch": epoch+1,
            "model": state_dict,
            "lr": optimizer.param_groups[0]['lr']
        }, save_name)

    end_time = time.time()
    print('Train Loss: %.6f Time: %d' % (epoch_loss, end_time - start_time))


def validate(net, val_dataloader, criterion):
    start_time = time.time()
    epoch_loss = 0.0
    net.eval()
    for i, sample in enumerate(val_dataloader):
        image, target = sample
        if CUDA:
            image = image.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = net(image)
            loss = criterion(output, target)
        pred = output.data.cpu().numpy()
        pred = np.argmin(pred, axis=1)
        t = np.argmin(target.cpu().numpy(), axis=1)

        epoch_loss += loss.item()
    end_time = time.time()
    print("Val Loss: %.6f Time: %d" % (epoch_loss, end_time - start_time))


if __name__ == "__main__":
    main()

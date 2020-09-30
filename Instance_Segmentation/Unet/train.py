import os
import torch
import time
import sys
import numpy as np
import argparse
import torch.nn as nn
from model.models import UNet
from model.loss import calc_loss
from torch.utils.data import DataLoader, random_split
from dataset.Customdata import CustomData
import torch.optim as optim
# 参考链接
# https://github.com/leijue222/portrait-matting-unet-flask
# https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/losses.py
# https://zhuanlan.zhihu.com/p/44958351
# Deep Layer Aggregation
# https://zhuanlan.zhihu.com/p/103426335


CUDA = torch.cuda.is_available()
best_loss = np.inf


def parse_args():

    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("--start_epoch", dest='start_epoch', default=1, type=int)

    parser.add_argument("--num_works", dest="num_works", help="num of data loading workers ", default=4, type=int)

    parser.add_argument("--epochs", dest="epochs", help="number of epochs (default: 80)", default=500, type=int)

    parser.add_argument("--batch_size", dest="batch_size", help="number of batch (default: 16)", default=16, type=int)

    parser.add_argument("--resume", dest="resume", help="resume training(default: False)", default=True, type=bool)

    parser.add_argument("--ckpt", dest="ckpt", help="load checkpoint model ", default="./checkpoints/best_model_2.075422.pth")

    parser.add_argument("--num_classes", dest="num_classes", help="number of classes", default=1, type=int)

    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-4, type=float)

    parser.add_argument("--verbose", dest='verbose', help="per verbose display the result", default=10, type=int)

    parser.add_argument("--save_dir", dest="save_dir", help="save directory", default="./checkpoints", type=str)

    parser.add_argument("--num_save", dest="num_save", help="save models at intervals (default: 1)", default=5, type=int)

    parser.add_argument("--mode", dest='mode', help="mode to run the code (default train)", default="train", type=str)

    parser.add_argument("--multiGPU", dest='mGPUs', default=False, type=bool)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    net = UNet(n_channels=3, n_classes=args.num_classes)
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

    if CUDA:
        net.to(torch.device("cuda"))
        net = nn.DataParallel(net)

    customdata = CustomData("/home/bruce/bigVolumn/Datasets/human_instance_segment", mode="train")
    train_size = int(0.9 * len(customdata))
    val_size = len(customdata) - train_size
    train_set, val_set = random_split(customdata, [train_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.000001)

    print('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(args.epochs, args.batch_size, len(train_set), len(val_set)))

    for epoch in range(start_epoch, args.epochs+1):
        train(net, train_dataloader, epoch, optimizer)
        validate(net, val_dataloader, epoch, args)


def train(net, train_dataloader, epoch, optimizer):
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

        # [batch, 2, 256, 256]
        loss = calc_loss(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

        if (i + 1) % args.verbose == 0:
            print("Epoch %03d, Learning Rate %g , Training Loss: %.6f" % (
            epoch + 1, optimizer.param_groups[0]["lr"], epoch_loss / batches))
    end_time = time.time()
    print('Train Loss: %.6f Time: %d' % (epoch_loss/batches, end_time - start_time))


def validate(net, val_dataloader, epoch, args):
    global best_loss
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
            loss = calc_loss(output, target)
        # pred = output.data.cpu().numpy()
        # pred = np.argmin(pred, axis=1)
        # t = np.argmin(target.cpu().numpy(), axis=1)
        epoch_loss += loss.item()

    if best_loss > epoch_loss:
        save_name = os.path.join(args.save_dir, "best_model_%.6f.pth" % epoch_loss)

        best_loss = epoch_loss
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            "model": state_dict,
            "epoch": epoch+1
        }, save_name)
        print("model has been saved in %s" % save_name, end="")

    end_time = time.time()
    print(", Val Loss: %.6f Time: %d" % (epoch_loss, end_time - start_time))


if __name__ == "__main__":
    main()



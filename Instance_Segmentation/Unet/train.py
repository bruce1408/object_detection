import os
import torch
import time
import sys
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from model.models import UNet
from dataset.Customdata import BasicDataset
from torch.utils.data import DataLoader, random_split
# 参考链接 https://github.com/leijue222/portrait-matting-unet-flask


def parse_args():

    parser = argparse.ArgumentParser(description="Unet")

    parser.add_argument("--num_works", dest="num_works", help="num of data loading workers ", default=16, type=int)

    parser.add_argument("--epochs", dest="epochs", help="number of epochs (default: 80)", default=500, type=int)

    parser.add_argument("--batch_size", dest="batch_size", help="number of batch (default: 16)", default=4, type=int)

    parser.add_argument("--resume", dest="resume", help="resume training(default: False)", default=False, type=bool)

    parser.add_argument("--ckpt", dest="ckpt", help="load checkpoint model ", default="./checkpoints/fcn_epoch_076_loss_0.395389.pth")

    parser.add_argument("--num_classes", dest="num_classes", help="number of classes", default=1, type=int)

    parser.add_argument("--back_bone", dest="back_bone", help="backbone network to extract feature", default='vgg')

    parser.add_argument("--lr", dest="lr", help="learning rate", default=1e-2, type=float)

    parser.add_argument("--verbose", dest='verbose', help="per verbose display the result", default=10, type=int)

    parser.add_argument("--save_dir", dest="save_dir", help="save directory", default="./checkpoints", type=str)

    parser.add_argument("--num_save", dest="num_save", help="save models at intervals (default: 1)", default=5, type=int)

    parser.add_argument("--mode", dest='mode', help="mode to run the code (default train)", default="train", type=str)

    parser.add_argument("--multiGPU", dest='mGPUs', default=False, type=bool)

    args = parser.parse_args()

    return args


CUDA = torch.cuda.is_available()
best_loss = np.inf


def main():
    args = parse_args()
    net = UNet(n_channels=3, n_classes=args.num_classes)
    start_epoch = 0
    # 加载之前的模型继续训练
    if args.resume:
        ckpt = args.ckpt
        checkpoints = torch.load(ckpt)
        state_dict = checkpoints['model']
        net.load_state_dict(state_dict)
        start_epoch = checkpoints['epoch']
        print('load the model from {} and start epoch is: {}'.format(args.ckpt, args.start_epoch))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if CUDA:
        print("use GPU to train the models")
        net.to(torch.device('cuda'))
        net = nn.DataParallel(net)

    customdata = BasicDataset("/home/bruce/bigVolumn/Datasets/human_instance_segment/training",
                 "/home/bruce/bigVolumn/Datasets/human_instance_segment/trainMask", mode='train', scale=0.5)

    train_size = int(0.9*len(customdata))
    val_size = len(customdata) - train_size
    train_set, val_set = random_split(customdata, [train_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    # val_data = BasicDataset("/home/bruce/bigVolumn/Datasets/human_instance_segment/testing")

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.000001)
    criterion = nn.BCEWithLogitsLoss()

    print("Start training: Total epochs: {}, Batch size: {}, Training size, Validation size: {}".
          format(args.epochs, args.batch_size, len(train_set), len(val_set)))

    for epoch in range(start_epoch, args.epochs+1):
        train(net, train_dataloader, epoch, criterion, optimizer, args)
        validate(net, val_dataloader, criterion, epoch, args)


def train(net, train_dataloader, epoch, criterion, optimizer, args):
    start_time = time.time()
    net.train()
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
        optimizer.step()

        epoch_loss += loss.item()
        batches += 1

        if (i+1) % args.verbose == 0:
            print("Epoch %03d, Learning Rate %g , Training Loss: %.6f" % (
            epoch + 1, optimizer.param_groups[0]["lr"], epoch_loss / batches))

    end_time = time.time()
    print('Train Loss: %.6f Time: %d' % (epoch_loss/batches, end_time - start_time))


def validate(net, val_dataloader, criterion, epoch, args):
    global best_loss
    net.eval()
    start_time = time.time()
    epoch_loss = 0.
    for i, samples in enumerate(val_dataloader):
        image, target = samples
        if CUDA:
            image = image.cuda()
            target = target.cuda()
        with torch.no_grad():
            output = net(image)
            loss = criterion(output, target)
        pred = output.data.cpu().numpy()




if __name__ == "__main__":
    main()


# torch.save({'epoch': epochID + 1,
#             'state_dict': model.state_dict(),
#             'best_loss': lossMIN,
#             'optimizer': optimizer.state_dict(),
#             'alpha': loss.alpha,
#             'gamma': loss.gamma},
#            checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
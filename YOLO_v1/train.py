# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/12 18:30
@Author        : Bruce
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""
import warnings
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models as officalModel

from models.vgg_yolo import vgg16_bn
from models.resnet_yolo import resnet50
from models.yoloLoss import yoloLoss
from utils.dataset import yoloDataset

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 判断GPU是否可用
use_gpu = torch.cuda.is_available()

# 数据文件
# file_root = '/home/bruce/PycharmProjects/yolov1_pytorch/datasets'
file_root = "/home/chenxi/object_detection/data/yolo_v1_datasets/"

# 超参数
learning_rate = 0.001
num_epochs = 100
batch_size = 32
resume = False

# ---------------------数据读取---------------------
train_dataset = yoloDataset(
    root=file_root, list_file='images.txt', train=True, transform=[transforms.ToTensor()])

# val_dataset = yoloDataset(root=file_root, list_file='voc2007test.txt', train=False, transform=[transforms.ToTensor()])

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print('the train dataset has %d images' % (len(train_dataset)))
# print('the test dataset has %d images' % (len(val_dataset)))
print('the batch_size is %d' % batch_size)


# --------------------- 网络选择 ----------------------
use_resnet = True
if use_resnet:
    net = resnet50()  # 自己定义的resnet 网络
else:
    net = vgg16_bn()
"""
models 模型表示官方代码
自己定义的是resnet50
"""
if resume:
    print("loading weight from checkpoints/best.pth")
    net.load_state_dict(torch.load('checkpoints/best.pth'))
else:
    print('loading pre-trained model ......')
    if use_resnet:  # 对应自己网络结构的加载官方预训练模型参数
        offiNet = officalModel.resnet50(pretrained=True)  # 官方模型
        offiNew_state_dict = offiNet.state_dict()
        dd = net.state_dict()
        for k in offiNew_state_dict.keys():
            print(k)
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = offiNew_state_dict[k]
        net.load_state_dict(dd)
    else:
        vgg = officalModel.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and k.startswith('features'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)

if use_gpu:
    print('this computer has gpu %d and current is %s' %
          (torch.cuda.device_count(), torch.cuda.current_device()))
    net.cuda()


# ---------------------损失函数---------------------
criterion = yoloLoss(7, 2, 5, 0.5)
# ---------------------优化器----------------------
# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr':learning_rate*0.5}]
    else:
        params += [{'params': [value], 'lr':learning_rate}]
optimizer = torch.optim.SGD(
    params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)


# ---------------------训练---------------------
logfile = open(
    '/home/chenxi/object_detection/YOLO_v1/checkpoints/log.txt', 'w')
num_iter = 0
best_test_loss = np.inf

for epoch in range(num_epochs):
    # train
    net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.

    for i, (images, target) in enumerate(train_loader):
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        # print('the pred shape is: ', pred.shape)  # [batch, 14, 14, 30]
        # print('target shape ', target.shape)  # [batch, 14, 14, 30]
        loss = criterion(pred, target)
        total_loss += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' %
                  (epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1

    # validation
    print("================= begin to validation ====================")
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(val_loader):
        if use_gpu:
            images, target = images.cuda(), target.cuda()

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.item()
    validation_loss /= len(val_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), 'checkpoints/best.pth')  # 只保留参数,速度更快
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()

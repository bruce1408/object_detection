import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import random_split, DataLoader
from datasets.Customdata import CustomData
from PIL import Image
from models.fcn import FCNs
from train import parse_args
import torch.optim as optim
import argparse

imagePath = "/home/bruce/PycharmProjects/CV-Papers-Codes/FCN/data/testImages"


def modelTest():
    batch_size = 16
    back_bone = 'vgg'
    num_classes = 2
    modelPath = "./checkpoints/fcn_epoch_110.pth"
    net = FCNs(num_classes, back_bone)
    net.eval()

    checkpoint = torch.load(modelPath)
    net.load_state_dict(checkpoint['model'])
    CUDA = torch.cuda.is_available()
    if CUDA:
        net.to(torch.device('cuda'))
        net = nn.DataParallel(net)

    print('load the models from %s and epoch is %d' % (modelPath, checkpoint['epoch']))

    test_set = CustomData(imagePath, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for i, sample in enumerate(test_loader):
        image, path = sample
        if CUDA:
            image = image.cuda()

        with torch.no_grad():
            output = net(image)

        pred = output.data.cpu().numpy()
        print(pred.shape)
        pred = np.argmin(pred, axis=1)
        for j, p in enumerate(path):
            im = Image.fromarray(pred.astype('uint8')[j]*255, "L")
            im.save(os.path.join('./output', p.split("/")[-1]))
    print("pred result is over!")


if __name__ == "__main__":
    modelTest()
# CUDA = torch.cuda.is_available()
#
# def modelTest(**kwargs):
#     data_loader = kwargs['data_loader']
#     mymodel = kwargs['mymodel']
#
#     start_time = time.time()
#     mymodel.eval()
#
#     for i, sample in enumerate(data_loader):
#         image, path = sample
#         if CUDA:
#             image = image.cuda()
#
#         with torch.no_grad():
#             output = mymodel(image)
#
#         pred = output.data.cpu().numpy()
#         pred = np.argmin(pred, axis=1)
#         for j, p in enumerate(path):
#             im = Image.fromarray(pred.astype('uint8')[j] * 255, "L")
#             im.save(os.path.join("./output", p.split("/")[-1]))
#
#     end_time = time.time()
#     print('Testing Time: %d s' % (end_time - start_time))
#
# if __name__ == "__main__":
#     mymodel = FCNs(2, "vgg")
#     # checkpoint
#     modelPath = "./checkpoints/fcn_epoch_150.pth"
#
#     checkpoint = torch.load(modelPath)
#     state_dict = checkpoint["model"]
#
#     mymodel.load_state_dict(state_dict)
#     print(f"Model loaded from {modelPath}")
#
#     # initialize model-saving directory
#
#     if CUDA:
#         mymodel.to(torch.device("cuda"))
#         mymodel = nn.DataParallel(mymodel)
#         test_set = CustomData(imagePath, mode="test")
#         test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
#
#     modelTest(mymodel=mymodel, data_loader=test_loader)


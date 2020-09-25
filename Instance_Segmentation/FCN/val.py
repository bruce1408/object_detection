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
    batch_size = 8
    back_bone = 'vgg'
    num_classes = 2
    modelPath = "./checkpoints/fcn_epoch_076_loss_0.389963.pth"
    net = FCNs(num_classes, back_bone)
    checkpoint = torch.load(modelPath)
    net.load_state_dict(checkpoint['model'])
    CUDA = torch.cuda.is_available()
    if CUDA:
        net.to(torch.device('cuda'))
        net = nn.DataParallel(net)

    print('load the models from %s and epoch is %d' % (modelPath, checkpoint['epoch']))

    test_set = CustomData(imagePath, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    net.eval()

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




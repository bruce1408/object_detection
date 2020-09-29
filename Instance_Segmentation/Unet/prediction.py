import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from dataset.Customdata import CustomData
from PIL import Image
from model.models import UNet
from train import parse_args
import torch.optim as optim
import argparse

imagePath = "/home/bruce/bigVolumn/Datasets/human_instance_segment"
# modelPath = "./checkpoints/fcn_epoch_180.pth"
modelPath = "./checkpoints/best_model_0.355226.pth"


def modelTest():
    batch_size = 16
    num_classes = 1
    net = UNet(3, num_classes)
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
            output = output.squeeze(1)
        output = F.sigmoid(output)
        pred = output.data.cpu().numpy()
        # print(pred.shape)
        pred[np.where(pred < 0.4)] = 0
        pred[np.where(pred >= 0.4)] = 1
        for j, p in enumerate(path):
            im = Image.fromarray(pred.astype('uint8')[j]*255, "L")
            im.save(os.path.join('./output', p.split("/")[-1]))
    print("pred result is over!")


if __name__ == "__main__":
    modelTest()

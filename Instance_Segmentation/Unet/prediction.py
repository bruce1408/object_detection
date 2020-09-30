import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from dataset.Customdata import CustomData
from model.models import UNet
import matplotlib.pyplot as plt
from train import parse_args
import torch.optim as optim
from PIL import Image


imagePath = "/home/bruce/bigVolumn/Datasets/human_instance_segment"
# modelPath = "./checkpoints/fcn_epoch_180.pth"
modelPath = "./checkpoints/best_model_0.355226.pth"
# modelPath = "./checkpoints/best_model_0.440269.pth"


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
    indexImg = 0
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
        pred[np.where(pred < 0.04)] = 0
        pred[np.where(pred >= 0.04)] = 1
        for j, p in enumerate(path):
            imlist = list()
            realMask = p.split("/")[-1].split(".")[0]+"_matte.png"
            imgpath = os.path.join(imagePath, "testMask", realMask)
            img1 = Image.open(imgpath).resize((300, 300))
            imlist.append(img1)
            im = Image.fromarray(pred.astype('uint8')[j]*255, "L")
            imlist.append(im)
            im.save(os.path.join('./output', p.split("/")[-1]))
            savename = os.path.join("./outputCompare", str(indexImg) + ".png")
            plot_img(imlist, savename)
            indexImg += 1

    print("pred result is over!")


def plot_img(im, savename):
    """
    第一种画图方式
    """
    # toImage = Image.new('RGB', (600, 300))
    # toImage.paste(im[0], (0, 0))
    # toImage.paste(im[1], (300, 0, 600, 300))
    # toImage.save(savename)
    """
    第二种画图方式
    """
    plt.figure()
    title = ['realMask', 'predMask']
    for i in range(1, 3):
        plt.subplot(1, 2, i)
        plt.title(title[i-1])
        plt.imshow(im[i-1], cmap='gray')
    plt.savefig(savename)
    plt.close()


if __name__ == "__main__":
    modelTest()

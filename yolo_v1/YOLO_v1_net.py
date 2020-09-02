import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50
from torchsummary import summary
NUM_BBOX = 2
"""
官方使用ResNet50模型，然后修改层数的最后两层
"""
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

class YOLO_v1(nn.Module):
    def __init__(self):
        super(YOLO_v1, self).__init__()
        model = resnet50(pretrained=True)
        self.in_size = model.fc.in_features

        self.features = nn.Sequential(*list(model.children())[:-2])

        # 4个卷积层,输入尺寸是[2, 2048, 14, 14],从这里开始以及开始自己定义的网络结构了.
        self.Conv_layers = nn.Sequential(
            # 卷积层1
            nn.Conv2d(self.in_size, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            # 卷积层2
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),  # [7, 7, 1024]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            # 卷积层3
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            # 卷积层4
            nn.Conv2d(1024, 1024, 3, padding=1),  # [7, 7, 1024],当stride=1, pad=1, kernel=3,尺寸不变
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )

        # 2个全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),

            nn.Linear(4096, 7*7*30),
            nn.Sigmoid())

    def forward(self, input):
        # input 结果 [8, 2048, 14, 14]
        input = self.features(input)

        # input 结果 [8, 1024, 7, 7]
        input = self.Conv_layers(input)
        input = input.view(input.size()[0], -1)
        input = self.fc_layers(input)
        return input.reshape(-1, (5 * NUM_BBOX + len(CLASSES)), 7, 7)


if __name__ == "__main__":
    model = YOLO_v1()
    if torch.cuda.is_available():
        summary(model.cuda(), (3, 448, 448))
        x = torch.rand(size=(8, 3, 448, 448)).to("cuda")
        output = model(x)
        print("output shape is: ", output.size())
    else:
        summary(model, (3, 448, 448))





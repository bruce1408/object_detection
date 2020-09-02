import xml.etree.ElementTree as ET
import os

import cv2

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

DATASET_PATH = "/home/chenxi/dataset/VOCdevkit/VOC2012/"


def convert(size, box):
    """
    将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    xcenter = x * dw
    ycenter = y * dh
    h = h * dh
    w = w * dw
    return xcenter, ycenter, w, h


def convert_annotation(image_id):
    """
    把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化
    """
    in_file = open(DATASET_PATH + 'Annotations/%s' % (image_id)+".xml")
    image_id = image_id.split('.')[0]
    out_file = open('labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                  float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir(DATASET_PATH + 'Annotations')
    for file in filenames:
        convert_annotation(file)


def show_labels_img(imgname):
    """
    imgname是输入图像的名称，无下标
    """
    img = cv2.imread(DATASET_PATH + "JPEGImages/" + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w, h)
    with open("./labels/" + imgname + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))  # 左上角坐标
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))  # 右下角坐标
            cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 33, 139, 3))

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    with open("/home/chenxi/dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt") as f:
        for eachline in f:
            eachline = eachline.strip()
            convert_annotation(eachline)

    # show_labels_img("2008_000008")




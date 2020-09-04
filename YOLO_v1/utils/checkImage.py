import cv2
import os
"""
检查图片是否存在
"""
labels = "/home/bruce/PycharmProjects/yolov1_pytorch/datasets"
img = os.path.join(labels, "voc2007test.txt")
index = 0
with open(img, 'r') as f:
    for eachline in f:
        eachline = eachline.strip()
        imgname = eachline.split(' ')[0]
        imgpath = os.path.join(labels, "images", '%s' % imgname)
        if os.path.exists(imgpath):
            pass
        else:
            print(imgname)
            index += 1
print(index)
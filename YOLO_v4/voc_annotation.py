import xml.etree.ElementTree as ET
from os import getcwd

"""
转化xml到txt文件
从图片中生成训练预测数据.
"""
sets = [('1016', 'train'), ('1016', 'val'), ('1016', 'test')]
wd = getcwd()  # 获得当前路径
classes = ["person"]


def convert_annotation(year, image_id, list_file):
    in_file = open('./VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    list_file.write('./VOCdevkit/VOC2007/JPEGImages/%s.jpg' % (image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')


for year, image_set in sets:
    image_ids = open('./VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()

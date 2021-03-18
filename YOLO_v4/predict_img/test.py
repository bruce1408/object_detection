import os
import random

# xmlfilepath=r"/home/rkb/disk/data/VOC2007/Annotations"   #人头数据集
# saveBasePath=r"/home/rkb/disk/data/VOC2007/Imagesets/Main/"

jpgfilepath = r'/home/chenxi/deepfashion_detection/videoToimg_v1'
saveBasePath = r'../predict_result'

# if not os.path.exists(saveBasePath):
# 	os.makedirs(saveBasePath)

# trainval_percent = 0.9  # 取出10%的数据作为测试集
# train_percent = 1
test_percent = 1

temp_jpg = os.listdir(jpgfilepath)
total_jpg = []
print('temp_jpg\n', temp_jpg)

for jpg in temp_jpg:
    if jpg.endswith(".jpg"):
        total_jpg.append(jpg)
print('total_jpg\n', total_jpg)

num = len(total_jpg)
print('num\n', num)

list = range(num)

tt = int(num * test_percent)
test = random.sample(list,tt)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# print('tv\n', tv)
# print('tr\n', tr)
#
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)
#
# print("train and val size\n", tv)
# print("traub suze\n", tr)

# ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
# ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
# ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
# fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test2.txt'),'w')

for i in list:
    name = total_jpg[i][:-4] + '\n'

    if i in test:
        ftest.write(name)

ftest.close()


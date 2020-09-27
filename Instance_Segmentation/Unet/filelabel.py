import os
from shutil import copyfile, move
imgpath = "/home/chenxi/dataset/human_mask/training"
labeldir = "/home/chenxi/dataset/human_mask/trainMask"
if not os.path.exists(labeldir):
    os.makedirs(labeldir)

filename = os.listdir(imgpath)
for i in filename:
    if i.find('matte') > 0:
        oldpath = os.path.join(imgpath, i)
        labelpath = os.path.join(labeldir, i)
        move(oldpath, labelpath)
        print(labelpath)




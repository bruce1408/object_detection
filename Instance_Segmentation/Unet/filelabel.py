import os
from shutil import copyfile, move
imgpath = "/home/bruce/bigVolumn/Datasets/human_instance_segment/testing"
labeldir = "/home/bruce/bigVolumn/Datasets/human_instance_segment/testMask"
if not os.path.exists(labeldir):
    os.makedirs(labeldir)

filename = os.listdir(imgpath)
for i in filename:
    if i.find('matte') > 0:
        oldpath = os.path.join(imgpath, i)
        labelpath = os.path.join(labeldir, i)
        move(oldpath, labelpath)
        print(labelpath)




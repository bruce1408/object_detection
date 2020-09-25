import cv2
from PIL import Image
import numpy as np

imgpath = "/home/bruce/bigVolumn/Downloads/background.jpg"

img = cv2.imread(imgpath)  # h, w, channel

h, w = img.shape[0:2]
print(h, w)

input_size = 448

padw, padh = 0, 0

if h > w:
    padw = (h - w) // 2
    img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), "constant", constant_values=0)
elif w > h:
    padh = (w - h) // 2
    img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), "constant", constant_values=0)









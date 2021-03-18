# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
import os
from yolo import YOLO
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2


# batch predict
yolo = YOLO()
# imgpath = "/home/bruce/bigVolumn/autolabelData/similarImg"
imgpath = "./test_img/"
resultPath = "./model_v1_img_sim"
if not os.path.exists(resultPath):
    os.makedirs(resultPath)
for index, i in tqdm(enumerate(os.listdir(imgpath))):
    img = os.path.join(imgpath, i)
    try:
        image = Image.open(img)
        r_image = np.array(yolo.detect_image(image))
        frame = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(resultPath, str(index)+'.JPEG'), frame)
        # r_image.save(os.path.join(os.path.join(resultPath, str(index)), 'JPEG'))
    except:
        print('Open Error! Try again!')
        continue
        # r_image.show()
    # yolo.close_session()

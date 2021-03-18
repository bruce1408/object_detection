import cv2
import os

txt_path = '/home/chenxi/yolov4-pytorch-master/input/detection-result_single_model/'
img_path = '/home/chenxi/deepfashion_detection/videoToimg_v1/'
output = '/home/chenxi/yolov4-pytorch-master/input/result'
img_list = os.listdir(img_path)

if not os.path.exists(output):
    os.mkdir(output)

'''
    (x1,y1)----------
    |               |
    |               |
    |               |
    ----------(x2,y2)
'''

font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
white = (255, 255, 255)
light_blue = (255, 200, 100)
green = (0, 255, 0)
light_red = (30, 30, 255)

for i in img_list:
    img = cv2.imread(img_path + i)
    f = open(txt_path + i.split('.')[0] + '.txt', 'r', encoding='utf-8').readlines()
    print(img_path + i)
    for j in f:
        class_name = j.strip().split(' ')[0]
        score = j.strip().split(' ')[1]
        print(class_name, score)
        x1, y1, x2, y2 = j.strip().split(' ')[2:]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 根据对角坐标进行画框
        cv2.putText(img, class_name, (int(x1), int(y1) - 6), font, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(img, score, (int(x1) + 80, int(y1) - 4), font, 0.6, color, 1, cv2.LINE_AA)  # 添加标题
        # cv2.imshow('img',img)
        # cv2.waitKey()	
        print('{}---done!'.format(i))
    cv2.imwrite(output + i, img)

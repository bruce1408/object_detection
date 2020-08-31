# object_detection
### VOC 数据集
voc 数据集的框是 xmin,ymin, xmax, ymax的格式,表示的是框的坐标

yolo 数据集表示的是框的 中心坐标位置,(归一化之后的),框的宽和高(归一化之后的)

计算方式:
((xmax + xmin)/2)/w, ((ymin + ymax)/2)/h, (xmax - xmin)/w, (ymax - ymin)/h

参考文献:
https://blog.csdn.net/weixin_41010198/article/details/106072347
"""
COCO: (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)
VOC: (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
"""
anchors = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]

object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

saturation = 1.5
exposure = 1.5
hue = .1

jitter = 0.3

thresh = .6

batch_size = 16

lr = 0.0001

decay_lrs = {
    60: 0.00001,
    90: 0.000001
}

momentum = 0.9
weight_decay = 0.0005


# multi-scale training:
# {k: epoch, v: scale range}
multi_scale = True

# number of steps to change input size
scale_step = 40  # 每40次iter就开始改变输入尺寸的大小

scale_range = (0, 4)

epoch_scale = {
    1:  (0, 8),
    15: (2, 5),
    30: (1, 6),
    60: (0, 7),
    75: (0, 9)
}

input_sizes = [(320, 320),
               (352, 352),
               (384, 384),
               (416, 416),
               (448, 448),
               (480, 480),
               (512, 512),
               (544, 544),
               (576, 576)]

input_size = (416, 416)

test_input_size = (416, 416)

strides = 32

debug = False


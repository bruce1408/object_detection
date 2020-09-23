import cv2
import numpy as np
from PIL import Image
import config.config as cfg
np.random.seed(0)


def random_scale_translation(img, boxes, jitter=0.2):
    """
    把图像和框都按照一定比例进行随机缩放,不会对框进行归一化
    Arguments:
    img -- PIL.Image
    boxes -- numpy array of shape (N, 4) N is number of boxes
    factor -- max scale size
    im_info -- dictionary {width:, height:}

    Returns:
    im_data -- numpy.ndarray
    boxes -- numpy array of shape (N, 4)
    """

    w, h = img.size

    dw = int(w*jitter)
    dh = int(h*jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    # scaled width, scaled height
    sw = w - pl - pr
    sh = h - pt - pb

    # 经过转换之后,图像大小变成 w - pr - 1, h - pb - 1
    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))
    # boxes = [ 155, 96, 350, 269]
    # update boxes accordingly
    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt

    # clamp boxes
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw-1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh-1)

    # if flip
    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw-1) - boxes[:, 2::-2]

    return cropped, boxes


def convert_color(img, source, dest):
    """
    Convert color

    Arguments:
    img -- numpy.ndarray
    source -- str, original color space
    dest -- str, target color space.

    Returns:
    img -- numpy.ndarray
    """

    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def rand_scale(s):
    scale = np.random.uniform(1, s)  # 均匀分布采样
    if np.random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort(img, hue=.1, sat=1.5, val=1.5):
    """
    图片颜色进行随机变换
    :param img:
    :param hue:
    :param sat:
    :param val:
    :return:
    """
    # 色调值随机选择
    hue = np.random.uniform(-hue, hue)

    # 饱和度设置
    sat = rand_scale(sat)

    # 亮度设置
    val = rand_scale(val)

    # 把RGB颜色空间转换成HSV颜色空间,这样的好处是HSV颜色空间对色调,饱和度,亮度值进行调整.图像处理领域用的更多
    img = img.convert('HSV')
    cs = list(img.split())

    # 更改饱和度和亮度
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))

    img = img.convert('RGB')
    return img


def random_hue(img, rate=.1):
    """
    adjust hue
    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue
    Returns:
    img -- numpy.ndarray
    """

    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    """
    adjust saturation

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    """
    adjust exposure (In fact, this function change V (HSV))

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img


def augment_img(img, boxes, gt_classes):
    """
    图像增强模块,只是对boxes随机拉伸，并不会进行归一化操作
    Apply data augmentation.
    1. convert color to HSV
    2. adjust hue(.1), saturation(1.5), exposure(1.5)
    3. convert color to RGB
    4. random scale (up to 20%)
    5. translation (up to 20%)
    6. resize to given input size.

    Arguments:
    img -- PIL.Image object
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    im_info -- dictionary {width:, height:}

    Returns:
    au_img -- numpy array of shape (H, W, 3)
    au_boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    au_gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    """

    # img = np.array(img).astype(np.float32)
    boxes = np.array(boxes)
    gt_classes = np.array(gt_classes)
    boxes = np.copy(boxes).astype(np.float32)

    for i in range(5):
        # random_scale_translation 图片进行缩放, 框也相应的进行等比例缩放
        img_t, boxes_t = random_scale_translation(img.copy(), boxes.copy(), jitter=cfg.jitter)

        # x1 != x2 并且 y1 != y2
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (boxes_t[:, 1] != boxes_t[:, 3])

        # True 保留整个boxes, False则设置为空
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            gt_classes = gt_classes[keep]
            break

    # 图像进行随机拉伸
    img = random_distort(img, cfg.hue, cfg.saturation, cfg.exposure)
    return img, boxes, gt_classes









import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from ultralytics import YOLO
from torchvision import transforms
import os
from glob import glob
import cv2
import numpy as np
def pre_transform(img, imgsz, auto, stride, scaleFill=False, scaleup=True, center=True):
    # 获取原始图像的形状
    shape = img.shape[1], img.shape[2]  # Height, Width
    new_shape = imgsz  # Desired size (Height, Width)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # New width, height after scaling

    # 计算边框填充大小
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # Divide padding into two sides
    dh /= 2

    # 缩放图像
    if shape[::-1] != new_unpad:  # 如果尺寸不匹配则需要缩放
        img = F.interpolate(img.unsqueeze(0), size=new_unpad, mode="bilinear", align_corners=False).squeeze(0)

    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    return  F.pad(img, (top,bottom,left,right), value=114 )  # 使用 114 灰度填充边框


def preprocess(image,half=True,imgsz=(224,224),auto=True,stride=32):
    image = pre_transform(image,imgsz,auto,stride).unsqueeze(0).contiguous()
    if half:
        image = image.half()
    return image/255

def my_classify_preprocess(img,half=True):
    resize = transforms.Resize(224)
    crop = transforms.CenterCrop((224,224))
    normalize = transforms.Normalize((0,0,0),(1,1,1))
    img = img.permute(1,2,0)[...,[2,1,0]]
    img = crop(resize(img.permute(2,0,1)/255))
    img = normalize(img)
    return img.half() if half else img




if __name__ == "__main__":
    model = YOLO(os.path.join(os.curdir,'project','model','boundary','box.pt'),task='classify')
    rt_model = YOLO(os.path.join(os.curdir,'project',"linux",'model','boundary','box.engine'),task='classify')
    model()
    img = cv2.imread(os.path.join(os.curdir,'testB','cla','1.jpg'))
    img = torch.Tensor(img).permute(2,0,1)
    img = my_classify_preprocess(img)

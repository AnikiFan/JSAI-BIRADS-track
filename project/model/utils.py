import cv2
import numpy as np
import torch
from typing import *
from torchvision import transforms
def make_mask(heatmap):
    mask = heatmap.squeeze()
    mask = (mask < 0.95) * mask
    return (mask * 255).to(torch.uint8)
    
def cv_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x,y,w,h
    
def make_box_map(origin, mask):
    TRESH = 30
    mask = (mask > 180).to(torch.uint8) * 255
    if mask.max() == 0:
        return origin
    array = mask
    H, W = array.shape
    left_edges = torch.where(array.any(dim=1).bool(), array.argmax(dim=1), torch.tensor(W + 1, device=array.device))

# Flip horizontally (left-right)
    flip_lr = torch.flip(array, dims=[1])  # dims=[1] means flipping along the horizontal axis

# Calculate right edges
    right_edges = W - torch.where(
        flip_lr.any(dim=1).bool(), flip_lr.argmax(dim=1), torch.tensor(W + 1, device=array.device)
    )

# Calculate top edges
    top_edges = torch.where(
        array.any(dim=0).bool(), array.argmax(dim=0), torch.tensor(H + 1, device=array.device)
    )

# Flip vertically (up-down)
    flip_ud = torch.flip(array, dims=[0])  # dims=[0] means flipping along the vertical axis

# Calculate bottom edges
    bottom_edges = H - torch.where(
        flip_ud.any(dim=0).bool(), flip_ud.argmax(dim=0), torch.tensor(H + 1, device=array.device)
    )

    leftmost = left_edges.min()
    rightmost = right_edges.max()
    topmost = top_edges.min()
    bottommost = bottom_edges.max()
    leftmost = leftmost - TRESH
    if leftmost < 0:
        leftmost = 0
    rightmost = rightmost + TRESH
    if rightmost > W:
        rightmost = W
    topmost = topmost - TRESH
    if topmost < 0:
        topmost = 0
    bottommost = bottommost + TRESH
    if bottommost > H:
        bottommost = H
    return origin[:,topmost:bottommost, leftmost:rightmost]

def make_masked(heatmap,origin):
    heatmap = heatmap.squeeze(0)
    return (heatmap * origin).to(torch.uint8)

class Preprocess:
    def __init__(self,half=True):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
            transforms.Normalize((0,0,0),(1,1,1)),
        ])
        self.half = half
    def __call__(self,img):
        img = self.transform(img[[2,1,0],...]/255).unsqueeze(0)
        return img.half() if self.half else img
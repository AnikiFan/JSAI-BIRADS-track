import cv2
import numpy as np
import torch
from typing import *
def make_mask(heatmap):
    mask = heatmap.squeeze()
    mask = (mask < 0.95) * mask
    return (mask * 255).to(torch.uint8)

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

def removeFrame(x: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    对张量进行裁剪。
    :param x: 输入ndarray，形状为 (C, H, W)
    :return: 裁剪后的图像在原图像中的位置(top,left,h,w)
    裁剪某一行/列的需满足以下条件之一：
    1. 黑色比例超过阈值(黑色这里定义为三通道相同，且都小于等于20)
    2. 白色比例超过阈值(白色这里定义为三通道相同，且都大于等于250)
    3. 彩色比例超过阈值(彩色这里定义为三通道与平均值之差的绝对值与三通道平均值之比之和大于0.5)
    """
    c, h, w = x.shape
    assert c == 3 or c == 1,"input shape should be (c,h,w)!"
    if not isinstance(x,torch.Tensor):
        x = torch.Tensor(x).to('cuda')
    
    row_tolerance = h * 0.05
    col_tolerance = w * 0.05
    max_row_tolerance_time = 2
    max_col_tolerance_time = 2

    channel_mean = x.float().mean(dim=0)
    channel_diff_abs = (x-channel_mean).abs()
    epsilon = 1e-8
    channel_is_color = (channel_diff_abs.sum(dim=0)/(channel_mean+epsilon))>0.5
    channel_is_same = (channel_diff_abs ==0)
    channel_is_black = (x<=20)
    channel_is_white = (x>=250)
    tmp = channel_is_same & (channel_is_white | channel_is_black)
    pure_mask = tmp[0]
    for i in range(1, tmp.shape[0]):
        pure_mask = pure_mask | tmp[i]
    pure_row_mask = (torch.sum(pure_mask, dim=1) / w) >= 0.85
    pure_col_mask = (torch.sum(pure_mask, dim=0) / h) >= 0.80
    color_row_mask = (torch.sum(channel_is_color, dim=1) / w) >= 0.5
    color_col_mask = (torch.sum(channel_is_color, dim=0) / h) >= 0.3
    row_mask = pure_row_mask | color_row_mask
    col_mask = pure_col_mask | color_col_mask

    left, right, top, bottom = 0, w - 1, 0, h - 1

    # 左边界
    tolerance_cnt = 0
    tolerance_time = 0
    p = left
    while p < right and tolerance_time < max_col_tolerance_time:
        if col_mask[p]:
            left = p
            p += 1
            if tolerance_cnt:
                tolerance_cnt = 0
                tolerance_time += 1
        else:
            if tolerance_cnt >= col_tolerance:
                break
            tolerance_cnt += 1
            p += 1

    # 右边界
    tolerance_cnt = 0
    tolerance_time = 0
    p = right
    while p > left and tolerance_time < max_col_tolerance_time:
        if col_mask[p]:
            right = p
            p -= 1
            if tolerance_cnt:
                tolerance_cnt = 0
                tolerance_time += 1
        else:
            if tolerance_cnt >= col_tolerance:
                break
            tolerance_cnt += 1
            p -= 1

    # 上边界
    tolerance_cnt = 0
    tolerance_time = 0
    p = top
    while p < bottom and tolerance_time < max_row_tolerance_time:
        if row_mask[p]:
            top = p
            p += 1
            if tolerance_cnt:
                tolerance_cnt = 0
                tolerance_time += 1
        else:
            if tolerance_cnt >= row_tolerance:
                break
            tolerance_cnt += 1
            p += 1

    # 下边界
    tolerance_cnt = 0
    tolerance_time = 0
    p = bottom
    while p > top and tolerance_time < max_row_tolerance_time:
        if row_mask[p]:
            bottom = p
            p -= 1
            if tolerance_cnt:
                tolerance_cnt = 0
                tolerance_time += 1
        else:
            if tolerance_cnt >= row_tolerance:
                break
            tolerance_cnt += 1
            p -= 1

    if (right - left) < w * 0.4:
        left, right = 0, w - 1
    if (bottom - top) < h * 0.4:
        top, bottom = 0, h - 1

    return top,left,bottom-top,right-left

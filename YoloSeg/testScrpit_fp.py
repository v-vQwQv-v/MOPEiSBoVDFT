import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json

from models.model import yolov11
from loss_1 import YoloSegLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = cv2.imread('need_to_vaild_rgbFrame_5.png')
label = json.load(open('need_to_vaild_label_5.json', 'r'))
y0 = torch.Tensor(label['instanceSemantics']) # 需要降采样

def label_to_gtMask(y0, bgMask=-1, downSample_shape = (128, 256)):
    idxs = torch.unique(y0)
    idxs = idxs[idxs != bgMask]
    masks = [(y0 == idx).float() for idx in idxs]
    y1 = torch.stack(masks).unsqueeze(0)
    y2 = F.interpolate(y1, size=downSample_shape, mode='nearest')
    return y2

y2 = label_to_gtMask(y0, bgMask=-1, downSample_shape=(128, 256))
# Convert the image to a PyTorch tensor
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # Add batch dimension
scal = int(img_tensor.shape[2] / y2.shape[2])

model = yolov11().to(device)
loss = YoloSegLoss(dice_weight=0.5, use_focal=True).to(device)

model.train()
img_tensor = img_tensor.to(device)
# Forward pass
with torch.no_grad():
    y_ls, y_a = model(img_tensor)

def feature_to_center(out_i, nc=80, reg_max=16, stride=4, conf_thresh=0):
    """
    输入:
       out_i: [B, C, H, W] → B: batch size, C: 通道数 (nc + 4 * reg_max), H: 高, W: 宽
       nc: 分类数 (80)
       reg_max: 回归分支的最大偏移量 (16)
       stride: 特征图的下采样率 (4)
       conf_thresh: 置信度阈值 (0)
    输出:
       centers: [N, 2] → N: 有效中心点数量, 2: 中心点坐标 (cx, cy)
    说明:  
    y_a[i] → 分为回归分支 (64通道) 和分类分支 (80通道)
       → 回归分支解码出 delta → 得到 bbox 中心点 cx, cy
       → 通过 meshgrid 生成每个 grid 的 base 坐标
       → cxcy = (grid + 0.5 + delta_xy) * stride
       → 分类分支得到置信度 mask
       → 最终返回 cxcy[conf > 阈值] → centers: [N, 2]
    """

    """拆成分类分支和回归分支"""
    B = out_i.shape[0]
    H = out_i.shape[2]
    W = out_i.shape[3]
    cls_logits = out_i[:, :nc, :, :]
    reg_logits = out_i[:, nc:, :, :]

    """创建每个网格位置的基准坐标 grid"""
    yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
    grid = torch.stack((xv, yv), dim=-1).to(device)
    grid = grid.view(1, H, W, 2)

    """对回归进行 softmax + DFL 解码为 offset(delta)"""
    reg = reg_logits.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2)  # → [B, H, W, 4, reg_max]
    prob = reg.softmax(-1)
    proj = torch.linspace(0, reg_max - 1, reg_max, device=device)
    delta = (prob * proj).sum(-1)  # [B, H, W, 4]

    """
    计算中心点坐标 cx, cy:
        grid 是左上角坐标 (ix, iy)
        + 0.5 + delta 表示网格内偏移(默认中心 + 偏移)
        * stride 把它还原到原图像坐标空间
    """
    cxcy = (grid + 0.5 + delta[..., 0:2]) * stride  # [B, H, W, 2]

    """扁平化 & 筛选高置信度预测"""
    cls = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, nc)
    cxcy = cxcy.view(B, -1, 2)
    conf, _ = cls.max(-1)     # 最大分类分数
    mask = conf > conf_thresh       # 阈值筛选
    centers = cxcy[mask]            # 取保留的中心点
    return centers

pred_masks = torch.einsum('bcn,bchw->bnhw', pred_coeffs, p)

测试尾 = 1  
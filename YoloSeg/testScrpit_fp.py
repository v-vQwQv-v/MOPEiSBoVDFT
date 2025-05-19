import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from models.model import yolov11

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = cv2.imread('need_to_vaild_rgbFrame_3.png')

# Convert the image to a PyTorch tensor
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # Add batch dimension
model = yolov11().to(device)

model.train()
img_tensor = img_tensor.to(device)
# Forward pass
with torch.no_grad():
    mc, p = model(img_tensor)

测试尾 = 1
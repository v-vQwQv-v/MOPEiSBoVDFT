from CDENet import MLP
from CDENet import FNOK
import torch
import torch.nn.functional as F
import os

def test_MLP():
    model = MLP()
    model_path = "./model/CDENet1000.pt"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    target_tensor = torch.ones(10, 5, 4)
    n = 0
    while n < 10:
        input_tensor = torch.randn(10, 5, 4)
        output_tensor = model(input_tensor[0])
        loss = F.mse_loss(output_tensor, target_tensor)
        print(output_tensor[0][0][0])
        print (f"loss:{loss}")
        n += 1

def test_FNOK():
    model = FNOK()
    model_path = "./model/CDENet999.pt"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    target_tensor = torch.ones(4, 100, 3)
    n = 0
    while n < 10:
        input_tensor = torch.randn(4, 6)
        output_tensor = model(input_tensor)
        loss = F.mse_loss(output_tensor, target_tensor)
        print(output_tensor[0][0][0])
        print (f"loss:{loss}")
        n += 1

# 测试函数
if __name__ == "__main__":
    pass

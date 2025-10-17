# 验证Python版本
import sys
print(f"Python版本: {sys.version}")  # 需显示3.10.x

# 验证PyTorch与CUDA
import torch
print(f"PyTorch版本: {torch.__version__}")  # 需显示2.0.1
print(f"CUDA是否可用: {torch.cuda.is_available()}")  # 需显示True（若有GPU）

# 验证DINOv3加载
from transformers import AutoModel
model = AutoModel.from_pretrained("./dinov3-vits16-pretrain-lvd1689m", device_map="auto")
print(f"DINOv3加载成功，设备: {model.device}")  # 需显示cuda:0或cpu
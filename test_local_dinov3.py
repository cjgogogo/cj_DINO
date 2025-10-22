import torch
from models.dinov3_feat import DINOv3FeatureExtractor

# 初始化模型（本地路径）
model = DINOv3FeatureExtractor(pretrained_path="./models/dinov3_weights")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 生成测试图像（1张512×512的随机图像）
test_img = torch.randn(1, 3, 512, 512).to(device)  # [B, 3, H, W]

# 提取特征
model.eval()
with torch.no_grad():
    feats = model(test_img)  # 输出4个尺度的特征

# 验证特征形状（关键：确保与后续模块兼容）
print("DINOv3特征形状验证：")
for i, feat in enumerate(feats):
    print(f"第{i+1}个尺度特征：{feat.shape}")
    # 预期形状：[1, 1025, 384]（1025=32×32+1，384是ViT-S的隐藏维度）
    assert feat.shape == (1, 1029, 384), f"特征形状错误，实际为{feat.shape}"

print("本地DINOv3加载与特征提取验证通过！")
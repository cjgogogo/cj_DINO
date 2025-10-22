import torch
from models.dinov3_feat import DINOv3FeatureExtractor

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DINOv3FeatureExtractor().to(device)

# 生成测试输入（1张512×512图像）
test_img = torch.randn(1, 3, 512, 512).to(device)  # [B, 3, H, W]

# 提取特征
feats = model(test_img)

# 验证输出（4个尺度，形状正确）
assert len(feats) == 4, "应输出4个尺度的特征"
for i, feat in enumerate(feats):
    # 特征形状应为 [B, num_tokens, hidden_dim]，num_tokens=32×32+1=1025
    assert feat.shape == (1, 1025, 384), f"第{i}个特征尺度形状错误，实际为{feat.shape}"
print("DINOv3特征提取模块验证通过！")
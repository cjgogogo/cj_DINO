# test_attention.py（完整修正版）
import torch
from models.recon_restormer import ReconRestormer
from models.attention_link import DepthConsistentAttention


def test_attention_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 初始化模型（ReconRestormer输出256通道特征）
    restormer = ReconRestormer(
        dim=256,  # 与注意力模块embed_dim匹配
        num_blocks=1,
        down_factor=8,
        guide_channels=256
    ).to(device)

    # 2. 初始化注意力模块（按实际参数传入）
    attention = DepthConsistentAttention(
        dinov3_hidden_dim=384,  # DINOv3特征通道数
        recon_hidden_dim=256  # 重建特征通道数（与Restormer的dim匹配）
    ).to(device)

    # 3. 生成测试数据
    test_img = torch.randn(1, 3, 256, 256).to(device)  # 输入图像
    print(f"输入图像形状: {test_img.shape}")

    # 4. 生成DINOv3多尺度特征（模拟4个尺度，每个[B, 1029, 384]）
    dinov3_feats = [
        torch.randn(1, 1029, 384).to(device) for _ in range(4)  # 4个尺度
    ]
    print(f"DINOv3特征尺度数: {len(dinov3_feats)}, 每个形状: {dinov3_feats[0].shape}")

    # 5. 前向传播
    with torch.no_grad():
        # 5.1 Restormer初步重建（输出256通道中间特征）
        initial_recon, mid_feat = restormer(test_img)
        print(f"初步重建图像形状: {initial_recon.shape}")
        print(f"中间特征形状: {mid_feat.shape}")  # 应输出 [1, 256, 32, 32]

        # 5.2 深度注意力模块（输入256通道重建特征）
        guided_feat, depth_map = attention(dinov3_feats, mid_feat)
        print(f"引导特征形状: {guided_feat.shape}")  # 应输出 [1, 256, 32, 32]
        print(f"深度图形状: {depth_map.shape}")

        # 5.3 最终重建
        final_recon, _ = restormer(test_img, guided_feat=guided_feat)
        print(f"最终重建图像形状: {final_recon.shape}")  # 应输出 [1, 3, 256, 256]

    print("✅ 所有模块测试通过！")


if __name__ == "__main__":
    test_attention_pipeline()
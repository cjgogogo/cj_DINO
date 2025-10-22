import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dinov3_feat import DINOv3FeatureExtractor
from models.attention_link import DepthConsistentAttention   # 保持别名
from models.recon_restormer import ReconRestormer as Restormer  # 保持别名
from models.gram_anchoring import GramAnchoring  # 假设已实现
from utils import DepthEstimator


class DefocusReconModel(nn.Module):
    """端到端失焦模糊重建模型，适配256通道Restormer"""

    def __init__(self):
        super(DefocusReconModel, self).__init__()
        # 1. DINOv3特征提取器（深层特征通道384）
        self.dinov3_feat = DINOv3FeatureExtractor()

        # 2. 深度估计器（用于创新点1的深度掩码）
        self.depth_estimator = DepthEstimator(device="cuda" if torch.cuda.is_available() else "cpu")

        # 3. 深度一致性注意力（创新点2，输入通道384→256，与Restormer匹配）
        self.attention = DepthConsistentAttention(
            dinov3_hidden_dim=384,  # DINOv3深层特征通道
            recon_hidden_dim=256  # 与Restormer的dim=256一致
        )

        # 4. Gram矩阵锚点库（创新点3）
        self.gram_anchor = GramAnchoring()

        # 5. 重建网络（使用256通道的ReconRestormer）
        self.restormer = Restormer(
            in_channels=3,
            out_channels=3,
            dim=256,  # 与修正后的Restormer一致
            num_blocks=1,
            down_factor=8,
            guide_channels=256  # 引导特征通道数（注意力输出为256）
        )

        # 6. 特征融合卷积（DINOv3注意力特征256 + Restormer中层特征256 → 256）
        self.feat_fusion = nn.Conv2d(
            in_channels=256 + 256,  # 核心修正：与256通道Restormer匹配
            out_channels=256,
            kernel_size=3,
            padding=1
        )

    def forward(self, blur_img):
        """
        输入：失焦模糊图像 (B, 3, H, W)，建议H=W=512
        输出：重建清晰图像 (B, 3, H, W) + Gram特征
        """
        B, C, H, W = blur_img.shape

        # ---------------------------
        # 步骤1：提取DINOv3多尺度特征
        # ---------------------------
        dinov3_feats = self.dinov3_feat(blur_img)  # [浅层, 中层, 深层]，深层通道384
        shallow_feat, mid_feat, deep_feat = dinov3_feats

        # ---------------------------
        # 步骤2：深度感知特征筛选（创新点1）
        # ---------------------------
        # 生成深度掩码（值越高表示越失焦）
        depth_mask = self.depth_estimator.predict_depth(blur_img)  # (B, 1, H, W)
        # 下采样到DINOv3深层特征尺寸（H/16, W/16）
        depth_mask_down = F.interpolate(
            depth_mask,
            size=deep_feat.shape[2:],
            mode="bilinear",
            align_corners=False
        )
        # 失焦区域特征加权（高深度值区域权重更高）
        weighted_deep_feat = deep_feat * (1 + depth_mask_down)  # (B, 384, H/16, W/16)

        # ---------------------------
        # 步骤3：注意力联动（创新点2）
        # ---------------------------
        # 从浅层特征提取聚焦区域掩码（高边缘强度区域）
        focus_mask = self._get_focus_mask(shallow_feat)  # (B, 1, H/4, W/4)

        # 获取Restormer初始特征（用于注意力输入）
        _, restormer_init_feat = self.restormer(blur_img)  # (B, 256, H/8, W/8)

        # 注意力计算：融合DINOv3特征与Restormer特征
        attended_feat, _ = self.attention(
            dinov3_feats=dinov3_feats,  # DINOv3多尺度特征
            recon_feat=restormer_init_feat  # Restormer初始特征（256通道）
        )  # 输出: (B, 256, H/8, W/8)（与Restormer下采样后尺寸匹配）

        # ---------------------------
        # 步骤4：特征融合与最终重建
        # ---------------------------
        # Restormer中层特征（256通道）
        restormer_mid_feat = restormer_init_feat  # (B, 256, H/8, W/8)

        # 融合注意力特征与Restormer特征
        fused_feat = self.feat_fusion(torch.cat([attended_feat, restormer_mid_feat], dim=1))  # (B, 256, H/8, W/8)

        # 最终重建（传入融合特征作为引导）
        pred_sharp, _ = self.restormer(blur_img, guided_feat=fused_feat)  # (B, 3, H, W)

        # ---------------------------
        # 步骤5：返回重建结果与Gram特征
        # ---------------------------
        return pred_sharp, self.gram_anchor.extract_feat(pred_sharp)

    def _get_focus_mask(self, shallow_feat):
        """从DINOv3浅层特征提取聚焦区域掩码（高边缘强度区域）"""
        B = shallow_feat.shape[0]
        # 计算特征梯度（边缘强度）
        grad_x = torch.abs(F.conv2d(
            shallow_feat,
            torch.tensor([[-1, 1]], device=shallow_feat.device).unsqueeze(0).unsqueeze(0),
            padding=1
        ))
        grad_y = torch.abs(F.conv2d(
            shallow_feat,
            torch.tensor([[-1], [1]], device=shallow_feat.device).unsqueeze(0).unsqueeze(0),
            padding=1
        ))
        edge_strength = (grad_x + grad_y).mean(dim=1, keepdim=True)  # (B, 1, H/4, W/4)

        # 筛选前20%高边缘强度区域作为聚焦区域
        threshold = torch.topk(
            edge_strength.view(B, -1),
            k=int(0.2 * edge_strength.numel() // B),  # 取前20%像素
            dim=1
        )[0][:, -1].view(B, 1, 1, 1)  # 阈值
        focus_mask = (edge_strength >= threshold).float()  # 二值掩码
        return focus_mask

# 测试模型前向传播dd
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefocusReconModel().to(device)

    # 测试输入：2张512x512的模糊图
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    pred, gram_feat = model(dummy_input)

    # 验证输出形状
    print(f"重建图像形状: {pred.shape}（预期 [2, 3, 512, 512]）")
    print(f"Gram特征形状: {gram_feat.shape}（根据GramAnchoring定义）")
    print("前向传播测试通过！")
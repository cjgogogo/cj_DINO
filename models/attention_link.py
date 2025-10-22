import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention

#ddd
class DepthConsistentAttention(torch.nn.Module):
    """深度一致性注意力联动模块"""

    def __init__(self, dinov3_hidden_dim=384, recon_hidden_dim=256):
        super().__init__()
        self.dinov3_hidden_dim = dinov3_hidden_dim  # DINOv3特征通道数
        self.recon_hidden_dim = recon_hidden_dim    # 重建特征通道数

        # 1. DINOv3特征通道对齐（384→256）
        self.feat_proj = torch.nn.ModuleList([
            torch.nn.Conv1d(dinov3_hidden_dim, recon_hidden_dim, kernel_size=1)
            for _ in range(4)  # 适配4个尺度
        ])

        # 2. 深度伪标签估计器（输入为DINOv3左右特征拼接：384×2=768通道）
        self.depth_estimator = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dinov3_hidden_dim * 2,  # 768通道（修正点）
                out_channels=recon_hidden_dim,
                kernel_size=3,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(recon_hidden_dim, 1, kernel_size=1)
        )

        # 3. 多头部注意力
        self.self_attn = MultiheadAttention(
            embed_dim=recon_hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def estimate_depth(self, dinov3_deep_feat):
        """从DINOv3深层特征估计深度伪标签"""
        # 去掉CLS token，保留图像patch特征 [B, 1028, 384]
        feat_patch = dinov3_deep_feat[:, 1:, :]
        B, N_patch, C = feat_patch.shape  # C=384

        # 重塑为2D特征图（H_feat×W_feat=1028）
        H_feat = 4
        W_feat = 257  # 4×257=1028，匹配N_patch
        feat_2d = feat_patch.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)  # [B, 384, 4, 257]

        # 拼接左右相邻patch特征（384+384=768通道）
        feat_left = feat_2d[:, :, :, :-1]  # [B, 384, 4, 256]
        feat_right = feat_2d[:, :, :, 1:]  # [B, 384, 4, 256]
        feat_concat = torch.cat([feat_left, feat_right], dim=1)  # [B, 768, 4, 256]

        # 估计深度并上采样
        depth = self.depth_estimator(feat_concat)  # [B, 1, 4, 256]
        depth = F.interpolate(depth, size=(64, 64), mode='bilinear', align_corners=True)
        return depth

    def forward(self, dinov3_feats, recon_feat):
        """前向传播：融合DINOv3特征和重建特征"""
        # 1. 估计深度伪标签
        depth = self.estimate_depth(dinov3_feats[-1])  # [B, 1, 64, 64]
        depth = F.interpolate(depth, size=recon_feat.shape[2:], mode='bilinear', align_corners=True)

        # 2. DINOv3特征通道对齐（384→256）
        proj_feats = []
        for i, feat in enumerate(dinov3_feats):
            feat_patch = feat[:, 1:, :]  # [B, 1028, 384]
            proj_feat = self.feat_proj[i](feat_patch.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 1028, 256]
            max_len = max([f.shape[1] - 1 for f in dinov3_feats])
            if proj_feat.shape[1] < max_len:
                proj_feat = F.pad(proj_feat, (0, 0, 0, max_len - proj_feat.shape[1]))
            proj_feats.append(proj_feat)
        fused_dinov3_feat = torch.mean(torch.stack(proj_feats), dim=0)  # [B, 1028, 256]

        # 3. 重建特征处理（32×32→1024序列，补零至1028）
        B, C_recon, H_recon, W_recon = recon_feat.shape  # [1, 256, 32, 32]
        recon_feat_seq = recon_feat.permute(0, 2, 3, 1).reshape(B, H_recon * W_recon, C_recon)  # [1, 1024, 256]
        recon_feat_seq = F.pad(recon_feat_seq, (0, 0, 0, 1028 - 1024))  # [1, 1028, 256]

        # 4. 深度一致性注意力计算
        attn_output, attn_weights = self.self_attn(
            query=recon_feat_seq,
            key=fused_dinov3_feat,
            value=fused_dinov3_feat,
            need_weights=True
        )

        # 5. 深度权重调整
        depth_seq = depth.reshape(B, 1, H_recon * W_recon)  # [1, 1, 1024]
        depth_seq = F.pad(depth_seq, (0, 4))  # 补零至1028
        depth_norm = (depth_seq - depth_seq.min()) / (depth_seq.max() - depth_seq.min() + 1e-6)
        depth_norm = depth_norm.permute(0, 2, 1)  # [1,1,1028] → [1,1028,1]
        attn_output = attn_output * (1 + depth_norm)  # 现在可广播相乘

        # 6. 重塑回2D引导特征（去除补零，恢复32×32）
        guided_feat = attn_output[:, :1024, :].reshape(B, H_recon, W_recon, C_recon).permute(0, 3, 1, 2)

        return guided_feat, depth
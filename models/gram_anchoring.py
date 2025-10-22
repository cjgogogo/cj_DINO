import torch
import torch.nn as nn
import torch.nn.functional as F


class GramAnchoring(nn.Module):
    """
    Gram Anchoring模块：
    1. 计算DINOv3特征的Gram矩阵（描述特征通道间的关联分布）
    2. 训练初期建立“锚点Gram矩阵”（锚定模糊图像的弱细节特征分布）
    3. 训练中通过“当前Gram矩阵与锚点的差异”约束模型，避免细节丢失
    """
    def __init__(self, patch_size=2):
        """
        Args:
            patch_size: 计算Gram矩阵时的局部块大小（2×2，平衡局部细节与计算量）
        """
        super().__init__()
        self.patch_size = patch_size  # 局部块尺寸，用于捕捉局部特征关联

    def compute_gram_matrix(self, feat):
        """
        计算单尺度特征的Gram矩阵（核心函数）
        Args:
            feat: DINOv3输出的单尺度特征，形状 [B, num_tokens, hidden_dim]（你的特征是[B, 1029, 384]）
        Returns:
            gram: 局部块Gram矩阵，形状 [B, num_blocks, hidden_dim, hidden_dim]
                  （num_blocks：特征图上的局部块数量）
        """
        # 步骤1：去除CLS token，仅保留图像patch的特征（[B, 1028, 384]）
        feat_patch = feat[:, 1:, :]  # 跳过第1个CLS token，聚焦图像内容
        B, N_patch, C = feat_patch.shape  # N_patch=1028（你的特征实际图像patch数）

        # 步骤2：将1D patch特征reshape为2D特征图（适配局部块划分）
        # 因1028≈32×32（32×32=1024，补4个patch对齐）
        H_feat = 32  # 特征图高度（与attention_link.py保持一致）
        W_feat = 32  # 特征图宽度
        # 补零对齐到H_feat×W_feat（确保能均匀划分局部块）
        if N_patch < H_feat * W_feat:
            pad_num = H_feat * W_feat - N_patch
            feat_patch = F.pad(feat_patch, (0, 0, 0, pad_num))  # [B, 1024, 384]
        # 重塑为2D特征图：[B, C, H_feat, W_feat]
        feat_2d = feat_patch.permute(0, 2, 1).reshape(B, C, H_feat, W_feat)

        # 步骤3：划分局部块（用unfold操作提取多个2×2的局部区域）
        # 输出形状：[B, C, num_block_h, num_block_w, patch_size, patch_size]
        feat_blocks = feat_2d.unfold(
            dimension=2, size=self.patch_size, step=self.patch_size  # 高度方向划分
        ).unfold(
            dimension=3, size=self.patch_size, step=self.patch_size  # 宽度方向划分
        )
        # 整理维度：[B, num_blocks, C, patch_size×patch_size]
        B, C, num_block_h, num_block_w, ph, pw = feat_blocks.shape
        num_blocks = num_block_h * num_block_w  # 总块数：16×16=256（32/2=16）
        feat_blocks = feat_blocks.permute(0, 2, 3, 1, 4, 5).reshape(
            B, num_blocks, C, ph * pw
        )

        # 步骤4：计算每个局部块的Gram矩阵（描述通道间的关联）
        # Gram矩阵公式：G = (X × X^T) / (C×ph×pw)，X为[C, ph×pw]的块特征
        gram_blocks = []
        for b in range(B):
            for i in range(num_blocks):
                block = feat_blocks[b, i]  # [C, ph×pw]
                gram = torch.matmul(block, block.t())  # [C, C]
                gram = gram / (C * ph * pw)  # 归一化，避免尺度差异
                gram_blocks.append(gram)
        # 整合为[B, num_blocks, C, C]
        gram = torch.stack(gram_blocks).reshape(B, num_blocks, C, C)

        return gram

    def forward(self, dinov3_feats, anchor_gram=None, is_train=True):
        """
        前向传播：计算当前特征与锚点的Gram损失，返回增强特征
        Args:
            dinov3_feats: DINOv3输出的4个尺度特征，每个[B, 1029, 384]
            anchor_gram: 训练初期的锚点Gram矩阵（4个尺度，与dinov3_feats对应）
            is_train: 是否训练模式（训练时计算损失，推理时仅返回特征）
        Returns:
            enhanced_feats: 增强后的特征（与输入同形状，用于后续注意力模块）
            gram_loss: Gram约束损失（仅训练时返回，用于约束特征分布）
        """
        # 步骤1：计算当前特征的Gram矩阵（4个尺度分别计算）
        current_gram_list = [self.compute_gram_matrix(feat) for feat in dinov3_feats]

        if not is_train:
            # 推理模式：直接返回原始特征（无需增强，因损失仅用于训练）
            return dinov3_feats, torch.tensor(0.0)

        # 训练模式：计算Gram损失（当前Gram与锚点Gram的差异）
        gram_loss = 0.0
        for current_gram, anchor in zip(current_gram_list, anchor_gram):
            # 每个尺度的Gram损失：L2距离（衡量分布差异）
            gram_loss += F.mse_loss(current_gram, anchor.detach())  # 锚点不参与梯度更新

        # 步骤2：特征增强（可选，用Gram损失的梯度引导特征调整）
        # 简单实现：将Gram损失的梯度反向传播到特征，增强与锚点一致的成分
        enhanced_feats = [
            feat + 0.01 * torch.autograd.grad(gram_loss, feat, retain_graph=True)[0]
            for feat in dinov3_feats
        ]

        return enhanced_feats, gram_loss
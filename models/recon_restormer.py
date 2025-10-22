import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint  # 引入梯度检查点


class ConvLayer(nn.Module):
    """基础卷积层（带BN和激活，轻量化设计）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)  # 节省显存

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiHeadSelfAttention(nn.Module):
    """轻量化多头自注意力（适配小显存）"""

    def __init__(self, dim, num_heads=2):  # 头数从8→2，降低计算量
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)  # 1x1卷积生成QKV（高效）
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)  # 输出投影

    def forward(self, x):
        """x: [B, dim, H, W]"""
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)  # 拆分Q/K/V

        # 重塑为注意力格式：[B, num_heads, H*W, head_dim]
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) hh ww -> b h (hh ww) d',
                                h=self.num_heads, d=self.head_dim),
            qkv
        )

        # 注意力计算（限制中间张量尺寸）
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, num_heads, H*W, head_dim]

        # 重塑回特征图格式
        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)
        return self.proj(out)


class RestormerBlock(nn.Module):
    """Restormer基础块（带梯度检查点，减少显存占用）"""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MultiHeadSelfAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(  # 轻量化前馈网络
            ConvLayer(dim, dim * 2, kernel_size=1, padding=0),  # 通道扩展从4→2倍
            ConvLayer(dim * 2, dim, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # 用梯度检查点包装计算，不存储中间激活值
        x = x + checkpoint(self.attn, self.norm1(x))  # 残差+注意力
        x = x + checkpoint(self.ffn, self.norm2(x))  # 残差+前馈网络
        return x


class ReconRestormer(nn.Module):
    """
    优化后的失焦模糊重建模块：
    1. 增加下采样层，降低特征图尺寸（核心显存优化）
    2. 轻量化参数，适配中小显存GPU
    3. 支持引导特征融合，保留原功能逻辑
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=16,  # 特征维度从64→16（大幅降显存）
                 num_blocks=1,  # 网络深度从4→1（减少计算）
                 down_factor=8):  # 下采样倍数（512→64，8倍）
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.down_factor = down_factor  # 下采样倍数（需与输入尺寸匹配）

        # 输入投影+下采样（核心优化：降低特征图尺寸）
        self.in_proj = nn.Sequential(
            ConvLayer(in_channels, dim),
            nn.Conv2d(dim, dim, kernel_size=down_factor, stride=down_factor)  # 下采样
        )

        # Restormer主干（轻量化）
        self.blocks = nn.ModuleList([
            RestormerBlock(dim) for _ in range(num_blocks)
        ])

        # 引导特征融合层（适配下采样后的尺寸）
        self.guide_fuse = ConvLayer(dim + 256, dim, kernel_size=1, padding=0)

        # 输出投影+上采样（恢复原始尺寸）
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, out_channels * (down_factor ** 2), kernel_size=1),
            nn.PixelShuffle(down_factor)  # 像素洗牌上采样（高效）
        )

    def forward(self, x, guided_feat=None):
        """
        Args:
            x: 输入模糊图像，形状 [B, 3, H, W]（建议H=W=256）
            guided_feat: 引导特征，形状 [B, 256, H/down_factor, W/down_factor]
        Returns:
            recon_img: 重建图像，形状 [B, 3, H, W]
            mid_feat: 中间特征，形状 [B, dim, H/down_factor, W/down_factor]
        """
        # 输入下采样：[B, 3, H, W] → [B, dim, H/down, W/down]
        x = self.in_proj(x)

        # 主干特征提取（轻量化计算）
        for block in self.blocks:
            x = block(x)

        # 保存中间特征（用于注意力模块）
        mid_feat = x.clone()

        # 融合引导特征（若有）
        if guided_feat is not None:
            # 确保引导特征与当前特征尺寸匹配（下采样后）
            guided_feat = F.interpolate(
                guided_feat,
                size=x.shape[2:],  # 匹配x的H/W
                mode='bilinear',
                align_corners=False
            )
            x = torch.cat([x, guided_feat], dim=1)  # 通道拼接
            x = self.guide_fuse(x)  # 融合降维

        # 输出上采样：[B, dim, H/down, W/down] → [B, 3, H, W]
        recon_img = self.out_proj(x)

        return recon_img, mid_feat
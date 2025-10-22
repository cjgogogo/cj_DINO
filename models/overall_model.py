# recon_restormer.py（完整修正版）
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) hh ww -> b h (hh ww) d',
                                h=self.num_heads, d=self.head_dim),
            qkv
        )

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = rearrange(out, 'b h (hh ww) d -> b (h d) hh ww', hh=H, ww=W)
        return self.proj(out)


class RestormerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MultiHeadSelfAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            ConvLayer(dim, dim * 2, kernel_size=1, padding=0),
            ConvLayer(dim * 2, dim, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = x + checkpoint(self.attn, self.norm1(x), use_reentrant=False)
        x = x + checkpoint(self.ffn, self.norm2(x), use_reentrant=False)
        return x


class ReconRestormer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 dim=256,  # 核心修正：通道数改为256，匹配注意力模块
                 num_blocks=1,
                 down_factor=8,
                 guide_channels=256):
        super().__init__()
        self.dim = dim
        self.down_factor = down_factor
        self.guide_channels = guide_channels

        # 输入投影+下采样
        self.in_proj = nn.Sequential(
            ConvLayer(in_channels, dim),  # 直接投影到256通道
            nn.Conv2d(dim, dim, kernel_size=down_factor, stride=down_factor)
        )

        # Restormer主干
        self.blocks = nn.ModuleList([
            RestormerBlock(dim) for _ in range(num_blocks)
        ])

        # 引导特征融合层（256 + 256 = 512输入通道）
        self.guide_fuse = ConvLayer(
            dim + guide_channels,
            dim,
            kernel_size=1,
            padding=0
        )

        # 输出投影+上采样
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, out_channels * (down_factor ** 2), kernel_size=1),
            nn.PixelShuffle(down_factor)
        )

    def forward(self, x, guided_feat=None):
        x = self.in_proj(x)  # [B, 256, H/down, W/down]

        for block in self.blocks:
            x = block(x)

        mid_feat = x.clone()  # 此时mid_feat通道数为256

        if guided_feat is not None:
            guided_feat = F.interpolate(
                guided_feat,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            x = torch.cat([x, guided_feat], dim=1)
            x = self.guide_fuse(x)

        recon_img = self.out_proj(x)
        return recon_img, mid_feat
import os
import torch
import numpy as np
import cv2
from torch import nn
from PIL import Image
from torchvision.transforms import functional as F


# ------------------------------
# 1. 图像预处理工具（加载、 resize、归一化）
# ------------------------------
def load_image(path, img_size=None):
    """加载图像并转为Tensor（适配失焦模糊场景的预处理）"""
    # 用PIL加载图像（支持多种格式）
    img = Image.open(path).convert("RGB")  # 转为RGB通道（避免灰度图或RGBA格式问题）

    # 若指定尺寸，Resize图像（保持比例，短边对齐img_size）
    if img_size is not None:
        w, h = img.size
        if w < h:
            new_w = img_size
            new_h = int(h * (img_size / w))
        else:
            new_h = img_size
            new_w = int(w * (img_size / h))
        img = img.resize((new_w, new_h), Image.BILINEAR)  # 双线性插值（适合平滑失焦区域）

    # 转为Tensor并归一化（像素值从[0,255]→[0,1]）
    img_tensor = F.to_tensor(img)  # 形状：(3, H, W)，值范围[0,1]
    return img_tensor


def normalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """标准化Tensor（使用ImageNet均值和方差，适配预训练模型DINOv3）"""
    return F.normalize(tensor, mean=mean, std=std)


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化（用于可视化，将Tensor转回[0,1]范围）"""
    tensor = tensor.clone()  # 避免修改原张量
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 还原公式：x = x*std + mean
    return torch.clamp(tensor, 0, 1)  # 确保值在[0,1]内（防止溢出）


# ------------------------------
# 2. 评估指标（失焦重建重点关注SSIM和边缘一致性）
# ------------------------------
class PSNR(nn.Module):
    """峰值信噪比（Peak Signal-to-Noise Ratio）：评估重建清晰度"""

    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range  # 像素值范围（失焦重建中通常为[0,1]）

    def forward(self, pred, target):
        # 计算MSE（均方误差）
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"))  # MSE为0时PSNR无穷大（完美重建）
        # PSNR公式：10 * log10( (data_range^2) / MSE )
        return 10 * torch.log10((self.data_range ** 2) / mse)


class SSIM(nn.Module):
    """结构相似性指数（Structural Similarity Index）：评估结构和纹理一致性（失焦重建更重要）"""

    def __init__(self, window_size=11, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.channel = 3  # RGB图像
        # 高斯窗口（模拟人眼对局部区域的敏感度）
        self.window = self._create_gaussian_window(window_size, self.channel)

    def _create_gaussian_window(self, window_size, channel):
        """生成高斯窗口（用于局部区域加权）"""
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * 1.0 ** 2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)  # 2D高斯核
        window = window.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)  # 扩展到3通道
        return window.to(torch.float32)

    def forward(self, pred, target):
        # 确保输入尺寸一致
        assert pred.shape == target.shape, "pred and target must have the same shape"

        # 计算均值（局部区域亮度）
        mu_x = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=self.channel)
        mu_y = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channel)

        # 计算方差（局部区域对比度）
        sigma_x = F.conv2d(pred ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu_x ** 2
        sigma_y = F.conv2d(target ** 2, self.window, padding=self.window_size // 2, groups=self.channel) - mu_y ** 2
        sigma_xy = F.conv2d(pred * target, self.window, padding=self.window_size // 2,
                            groups=self.channel) - mu_x * mu_y

        # SSIM公式：(2μxμy + C1) * (2σxy + C2) / [(μx² + μy² + C1) * (σx² + σy² + C2)]
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                    (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))

        return ssim_map.mean()


class EdgeConsistencyLoss(nn.Module):
    """边缘一致性损失：确保失焦重建后边缘清晰（失焦模糊的核心退化是边缘模糊）"""

    def __init__(self, edge_threshold=0.1):
        super().__init__()
        self.edge_threshold = edge_threshold
        # Sobel算子（用于提取边缘）
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        # 初始化Sobel核（x方向和y方向）
        self.sobel_x.weight.data = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.sobel_y.weight.data = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).repeat(3, 1, 1, 1)
        self.sobel_x.weight.requires_grad = False  # 固定Sobel核，不参与训练
        self.sobel_y.weight.requires_grad = False

    def forward(self, pred, target):
        # 提取预测图和目标图的边缘
        pred_edge_x = self.sobel_x(pred)
        pred_edge_y = self.sobel_y(pred)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2)  # 边缘强度

        target_edge_x = self.sobel_x(target)
        target_edge_y = self.sobel_y(target)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2)

        # 只关注强边缘区域（过滤噪声）
        mask = (target_edge > self.edge_threshold).float()  # 目标图中边缘强度高的区域
        loss = torch.mean(mask * (pred_edge - target_edge) ** 2)  # 仅对强边缘计算损失
        return loss


# ------------------------------
# 3. 失焦重建特有的损失函数（深度一致性、Gram材质约束）
# ------------------------------
class DepthConsistencyLoss(nn.Module):
    """深度一致性损失：确保重建的模糊程度与深度匹配（失焦模糊与深度强相关）"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # 控制惩罚强度

    def forward(self, pred_depth_mask, gt_depth):
        """
        pred_depth_mask: 模型预测的失焦程度掩码（与深度正相关，值越大越失焦）
        gt_depth: 真实深度图（归一化到[0,1]，值越大距离越远）
        """
        # 失焦程度应与深度正相关（假设对焦平面在中间，过近或过远都失焦，可根据数据集调整）
        # 这里简化为：pred_depth_mask 应与 gt_depth 趋势一致（用MSE约束）
        return torch.mean((pred_depth_mask - gt_depth) ** 2)


class GramConsistencyLoss(nn.Module):
    """Gram矩阵一致性损失：确保重建区域的材质纹理与锚点库一致（避免伪细节）"""

    def __init__(self, anchor_lib, reduction="mean"):
        super().__init__()
        self.anchor_lib = anchor_lib  # 材质锚点库（如{"skin": gram_matrix, "cloth": gram_matrix, ...}）
        self.reduction = reduction

    def _gram_matrix(self, x):
        """计算特征图的Gram矩阵（表示纹理风格）"""
        # x形状：(B, C, H, W)
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)  # 展平空间维度
        gram = torch.bmm(features, features.transpose(1, 2))  # 批次内矩阵乘法：(B, C, C)
        return gram / (c * h * w)  # 归一化

    def forward(self, pred_img, blur_img, dino_features):
        """
        pred_img: 重建图像（B, 3, H, W）
        blur_img: 失焦模糊图像（B, 3, H, W）
        dino_features: DINOv3提取的语义特征（用于识别材质类别）
        """
        total_loss = 0.0
        # 假设DINOv3特征已包含材质类别信息（如通过分类头预测材质）
        material_labels = torch.argmax(dino_features[:, :len(self.anchor_lib)], dim=1)  # 简化：取前N类为材质

        for i in range(pred_img.shape[0]):
            # 获取当前图像的材质锚点
            material = list(self.anchor_lib.keys())[material_labels[i]]
            anchor_gram = self.anchor_lib[material].to(pred_img.device)  # 锚点库中的Gram矩阵

            # 计算重建图像的Gram矩阵
            pred_gram = self._gram_matrix(pred_img[i:i + 1])  # 取单张图像

            # 计算与锚点的差异（L1损失，更鲁棒）
            loss = torch.mean(torch.abs(pred_gram - anchor_gram))
            total_loss += loss

        if self.reduction == "mean":
            return total_loss / pred_img.shape[0]
        return total_loss


# ------------------------------
# 4. 工具函数（文件夹创建、模型保存等）
# ------------------------------
def create_dirs(path_list):
    """创建多个文件夹（若不存在）"""
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


def save_images(pred, target, blur, save_path, epoch):
    """保存重建结果图像（用于可视化对比）"""
    # 反标准化并转为numpy（形状：(H, W, 3)）
    pred_np = denormalize_tensor(pred).squeeze().permute(1, 2, 0).cpu().numpy() * 255
    target_np = denormalize_tensor(target).squeeze().permute(1, 2, 0).cpu().numpy() * 255
    blur_np = denormalize_tensor(blur).squeeze().permute(1, 2, 0).cpu().numpy() * 255

    # 转为BGR格式（OpenCV默认）
    pred_np = cv2.cvtColor(pred_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    target_np = cv2.cvtColor(target_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    blur_np = cv2.cvtColor(blur_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 拼接成一行（模糊图 + 重建图 + 清晰图）
    combined = np.hstack([blur_np, pred_np, target_np])
    cv2.imwrite(os.path.join(save_path, f"epoch_{epoch}.png"), combined)
    print(f"Saved visualization to {os.path.join(save_path, f'epoch_{epoch}.png')}")
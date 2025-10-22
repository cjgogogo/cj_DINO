import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F


class DefocusDataset(Dataset):
    """失焦模糊重建数据集，加载模糊图-清晰图配对数据"""

    def __init__(self, txt_path, img_size=(512, 512), is_train=True):
        """
        Args:
            txt_path: 数据路径文件（每行格式：模糊图路径 清晰图路径）
            img_size: 图像Resize尺寸 (H, W)
            is_train: 是否为训练集（训练集启用数据增强）
        """
        # 读取配对路径（过滤空行和无效路径）
        self.pairs = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                blur_path, sharp_path = line.split()
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    self.pairs.append((blur_path, sharp_path))
                else:
                    print(f"警告：无效路径对 {blur_path} {sharp_path}，已跳过")

        self.img_size = img_size
        self.is_train = is_train

        # 基础预处理（训练/验证共享）
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # 转为[0,1]范围的Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        # 训练集数据增强（仅对模糊图增强，模拟真实失焦场景）
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),  # 随机调整锐度
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))],
                p=0.2  # 轻微高斯模糊，增强模型鲁棒性
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
                p=0.3  # 随机亮度/对比度调整
            )
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        # 读取图像（确保RGB格式）
        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        # 训练集数据增强（仅对模糊图应用）
        if self.is_train:
            blur_img = self.augment_transform(blur_img)

        # 基础预处理（统一尺寸、标准化）
        blur_tensor = self.base_transform(blur_img)
        sharp_tensor = self.base_transform(sharp_img)

        # 计算模糊度分数（拉普拉斯方差）
        blur_score = self.calculate_blur_score(np.array(blur_img))

        return {
            "blur": blur_tensor,  # 失焦模糊图 (3, H, W)
            "sharp": sharp_tensor,  # 清晰图（标签）(3, H, W)
            "blur_path": blur_path,  # 图像路径（用于调试）
            "sharp_path": sharp_path,
            "blur_score": blur_score  # 模糊度分数（0-1，越高越模糊）
        }

    def calculate_blur_score(self, img_np):
        """计算拉普拉斯方差作为模糊度指标（值越低越模糊）"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 归一化到0-1（根据数据集统计调整阈值，这里假设最大方差为1000）
        return np.clip(1 - (laplacian / 1000), 0, 1)


class DepthEstimator:
    """单目深度估计工具（用于创新点1的深度掩码生成）"""

    def __init__(self, model_type="DPT_Large", device="cuda"):
        """加载MiDaS深度估计模型（需安装：pip install midas torchvision）"""
        self.device = device
        # 加载MiDaS模型（自动下载权重，若本地有可指定路径）
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
        self.model.eval()

        # MiDaS预处理（适配模型输入要求）
        self.transform = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict_depth(self, img_tensor):
        """
        输入：清晰图Tensor (3, H, W) 或批量Tensor (B, 3, H, W)
        输出：深度图Tensor (B, 1, H, W)，值越高表示距离越远
        """
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)  # 单张图转批量格式

        # 预处理并预测深度
        input_tensor = self.transform(img_tensor).to(self.device)
        with torch.no_grad():
            depth = self.model(input_tensor)

        # 调整深度图尺寸与输入一致
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),  # 增加通道维度
            size=img_tensor.shape[-2:],
            mode="bicubic",
            align_corners=False
        ).squeeze(1)  # 移除中间通道维度

        # 归一化到0-1（便于作为掩码权重）
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.unsqueeze(1)  # 输出形状：(B, 1, H, W)


def visualize_batch(batch, save_path=None):
    """可视化一个batch的数据（模糊图+清晰图+模糊度分数）"""
    batch_size = batch["blur"].shape[0]
    plt.figure(figsize=(15, 5 * batch_size))

    for i in range(batch_size):
        # 反标准化（用于可视化）
        blur_img = denormalize(batch["blur"][i]).permute(1, 2, 0).cpu().numpy()
        sharp_img = denormalize(batch["sharp"][i]).permute(1, 2, 0).cpu().numpy()

        # 绘制模糊图
        plt.subplot(batch_size, 2, 2 * i + 1)
        plt.imshow(blur_img)
        plt.title(f"Blur (score: {batch['blur_score'][i]:.2f})")
        plt.axis("off")

        # 绘制清晰图
        plt.subplot(batch_size, 2, 2 * i + 2)
        plt.imshow(sharp_img)
        plt.title("Sharp (Ground Truth)")
        plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"可视化结果已保存到 {save_path}")
    plt.close()


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化Tensor，用于可视化（将[-1,1]范围转回[0,1]）"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def save_reconstructed_results(recon_img, sharp_img, blur_img, save_dir, idx):
    """保存重建结果（模糊图+重建图+清晰图对比）"""
    os.makedirs(save_dir, exist_ok=True)

    # 反标准化并转为numpy（0-1范围）
    blur_np = denormalize(blur_img).permute(1, 2, 0).cpu().numpy()
    recon_np = denormalize(recon_img).permute(1, 2, 0).cpu().numpy()
    sharp_np = denormalize(sharp_img).permute(1, 2, 0).cpu().numpy()

    # 拼接对比图并转为0-255
    combined = np.hstack([blur_np, recon_np, sharp_np])
    combined = (combined * 255).astype(np.uint8)

    save_path = os.path.join(save_dir, f"result_{idx}.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))  # 转BGR保存
    return save_path
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class DefocusDataset(Dataset):
    """失焦模糊重建数据集，加载模糊图-清晰图配对数据"""

    def __init__(self, txt_path, img_size=(512, 512), is_train=True):
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

        # 基础预处理（标准化+Resize）
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        # 训练集数据增强（仅对模糊图应用）
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5))], p=0.2),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.pairs[idx]

        # 读取图像（RGB格式）
        blur_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        # 训练集增强（仅模糊图）
        if self.is_train:
            blur_img = self.augment_transform(blur_img)

        # 预处理
        blur_tensor = self.base_transform(blur_img)
        sharp_tensor = self.base_transform(sharp_img)

        # 计算模糊度分数（拉普拉斯方差）
        blur_score = self.calculate_blur_score(np.array(blur_img))

        return {
            "blur": blur_tensor,  # (3, H, W)
            "sharp": sharp_tensor,  # (3, H, W)
            "blur_path": blur_path,
            "sharp_path": sharp_path,
            "blur_score": blur_score  # 0-1，越高越模糊
        }

    def calculate_blur_score(self, img_np):
        """计算拉普拉斯方差，归一化后作为模糊度指标"""
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_laplacian = 500  # 需根据你的数据集统计调整
        return np.clip(1 - (laplacian / max_laplacian), 0, 1)


class DepthEstimator:
    """单目深度估计工具（基于MiDaS，用于创新点1的深度掩码）"""

    def __init__(self, model_type="DPT_Large", device="cuda"):
        self.device = device
        # 加载MiDaS模型（自动下载权重）
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(device)
        self.model.eval()

        # MiDaS预处理
        self.transform = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict_depth(self, img_tensor):
        """
        输入：图像Tensor (B, 3, H, W) 或 (3, H, W)
        输出：归一化深度图 (B, 1, H, W)，值越高表示距离越远（越失焦）
        """
        is_single = img_tensor.ndim == 3
        if is_single:
            img_tensor = img_tensor.unsqueeze(0)  # 单张图转批量

        # 预处理
        input_tensor = self.transform(img_tensor).to(self.device)

        # 预测深度
        with torch.no_grad():
            depth = self.model(input_tensor)

        # 调整尺寸与输入一致
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=img_tensor.shape[-2:],
            mode="bicubic",
            align_corners=False
        )

        # 归一化到0-1（便于作为掩码权重）
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth.squeeze(0) if is_single else depth


def visualize_batch(batch, save_path=None):
    """可视化一个batch的模糊图与清晰图"""
    batch_size = batch["blur"].shape[0]
    plt.figure(figsize=(15, 5 * batch_size))

    for i in range(batch_size):
        # 反标准化
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
        plt.title("Sharp (GT)")
        plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"可视化已保存到 {save_path}")
    plt.close()


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化Tensor，用于可视化"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def save_reconstructed_results(recon_img, sharp_img, blur_img, save_dir, idx):
    """保存模糊图-重建图-清晰图三联对比"""
    os.makedirs(save_dir, exist_ok=True)

    # 反标准化并转numpy
    blur_np = denormalize(blur_img).permute(1, 2, 0).cpu().numpy()
    recon_np = denormalize(recon_img).permute(1, 2, 0).cpu().numpy()
    sharp_np = denormalize(sharp_img).permute(1, 2, 0).cpu().numpy()

    # 拼接并保存
    combined = np.hstack([blur_np, recon_np, sharp_np])
    combined = (combined * 255).astype(np.uint8)
    save_path = os.path.join(save_dir, f"result_{idx}.png")
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return save_path


# 测试工具类
if __name__ == "__main__":
    # 测试数据集
    dataset = DefocusDataset(txt_path="data/train.txt", img_size=(512, 512))
    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"模糊图形状: {sample['blur'].shape}")

    # 测试深度估计
    if torch.cuda.is_available():
        depth_estimator = DepthEstimator()
        depth = depth_estimator.predict_depth(sample["blur"].cuda())
        print(f"深度图形状: {depth.shape}")
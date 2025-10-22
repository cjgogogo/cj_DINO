import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import DefocusDataset  # 失焦数据集（含模糊图、清晰图、深度标注）
from model import DefocusDeblurModel  # 失焦重建模型（含上述创新模块）
from utils import PSNR, SSIM, DepthConsistencyLoss  # 新增深度一致性损失
import config


def train_one_epoch(model, train_loader, criterion, depth_loss, gram_loss, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        # 加载数据：失焦图、清晰图、深度图（区别于运动模糊的光流）
        blur_imgs = batch["blur"].to(device)  # 失焦模糊图像
        sharp_imgs = batch["sharp"].to(device)  # 清晰标签
        depth_maps = batch["depth"].to(device)  # 深度图（用于深度感知模块）

        # 模型输出：重建图像 + 预测的深度掩码（失焦程度）
        pred_imgs, depth_mask = model(blur_imgs)

        # 损失计算（适配失焦场景）
        l1_loss = criterion(pred_imgs, sharp_imgs)  # 基础重建损失
        d_loss = depth_loss(depth_mask, depth_maps)  # 深度一致性损失（确保失焦程度与深度匹配）
        g_loss = gram_loss(pred_imgs, blur_imgs, model.dino_features)  # 材质Gram约束损失
        total = l1_loss + 0.1 * d_loss + 0.05 * g_loss  # 联合损失

        # 反向传播与优化
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()
        pbar.set_postfix(loss=total.item())

    return total_loss / len(train_loader)


def validate(model, val_loader, metric_psnr, metric_ssim, device):
    model.eval()
    psnr_list = []
    ssim_list = []
    with torch.no_grad():
        for batch in val_loader:
            blur_imgs = batch["blur"].to(device)
            sharp_imgs = batch["sharp"].to(device)
            pred_imgs, _ = model(blur_imgs)

            # 计算评估指标（失焦重建更关注边缘和纹理恢复，SSIM更重要）
            psnr = metric_psnr(pred_imgs, sharp_imgs)
            ssim = metric_ssim(pred_imgs, sharp_imgs)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    # 配置与设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.save_dir, exist_ok=True)

    # 数据集与数据加载器（失焦数据集需包含深度信息）
    train_dataset = DefocusDataset(config.train_txt, config.img_size)
    val_dataset = DefocusDataset(config.val_txt, config.img_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 模型、损失、优化器
    model = DefocusDeblurModel().to(device)
    criterion = torch.nn.L1Loss()
    depth_loss = DepthConsistencyLoss()  # 自定义深度损失
    gram_loss = GramConsistencyLoss(anchor_lib=config.material_anchor_lib)  # 材质Gram损失
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 训练循环
    best_ssim = 0.0
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, depth_loss, gram_loss, optimizer, device)
        val_psnr, val_ssim = validate(model, val_loader, PSNR(), SSIM(), device)

        print(f"Train Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f}")

        # 保存最优模型（以SSIM为指标，更贴合视觉质量）
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            print(f"Saved best model (SSIM: {best_ssim:.4f})")

        scheduler.step()


if __name__ == "__main__":
    main()
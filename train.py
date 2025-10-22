import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 可视化训练过程
from tqdm import tqdm
import numpy as np

# 导入自定义模块（与仓库结构匹配）
from utils import DefocusDataset, visualize_batch, save_reconstructed_results, denormalize
from models.overall_model import DefocusReconModel
from models.gram_anchoring import GramLoss


def calculate_psnr(pred, target):
    """计算PSNR（峰值信噪比，评估重建质量）"""
    mse = nn.MSELoss()(pred, target)
    return 10 * torch.log10(1.0 / mse)  # 假设像素值范围[0,1]


def main():
    # ==============================================
    # 1. 配置参数（根据你的GPU和数据集调整）
    # ==============================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 训练参数
    epochs = 50
    batch_size = 4  # 8G显存建议4，16G显存建议8
    lr = 1e-4  # 初始学习率
    img_size = (512, 512)  # 图像尺寸（需与模型输入匹配）
    val_interval = 1  # 每多少轮验证一次
    save_interval = 10  # 每多少轮保存一次中间模型

    # 路径配置
    data_root = "data"
    train_txt = os.path.join(data_root, "train.txt")
    val_txt = os.path.join(data_root, "val.txt")
    log_dir = "logs"  # TensorBoard日志
    checkpoint_dir = "checkpoints"  # 模型保存目录
    result_dir = "results"  # 重建结果保存目录

    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # 初始化TensorBoard
    writer = SummaryWriter(log_dir)

    # ==============================================
    # 2. 数据加载
    # ==============================================
    # 训练集（启用数据增强）
    train_dataset = DefocusDataset(
        txt_path=train_txt,
        img_size=img_size,
        is_train=True
    )
    # 验证集（不启用增强）
    val_dataset = DefocusDataset(
        txt_path=val_txt,
        img_size=img_size,
        is_train=False
    )

    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError(f"训练集为空！请检查 {train_txt} 中的路径是否正确")
    if len(val_dataset) == 0:
        raise ValueError(f"验证集为空！请检查 {val_txt} 中的路径是否正确")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 多线程加载
        pin_memory=True  # 加速GPU传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 可视化一个训练batch（调试用）
    visualize_batch(next(iter(train_loader)), save_path=os.path.join(result_dir, "train_batch_sample.png"))

    # ==============================================
    # 3. 模型初始化与配置
    # ==============================================
    # 初始化端到端模型
    model = DefocusReconModel().to(device)

    # 加载DINOv3预训练权重（修复路径问题）
    dinov3_weight_path = "models/dinov3_weights/pytorch_model.bin"
    if os.path.exists(dinov3_weight_path):
        try:
            # 仅加载DINOv3部分权重（避免与其他模块冲突）
            dinov3_state_dict = torch.load(dinov3_weight_path, map_location=device)
            model.dinov3_feat.model.load_state_dict(dinov3_state_dict, strict=False)
            print("✅ DINOv3预训练权重加载成功")
        except Exception as e:
            print(f"⚠️ DINOv3权重加载警告：{e}（继续使用随机初始化）")
    else:
        print(f"⚠️ 未找到DINOv3权重 {dinov3_weight_path}（继续使用随机初始化）")

    # 冻结DINOv3前8层（保留预训练特征，仅微调深层）
    for param in list(model.dinov3_feat.model.parameters())[:-4]:
        param.requires_grad = False
    print("✅ 已冻结DINOv3前8层参数，仅微调深层")

    # ==============================================
    # 4. 损失函数与优化器
    # ==============================================
    # 混合损失：像素级损失 + Gram材质一致性损失
    loss_pixel = nn.L1Loss().to(device)  # 主损失：约束像素级重建
    loss_gram = GramLoss().to(device)    # 辅助损失：约束材质细节

    # 优化器（AdamW带权重衰减，防止过拟合）
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 仅优化可训练参数
        lr=lr,
        weight_decay=1e-5
    )

    # 学习率调度器（余弦退火，动态调整学习率）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # 退火周期
        eta_min=1e-6   # 最小学习率
    )

    # ==============================================
    # 5. 训练与验证循环
    # ==============================================
    best_val_psnr = 0.0  # 记录最佳验证PSNR

    for epoch in range(1, epochs + 1):
        # ---------------------------
        # 训练阶段
        # ---------------------------
        model.train()
        train_loss_total = 0.0
        train_psnr_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [训练]")
        for batch_idx, batch in enumerate(pbar):
            # 加载数据
            blur_imgs = batch["blur"].to(device)
            sharp_imgs = batch["sharp"].to(device)

            # 前向传播（含深度掩码与注意力机制）
            pred_imgs = model(blur_imgs)

            # 计算损失
            loss_p = loss_pixel(pred_imgs, sharp_imgs)
            loss_g = loss_gram(pred_imgs, sharp_imgs)
            total_loss = loss_p + 0.1 * loss_g  # Gram损失权重0.1（可微调）

            # 反向传播与优化
            optimizer.zero_grad()  # 清零梯度
            total_loss.backward()  # 反向传播
            optimizer.step()       # 更新参数

            # 计算PSNR（训练集）
            psnr = calculate_psnr(pred_imgs, sharp_imgs)

            # 累计损失与PSNR
            train_loss_total += total_loss.item() * blur_imgs.size(0)
            train_psnr_total += psnr.item() * blur_imgs.size(0)

            # 实时显示进度
            pbar.set_postfix({
                "L1损失": f"{loss_p.item():.4f}",
                "Gram损失": f"{loss_g.item():.4f}",
                "PSNR": f"{psnr.item():.2f} dB"
            })

            # 每100个batch记录一次训练图像（TensorBoard）
            if batch_idx % 100 == 0:
                writer.add_images(
                    "Train/Blur", denormalize(blur_imgs[:4]),  # 取前4张
                    global_step=epoch * len(train_loader) + batch_idx
                )
                writer.add_images(
                    "Train/Reconstructed", denormalize(pred_imgs[:4]),
                    global_step=epoch * len(train_loader) + batch_idx
                )

        # 计算训练集平均损失与PSNR
        train_loss_avg = train_loss_total / len(train_dataset)
        train_psnr_avg = train_psnr_total / len(train_dataset)
        print(f"\nEpoch {epoch} | 训练集平均损失: {train_loss_avg:.4f} | 平均PSNR: {train_psnr_avg:.2f} dB")

        # 记录训练指标到TensorBoard
        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("PSNR/Train", train_psnr_avg, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # ---------------------------
        # 验证阶段（每val_interval轮）
        # ---------------------------
        if epoch % val_interval == 0:
            model.eval()
            val_loss_total = 0.0
            val_psnr_total = 0.0

            with torch.no_grad():  # 关闭梯度计算
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [验证]")
                for batch_idx, batch in enumerate(pbar_val):
                    blur_imgs = batch["blur"].to(device)
                    sharp_imgs = batch["sharp"].to(device)

                    # 前向传播
                    pred_imgs = model(blur_imgs)

                    # 计算损失与PSNR
                    loss_p = loss_pixel(pred_imgs, sharp_imgs)
                    loss_g = loss_gram(pred_imgs, sharp_imgs)
                    total_loss = loss_p + 0.1 * loss_g
                    psnr = calculate_psnr(pred_imgs, sharp_imgs)

                    # 累计指标
                    val_loss_total += total_loss.item() * blur_imgs.size(0)
                    val_psnr_total += psnr.item() * blur_imgs.size(0)

                    # 实时显示进度
                    pbar_val.set_postfix({"验证PSNR": f"{psnr.item():.2f} dB"})

                    # 保存前5个验证样本的重建结果
                    if batch_idx == 0:
                        for i in range(min(5, blur_imgs.size(0))):
                            save_reconstructed_results(
                                recon_img=pred_imgs[i],
                                sharp_img=sharp_imgs[i],
                                blur_img=blur_imgs[i],
                                save_dir=os.path.join(result_dir, f"epoch_{epoch}"),
                                idx=i
                            )

            # 计算验证集平均指标
            val_loss_avg = val_loss_total / len(val_dataset)
            val_psnr_avg = val_psnr_total / len(val_dataset)
            print(f"Epoch {epoch} | 验证集平均损失: {val_loss_avg:.4f} | 平均PSNR: {val_psnr_avg:.2f} dB")

            # 记录验证指标到TensorBoard
            writer.add_scalar("Loss/Val", val_loss_avg, epoch)
            writer.add_scalar("PSNR/Val", val_psnr_avg, epoch)
            writer.add_images(
                "Val/Reconstructed", denormalize(pred_imgs[:4]),  # 取前4张
                global_step=epoch
            )

            # 保存最佳模型（按验证PSNR）
            if val_psnr_avg > best_val_psnr:
                best_val_psnr = val_psnr_avg
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_psnr": best_val_psnr
                }, os.path.join(checkpoint_dir, "best_model.pth"))
                print(f"📌 保存最佳模型（PSNR: {best_val_psnr:.2f} dB）")

        # ---------------------------
        # 其他操作
        # ---------------------------
        # 保存中间模型
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))
            print(f"💾 保存中间模型（epoch {epoch}）")

        # 更新学习率
        scheduler.step()

    # 训练结束
    print(f"\n训练完成！最佳验证PSNR: {best_val_psnr:.2f} dB（模型保存于 {checkpoint_dir}）")
    writer.close()


if __name__ == "__main__":
    main()
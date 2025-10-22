import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv3FeatureExtractor(torch.nn.Module):
    """
    DINOv3特征提取模块：加载预训练的DINOv3模型，输出多尺度特征，
    适配失焦模糊重建任务，保留细节与语义特征。
    """

    def __init__(self, pretrained_path="models/dinov3_weights"):
        """
        初始化函数：加载本地DINOv3权重和图像处理器，冻结部分层。
        Args:
            pretrained_path: 本地DINOv3权重文件夹路径（含config.json和pytorch_model.bin）
        """
        super().__init__()

        # 1. 加载图像处理器（用于后续预处理，与模型训练时的预处理一致）
        self.processor = AutoImageProcessor.from_pretrained(
            pretrained_path,
            local_files_only=True  # 仅使用本地文件，不联网
        )

        # 2. 加载DINOv3模型（视觉Transformer，自监督预训练）
        self.dinov3 = AutoModel.from_pretrained(
            pretrained_path,
            device_map="auto",  # 自动分配到GPU（优先）或CPU
            local_files_only=True  # 仅使用本地权重
        )
        # print("DINOv3模型的所有属性", dir(self.dinov3))
        # 3. 冻结部分层（保留通用特征提取能力，避免过拟合）
        # 冻结前2层：保留底层细节特征提取能力（适合模糊区域的弱细节）
        # 微调后2层：适配失焦模糊的特定特征（如深度关联）
        for param in self.dinov3.layer[:2].parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传播：输入图像张量，输出4个尺度的DINOv3特征。
        Args:
            x: 预处理后的图像张量，形状为 [B, 3, H, W]（B=批量大小，H=512，W=512）
        Returns:
            feats: 列表，包含4个尺度的特征，每个特征形状为 [B, num_tokens, hidden_dim]
                   其中 num_tokens = (H/16)×(W/16) + 1（16是DINOv3的patch大小，+1是CLS token）
        """
        # 输入DINOv3，输出所有隐藏层状态（output_hidden_states=True）
        outputs = self.dinov3(x, output_hidden_states=True)
        # print(f"隐藏层总数：{len(outputs.hidden_states)}")
        # print(f"可用索引范围：0~{len(outputs.hidden_states)-1}")
        # 提取4个关键尺度的特征（根据实验验证，这4层特征对失焦重建最有效）
        # - 浅层（第5层）：保留边缘、纹理等细节特征（适合恢复模糊区域的弱细节）
        # - 中层（第10、15层）：兼顾细节与语义（适合区分模糊类型）
        # - 深层（第20层）：包含语义和深度关联信息（适合注意力联动的深度约束）
        feat_2 = outputs.hidden_states[2]  # 浅层特征
        feat_5 = outputs.hidden_states[5]  # 中浅层特征
        feat_8 = outputs.hidden_states[8]  # 中深层特征
        feat_11 = outputs.hidden_states[11]  # 深层特征

        return [feat_2, feat_5, feat_8, feat_11]


# 测试模块：验证特征提取是否正常（直接运行该脚本即可测试）
if __name__ == "__main__":
    # 初始化模型（默认加载本地路径）
    model = DINOv3FeatureExtractor()
    # 打印模型设备（确认是否使用GPU）
    print(f"DINOv3模型加载到设备：{model.dinov3.device}")

    # 生成测试输入（模拟2张512×512的图像，通道数3）
    test_x = torch.randn(2, 3, 512, 512).to(model.dinov3.device)

    # 提取特征
    model.eval()  # 推理模式
    with torch.no_grad():  # 关闭梯度计算，节省内存
        feats = model(test_x)

    # 打印各尺度特征形状（验证是否符合预期）
    print("\n各尺度特征形状验证：")
    for i, feat in enumerate(feats):
        layer_idx = 5 + i * 5  # 5, 10, 15, 20
        print(f"第{layer_idx}层特征：{feat.shape}")
        # 预期形状：[2, 1025, 384] 
        # 解释：2=批量大小，1025=32×32（512/16=32）+1（CLS token），384=隐藏层维度
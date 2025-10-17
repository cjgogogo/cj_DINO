# DINOv3 模型使用示例
# 本脚本展示了如何使用 Hugging Face transformers 库加载和使用 DINOv3 模型进行图像特征提取

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

# 加载测试图像
# 使用 COCO 数据集中的一张示例图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print(f"图像加载完成，尺寸: {image.size}")

# 指定要使用的预训练模型名称
# 使用本地下载的 DINOv3 ViT-S/16 模型
pretrained_model_name = "./dinov3-vits16-pretrain-lvd1689m"

# 加载与模型匹配的图像处理器（Image Processor）
# 图像处理器负责将输入图像转换为模型所需的格式
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
print("图像处理器加载完成")

# 加载预训练的 DINOv3 模型
# device_map="auto" 参数让 transformers 自动选择最佳设备（GPU/CPU）
model = AutoModel.from_pretrained(
    pretrained_model_name,
    device_map="auto",
)
print(f"模型加载完成，设备: {model.device}")

# 预处理输入图像
# return_tensors="pt" 返回 PyTorch 张量格式
# .to(model.device) 确保输入数据与模型在同一设备上
inputs = processor(images=image, return_tensors="pt").to(model.device)

print(f"输入预处理完成，输入形状: {inputs['pixel_values'].shape}")

# 执行模型推理
# torch.inference_mode() 上下文管理器优化推理性能，禁用梯度计算
with torch.inference_mode():
    outputs = model(**inputs)
print(outputs.keys())
# 提取汇聚输出（通常来自 CLS token）
# pooler_output 是图像的全局特征表示，适用于下游任务
pooled_output = outputs.pooler_output
print(f"汇聚输出形状: {pooled_output.shape}")
print(f"汇聚输出前5个元素: {pooled_output[0][:5].tolist()}")

# 提取最后隐藏状态（每个图像块的特征）
# last_hidden_state 包含每个图像块（patch）的特征向量
last_hidden_state = outputs.last_hidden_state
print(f"最后隐藏状态形状: {last_hidden_state.shape}")
print(f"包含 {last_hidden_state.shape[1]} 个图像块，每个块的特征维度为 {last_hidden_state.shape[2]}")

# 打印模型架构和参数信息
print("\n=== DINOv3 模型架构信息 ===")

# 计算模型参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"可训练参数量: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

# 打印模型配置信息
config = model.config
print(f"\n模型架构配置:")
print(f"- 模型类型: {config.model_type}")
print(f"- 隐藏层维度: {config.hidden_size}")
print(f"- 注意力头数: {config.num_attention_heads}")
print(f"- Transformer层数: {config.num_hidden_layers}")
print(f"- 中间层维度: {config.intermediate_size}")
print(f"- 图像尺寸: {config.image_size}x{config.image_size}")
print(f"- Patch尺寸: 16x16 (推断)")
print(f"- Register tokens数量: {config.num_register_tokens}")

# 计算理论计算量 (FLOPs)
# 简化的 FLOPs 估算，主要考虑 Transformer 层的计算
batch_size = inputs['pixel_values'].shape[0]
seq_length = last_hidden_state.shape[1]  # 201 (196 patches + 1 CLS + 4 registers)
hidden_size = config.hidden_size
num_layers = config.num_hidden_layers
intermediate_size = config.intermediate_size

# 每层的主要计算量估算
# 1. Multi-head attention: 4 * batch_size * seq_length * hidden_size^2
# 2. Feed-forward network: 2 * batch_size * seq_length * hidden_size * intermediate_size
attention_flops_per_layer = 4 * batch_size * seq_length * hidden_size * hidden_size
ffn_flops_per_layer = 2 * batch_size * seq_length * hidden_size * intermediate_size
total_flops_per_layer = attention_flops_per_layer + ffn_flops_per_layer
total_flops = total_flops_per_layer * num_layers

print(f"\n计算量估算 (单次前向传播):")
print(f"- 每层注意力机制 FLOPs: {attention_flops_per_layer/1e9:.2f} GFLOPs")
print(f"- 每层前馈网络 FLOPs: {ffn_flops_per_layer/1e9:.2f} GFLOPs")
print(f"- 总计算量: {total_flops/1e9:.2f} GFLOPs")

print(model)


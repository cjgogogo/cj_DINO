class DINOv3FeatureExtractor(torch.nn.Module):
    def __init__(self, pretrained_path="./models/dinov3_weights"):  # 本地路径
        super().__init__()
        # 加载本地处理器和模型
        self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
        self.dinov3 = AutoModel.from_pretrained(
            pretrained_path,
            device_map="auto",  # 自动分配到GPU/CPU
            local_files_only=True  # 强制使用本地文件，不联网
        )
        # 冻结部分层（保持不变）
        for param in self.dinov3.encoder.layer[:2].parameters():
            param.requires_grad = False
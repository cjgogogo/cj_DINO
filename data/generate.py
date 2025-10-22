import os

# 数据集根目录（根据实际路径修改）
dataset_root = "dd_dp_dataset_png"
# 子集划分（train/val/test）
subsets = ["train", "val", "test"]
# 模糊图文件夹后缀（如"_l"）和清晰图后缀（如"_c"）
blur_suffix = "l"
sharp_suffix = "c"
# 图像格式
img_extensions = (".png", ".jpg", ".jpeg")

for subset in subsets:
    # 模糊图和清晰图的文件夹路径
    blur_dir = os.path.join(dataset_root, f"{subset}_{blur_suffix}")
    sharp_dir = os.path.join(dataset_root, f"{subset}_{sharp_suffix}")
    # 输出txt路径
    output_txt = os.path.join(os.path.dirname(dataset_root), f"{subset}.txt")

    if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
        print(f"警告：{blur_dir} 或 {sharp_dir} 不存在，跳过该子集")
        continue

    # 收集配对路径
    pair_list = []
    # 遍历模糊图文件夹
    for img_name in os.listdir(blur_dir):
        if img_name.lower().endswith(img_extensions):
            # 拼接模糊图和清晰图路径（假设文件名相同）
            blur_path = os.path.join(blur_dir, img_name)
            sharp_path = os.path.join(sharp_dir, img_name)
            # 检查清晰图是否存在
            if os.path.exists(sharp_path):
                pair_list.append(f"{blur_path} {sharp_path}")

    # 写入txt
    with open(output_txt, "w") as f:
        f.write("\n".join(pair_list))
    print(f"生成 {output_txt}，共 {len(pair_list)} 对数据")
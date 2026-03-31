import numpy as np
import os
from skimage import measure
from tqdm import tqdm

def create_consolidated_mask(file_list, output_path, area_ratio_threshold=0.03, mask_key='mask'):
    """
    读取所有npz文件，过滤小面积mask，并合并为一个大的mask。
    
    参数:
    - file_list: 包含所有npz文件路径的列表
    - output_path: 输出npz文件的路径
    - area_ratio_threshold: 面积阈值（相对于全图面积的比例）
    - mask_key: npz文件中mask对应的键名 (通常是 'mask' 或 'arr_0')
    """
    
    combined_mask = None
    total_files = len(file_list)
    
    print(f"开始处理 {total_files} 个文件...")

    for i, file_path in enumerate(tqdm(file_list)):
        # 1. 加载 npz 文件
        with np.load(file_path) as data:
            if mask_key not in data:
                # 如果找不到指定的键，尝试列出所有键并取第一个
                actual_key = data.files[0]
                mask = data[actual_key]
            else:
                mask = data[mask_key]

        # 确保是二值图
        mask = (mask > 0).astype(np.uint8)
        
        # 2. 计算面积阈值
        h, w = mask.shape
        total_pixels = h * w
        min_area_pixels = total_pixels * area_ratio_threshold
        
        # 3. 识别连通域并过滤
        # connectivity=2 表示 8-连通（包括对角线）
        labels = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labels)
        
        # 创建一个清理后的临时 mask
        cleaned_mask = np.zeros_like(mask)
        
        for reg in regions:
            if reg.area >= min_area_pixels:
                # 只有面积足够大的块才会被保留
                cleaned_mask[labels == reg.label] = 1
        
        # 4. 合并到总 mask
        if combined_mask is None:
            combined_mask = cleaned_mask
        else:
            # 使用逻辑“或”运算合并
            combined_mask = np.logical_or(combined_mask, cleaned_mask).astype(np.uint8)

    # 5. 保存结果
    if combined_mask is not None:
        np.savez_compressed(output_path, mask=combined_mask)
        print(f"\n处理完成！")
        print(f"合并后的标注已保存至: {output_path}")
        print(f"最终 Mask 尺寸: {combined_mask.shape}")
        print(f"激活像素总数: {np.sum(combined_mask)}")
    else:
        print("未发现有效 mask 数据。")

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的 npz 文件都在这个目录下
    input_dir = "./data/masks_npz"
    output_file = "final_merged_label.npz"
    
    # 获取目录下所有的 npz 文件
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not files:
        print("错误：未在目录下找到 npz 文件。")
    else:
        # 执行程序：去除面积小于 3% 的块并合并
        create_consolidated_mask(files, output_file, area_ratio_threshold=0.03)
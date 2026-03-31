#本程序内存占用较大
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from loguru import logger
import time
from sklearn.metrics import precision_recall_curve

def compute_pixel_metrics_and_save_masks(heatmap_dir, mask_dir, save_mask_dir=None):
    """
    heatmap_dir: 存放 .npy 原始热力图的文件夹
    mask_dir: 存放 Ground Truth 二值掩码图 (.png) 的文件夹
    save_mask_dir: 如果不为 None，则根据最佳阈值将结果保存为黑白 .png 到该路径
    """
    
    npy_files = sorted(glob.glob(os.path.join(heatmap_dir, "*.npy")))
    if not npy_files:
        print(f"错误：在 {heatmap_dir} 中未找到 .npy 文件")
        return

    all_gt = []
    all_pr = []
    processed_data = [] # 新增：用于存储图片信息，方便后续保存

    print(f"第一阶段：开始读取 {len(npy_files)} 张图片并计算指标...")

    for npy_path in tqdm(npy_files):
        # --- A. 加载热力图 ---
        pr_map = np.load(npy_path) 

        # --- B. 匹配并加载对应的掩码图 (GT) ---
        file_name = os.path.basename(npy_path).replace(".npy", "_mask.png")
        mask_path = os.path.join(mask_dir, file_name)
        
        if not os.path.exists(mask_path):
            continue
            
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_bin = (gt_mask > 127).astype(np.int32)
        h, w = gt_mask.shape

        # --- C. 上采样热力图 ---
        pr_map_resized = cv2.resize(pr_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- D. 存入全局列表用于指标 ---
        all_gt.append(gt_mask_bin.flatten())
        all_pr.append(pr_map_resized.flatten())

        # --- E. 存入临时列表用于后续保存 ---
        # 存入原始 pr_map（小图）以节省内存，保存时再 resize
        processed_data.append({
            'name': file_name,
            'h': h,
            'w': w,
            'pr_map': pr_map 
        })

    # --- F. 合并数据并计算 F1 ---
    print("\n正在合并数据并搜索最优阈值...")
    y_true = np.concatenate(all_gt)
    y_scores = np.concatenate(all_pr)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_pre = precisions[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]

    print("-" * 30)
    logger.info(f"Max F1-score:  {best_f1*100:.2f}%")
    logger.info(f"Precision:     {best_pre*100:.2f}%")
    logger.info(f"Recall:        {best_recall*100:.2f}%")
    logger.info(f"Best Threshold: {best_threshold:.6f}")
    print("-" * 30)


    # --- G. 根据最佳阈值保存黑白掩码图 ---
    if save_mask_dir:
        print(f"\n第二阶段：根据最佳阈值 {best_threshold:.4f} 生成并保存黑白掩码...")
        if not os.path.exists(save_mask_dir):
            os.makedirs(save_mask_dir)
            
        for data in tqdm(processed_data):
            # 重新 resize 到原图大小
            full_res_pr = cv2.resize(data['pr_map'], (data['w'], data['h']), interpolation=cv2.INTER_LINEAR)
            
            # 执行二值化
            binary_mask = np.zeros((data['h'], data['w']), dtype=np.uint8)
            binary_mask[full_res_pr > best_threshold] = 255
            
            # 保存
            save_path = os.path.join(save_mask_dir, data['name'])
            cv2.imwrite(save_path, binary_mask)
            
        print(f"保存完成！结果存放在: {save_mask_dir}")

    return best_f1, best_threshold
if __name__ == "__main__":
    # 请根据你的实际路径修改
    HEATMAP_DIR = "e:/cvlab/data0312/PF/musc_results/ViT-B-32"  # MUSC 保存的 .npy 文件夹
    MASK_DIR = "e:/cvlab/data0312/PF/gt_masks"      # 原始标注二值图文件夹
    SAVE_MASK_DIR="e:/cvlab/data0312/PF/musc_results/ViT-B-32/direct_output"
    LOG_NAME=f"log_{time.strftime('%Y%m%d_%H%M%S')}.log"
    LOG_PATH=os.path.join(SAVE_MASK_DIR,LOG_NAME)
    logger.add(LOG_PATH, rotation="10 MB", encoding="utf-8")
    
    compute_pixel_metrics_and_save_masks(HEATMAP_DIR, MASK_DIR,SAVE_MASK_DIR)



import numpy as np
import cv2
from PIL import Image
import os
import glob
import pandas as pd
from loguru import logger
import time
# ================= 配置区域 =================
# 建议通过 base_name 统一管理
INPUT_DIR = {
    "image": "e:/cvlab/data0312/PF/images",                      # 原图目录
    "sam": "e:/cvlab/data0312/PF/sam_results/npz/",              # SAM 结果目录
    # "musc": "e:/cvlab/data0312/PF/musc_results/ViT-L-14-336" , # MUSC 热力图目录
    "musc": "e:/cvlab/data0312/PF/musc_results/ViT-B-32",
    "musc_norm":"e:/cvlab/data0312/PF/musc_results/ViT-B-32/norm",
    # "musc_norm":"e:/cvlab/data0312/PF/musc_results/ViT-L-14-336/norm",
    "gt":"e:/cvlab/data0312/PF/gt_masks"
}
logger.add(f"log_{time.strftime('%Y%m%d_%H%M%S')}.log", rotation="10 MB", encoding="utf-8")
IMAGE_PATHS = []
IMAGE_PATHS.extend(glob.glob(os.path.join(INPUT_DIR['image'], '*.png')))
FILE_NAMES = [os.path.splitext(os.path.basename(p))[0] for p in IMAGE_PATHS]
# logger.info(f"Found {len(FILE_NAMES)} images. Starting processing...")
# FILE_NAMES=['PF207','PF233','PF244','PF264']
OUTPUT_DIR = "e:/cvlab/data0312/PF/final_output_test/"            # 最终结果输出主目录

# 判定阈值调优
AREA_RATIO_THRESHOLD = 0.03  # 面积判定阈值  #0.9 0.12
# decision_cfg = {
#     "high_threshold": 0.90,
#     # "low_threshold": 0.1,
#     "mean_thresh": 0.76,
#     "high_ratio_thresh": 0.12,
#     # "low_ratio_thresh": 0.12
# }

decision_cfg = {
    # "high_threshold": 0.90,
    "low_threshold": 0.4,
    "mean_thresh": 0.735,
    # "high_ratio_thresh": 0.12,
    "low_ratio_thresh":0.10
}
logger.info("cfg:{}",decision_cfg)
# decision_cfg = {
#     "high_threshold": 0.85,
#     # "low_threshold": 0.2,
#     "mean_thresh": 0.91,
#     "high_ratio_thresh": 0.13,
#     # "low_ratio_thresh": 0.12
# }

# 确保输出目录存在
for sub_dir in ["masks_npz", "heatmaps_upsampled", "visual_results"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)
# ============================================
def global_normalization(input_dir, output_dir, percentile=99.0):
    """
    对文件夹下的所有 .npy 文件执行全局 99% 分位数归一化
    """
    # 1. 获取所有文件路径
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        logger.info(f"错误：在 {input_dir} 中未找到 .npy 文件")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 第一阶段：计算全局统计量 ---
    logger.info(f"正在分析 {len(npy_files)} 个文件以获取全局统计量...")
    all_values_sampled = []
    
    for f_path in npy_files:
        data = np.load(f_path)
    # 取消 [::50]，使用全量数据
        sampled_data = data.flatten() 
        all_values_sampled.append(sampled_data)
    
    # 合并采样数据
    combined_scores = np.concatenate(all_values_sampled)
    
    global_min = combined_scores.min()
    global_vmax = np.percentile(combined_scores, percentile)
    
    # 防止除以 0（如果图片全是纯色）
    denom = global_vmax - global_min if global_vmax > global_min else 1e-7
    
    logger.info(f"\n统计完成！")
    logger.info(f"全局最小值 (vmin): {global_min:.6f}")
    logger.info(f"全局 {percentile}% 分位数 (vmax): {global_vmax:.6f}")
    
    # 释放统计用的内存
    del combined_scores
    del all_values_sampled

    # --- 第二阶段：应用归一化并保存 ---
    logger.info(f"\n正在应用归一化并保存至 {output_dir}...")
    for f_path in npy_files:
        data = np.load(f_path)
        
        # 线性映射
        norm_data = (data - global_min) / denom
        
        # 关键步骤：截断到 [0, 1]
        # 高于 99% 分位数的像素会被固定为 1.0 (最红/最强异常)
        norm_data = np.clip(norm_data, 0.0, 1.0)
        
        # 构造保存路径
        file_name = os.path.basename(f_path)
        save_path = os.path.join(output_dir, file_name)
        
        np.save(save_path, norm_data)

    logger.info("\n所有文件处理完毕！")
def interpolation_save(heatmap, target_shape, save_path=None):
    """
    上采样热力图并保存。
    同时保存原始 .npy 数据和 OpenCV 伪彩色可视化图。
    
    参数:
    - heatmap: 归一化后的热力图 (H_small, W_small), 0.0~1.0
    - target_shape: 目标图像形状 (H, W, C) 或 (H, W)
    - save_path: .npy 文件的保存路径
    - save_vis: 是否保存 OpenCV 伪彩色图
    """
    # 1. 执行上采样
    target_h, target_w = target_shape[:2]
    upsampled_heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    if save_path:
        # 2. 保存原始 .npy 数据（用于后续算法计算，不丢失精度）
        np.save(save_path, upsampled_heatmap)
        # logger.info(f"-> 原始热力图数据已保存: {save_path}")

        # 3. 使用 OpenCV 保存可视化图片
        # 将 0.0-1.0 映射到 0-255 整数
        # np.clip 确保数值不会因为浮点误差超出 255
        heatmap_8bit = (np.clip(upsampled_heatmap, 0, 1) * 255).astype(np.uint8)
        
        # 应用伪彩色映射 (JET 模式：蓝色为低分，红色为高分)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # 构造图片保存路径（将 .npy 换成 .jpg）
        vis_path = os.path.splitext(save_path)[0] + ".png"
        cv2.imwrite(vis_path, heatmap_color)
        logger.info(f"-> 上采样热力图及可视化图已保存: {vis_path}")

    return upsampled_heatmap

def calculate_anomaly_index(mask, heatmap, cfg,high):
    """计算异常指标并判定"""
    if mask.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    region_scores = heatmap[mask > 0]
    area = len(region_scores)
    if area == 0:
        return False, {"mean_score": 0, "low_ratio": 0}
    mean_score = np.mean(region_scores)
    std_score = np.std(region_scores)
    if high:
        high_pixel_count = np.sum(region_scores > cfg["high_threshold"])
        # low_pixel_count = np.sum(region_scores < cfg["low_threshold"])
        high_ratio = high_pixel_count / area
        # low_ratio = low_pixel_count / area
        # top_k = max(1, len(region_scores) // 10)
        # s_top10 = np.mean(np.sort(region_scores)[-top_k:])
        # 判定逻辑
        # is_normal = (mean_score < cfg["mean_thresh"]) or (low_ratio > cfg["low_ratio_thresh"])
        is_defect =not (low_ratio>cfg["low_ratio_thresh"])
        metrics = {"mean_score": mean_score, "high_ratio": low_ratio, "std_score":std_score,"area": area}
        return not is_defect, metrics
    else:
        # high_pixel_count = np.sum(region_scores > cfg["high_threshold"])
        low_pixel_count = np.sum(region_scores < cfg["low_threshold"])
        # high_ratio = high_pixel_count / area
        low_ratio = low_pixel_count / area
        # top_k = max(1, len(region_scores) // 10)
        # s_top10 = np.mean(np.sort(region_scores)[-top_k:])
        # 判定逻辑
        if std_score<0.085:
            is_normal = (mean_score < cfg["mean_thresh"]-3) or (low_ratio > cfg["low_ratio_thresh"])
        # is_defect =not (low_ratio>cfg["low_ratio_thresh"])
        else:
            is_normal = (mean_score < cfg["mean_thresh"]) or (low_ratio > cfg["low_ratio_thresh"])
        metrics = {"mean_score": mean_score, "low_ratio": low_ratio, "std_score":std_score,"area": area}
        return is_normal, metrics
def calculate_metrics(pred, gt):
    """
    计算单张图的像素级指标
    pred, gt: 0/1 二值化矩阵
    """
    # 确保是布尔类型
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    # 交并比
    iou = intersection / union if union > 0 else 1.0
    # 召回率 (查全率)
    recall = intersection / gt_sum if gt_sum > 0 else 1.0
    # 精确率 (查准率)
    precision = intersection / pred_sum if pred_sum > 0 else 1.0
    # Dice系数 (F1-Score)
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    
    return iou, recall, precision, dice
def outcome_save(pred_mask, raw_image_rgb, save_dir, file_name):
    """保存最终展示图（黑白图 + 蓝色透明标注图）"""
    # 1. 黑白图
    bw_mask = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_final_mask.png"), bw_mask)

    # 2. 蓝色标注图
    img_bgr = cv2.cvtColor(raw_image_rgb, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    overlay[pred_mask > 0] = [139, 0, 0] # 深蓝色
    
    visual_img = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
    # 画白边
    contours, _ = cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visual_img, contours, -1, (255, 255, 255), 1)
    
    cv2.imwrite(os.path.join(save_dir, f"{file_name}_visual.jpg"), visual_img)

# --- 主程序流 ---
#全体归一化
PERCENTAGE=99.9
logger.info("PERCENTAGE:{}",PERCENTAGE)
global_normalization(INPUT_DIR['musc'],INPUT_DIR['musc_norm'],PERCENTAGE)
#量化指标结果
results_list = []
# 1. 加载数据
for FILE_ID in FILE_NAMES:
    gt_path = os.path.join(INPUT_DIR["gt"], f"{FILE_ID}_mask.png")
    img_path = os.path.join(INPUT_DIR["image"], f"{FILE_ID}.png")
    raw_img = np.array(Image.open(img_path).convert("RGB"))
    gt_img = cv2.imread(gt_path, 0)
    gt_mask = (gt_img > 127).astype(np.uint8) # 二值化处理

    masks_path = os.path.join(INPUT_DIR["sam"], f"{FILE_ID}.npz")
    masks_data = np.load(masks_path)
    all_raw_masks = masks_data['masks']
    all_raw_scores = masks_data['scores']

    heatmap_path = os.path.join(INPUT_DIR["musc_norm"], f"{FILE_ID}.npy")
    heatmap_low = np.load(heatmap_path)

    # 2. 上采样热力图到原图尺寸并保存热力图
    h_upsampled_path = os.path.join(OUTPUT_DIR, "heatmaps_upsampled", f"{FILE_ID}_upsampled.npy")
    # upsampled_heatmap = interpolation_save(heatmap_low, raw_img.shape, h_upsampled_path)
    upsampled_heatmap = interpolation_save(heatmap_low, raw_img.shape, None)

    # 3. 筛选掩码
    sam_area_total = raw_img.shape[0] * raw_img.shape[1]
    final_segmentation = []
    final_predicted_iou = []

    logger.info(f"开始筛选 {FILE_ID} 的掩码，原始目标数: {len(all_raw_masks)}")

    for i, m in enumerate(all_raw_masks):
        area_px = np.sum(m)
        # 逻辑分流
        if area_px > sam_area_total * AREA_RATIO_THRESHOLD:
            # 大面积逻辑：热力图投票
            is_normal, metrics = calculate_anomaly_index(m, upsampled_heatmap, decision_cfg,high=False)

            if is_normal:
                logger.info("is_normal")
                logger.info("mean_score:{}",metrics['mean_score'])
                logger.info("low_ratio:{}",metrics['low_ratio'])
                logger.info("std_score:{}",metrics['std_score'])
                logger.info("------")
            else:
                final_segmentation.append(m)
                final_predicted_iou.append(all_raw_scores[i])
                logger.info("is_defect")
                logger.info("mean_score:{}",metrics['mean_score'])
                logger.info("low_ratio:{}",metrics['low_ratio'])
                logger.info("std_score:{}",metrics['std_score'])
                logger.info("------")
        else:
            # 小面积逻辑：直接保留（气孔）
            final_segmentation.append(m)
            final_predicted_iou.append(all_raw_scores[i])

    # 4. 保存筛选后的 NPZ 数据
    if len(final_segmentation) > 0:
        save_npz_path = os.path.join(OUTPUT_DIR, "masks_npz", f"{FILE_ID}_filtered.npz")
        np.savez_compressed(
            save_npz_path, 
            masks=np.stack(final_segmentation), 
            scores=np.array(final_predicted_iou)
        )
        
        # 5. 生成合并掩码并保存图像
        pred_mask_merged = (np.sum(final_segmentation, axis=0) > 0).astype(np.uint8)
        visual_save_dir = os.path.join(OUTPUT_DIR, "visual_results")
        outcome_save(pred_mask_merged, raw_img, visual_save_dir, FILE_ID)
        # 计算指标

        iou, recall, precision, dice = calculate_metrics(pred_mask_merged, gt_mask)
        results_list.append({
        "FileName": FILE_ID,
        "IoU": iou,
        "Recall": recall,
        "Precision": precision,
        "Dice": dice
    })
        logger.info(f"任务完成！判定后缺陷数: {len(final_predicted_iou)}")
    else:
        logger.info("警告：该图片未发现任何有效缺陷（所有大面积区域均被热力图判定为正常）。")

logger.info("汇总统计指标...")
df = pd.DataFrame(results_list)
summary = {
    "mIoU": df["IoU"].mean(),
    "mRecall": df["Recall"].mean(),
    "mPrecision": df["Precision"].mean(),
    "mDice": df["Dice"].mean()
}

logger.info("\n" + "="*30)
logger.info(" 量化评价报告 ")
logger.info("="*30)
logger.info(df.to_string(index=False))
logger.info("-" * 30)
logger.info(f"平均交并比 (mIoU): {summary['mIoU']:.4f}")
logger.info(f"平均召回率 (mRecall): {summary['mRecall']:.4f}")
logger.info(f"平均精确率 (mPrecision): {summary['mPrecision']:.4f}")
logger.info(f"平均F1-Score系数: {summary['mDice']:.4f}")
logger.info("="*30)

# 5. 保存结果到 CSV
df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False)
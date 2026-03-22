"""
用于对AMG生成的分割图像得到的大面积脱落进行异常判断的程序
输入为musc得到的热力图和sam得到的分割掩码
"""


import numpy as np
import cv2
from PIL import Image
# 1. 加载数据
sam_image = Image.open('sam_results/image_001.png').convert("RGB")
sam_image_np = np.array(sam_image)
masks_data = np.load('sam_results/image_001.npz')['masks'] # 形状: (N, H, W)
heatmap = np.load('musc_results/image_001.npy')            # 形状: (H, W)
gt = cv2.imread('gt_masks/image_001.png', 0) / 255.0      # 真实 Mask

final_masks = []
height, width = sam_image_np.shape[:2]
sam_are=height*width
# 2. 遍历 SAM 分出的每一个块
for m in masks_data:
    area = np.sum(m)

    
    if area < 500: # 假设这是微小气孔的阈值
        final_masks.append(m)
            
    else: # 大面积脱落逻辑
        # 计算该 Mask 区域在热力图中的平均分
        score = np.mean(heatmap[m == 1]) 
        if score > 0.5: # 判定为缺陷
            final_masks.append(m)

# 3. 合并预测结果并计算 IoU
pred_mask = (np.sum(final_masks, axis=0) > 0).astype(np.uint8)
iou = calculate_iou(pred_mask, gt) # 与真实掩码图对比
print(f"当前图像检测 IoU: {iou}")
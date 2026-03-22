# #auto mask
# import matplotlib.patches as patches
# import torch
# import torchvision
# print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("CUDA is available:", torch.cuda.is_available())
# import sys

# import os
# # if using Apple MPS, fall back to CPU for unsupported ops
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image


# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )


# np.random.seed(3)

# # def show_anns(anns, borders=True):
# #     if len(anns) == 0:
# #         return
# #     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
# #     ax = plt.gca()
# #     ax.set_autoscale_on(False)

# #     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
# #     img[:, :, 3] = 0
# #     for ann in sorted_anns:
# #         m = ann['segmentation']
# #         color_mask = np.concatenate([np.random.random(3), [0.5]])
# #         img[m] = color_mask 
# #         if borders:
# #             import cv2
# #             contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# #             # Try to smooth contours
# #             contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
# #             cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

# #     ax.imshow(img)


# def save_anns_with_labels(image, anns, save_path):
#     plt.figure(figsize=(20, 20))
#     plt.imshow(image)
    
#     if len(anns) > 0:
#         # 按面积排序
#         sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#         ax = plt.gca()
#         ax.set_autoscale_on(False)

#         img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#         img[:, :, 3] = 0
        
#         for ann in sorted_anns:
#             m = ann['segmentation']
#             color_mask = np.concatenate([np.random.random(3), [0.5]])
#             img[m] = color_mask 
            
#             # 绘制框和分数
#             x, y, w, h = ann['bbox']
#             score = ann['predicted_iou']
#             rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.8)
#             ax.add_patch(rect)
#             ax.text(x, y - 5, f"{score:.2f}", color='yellow', fontsize=10, weight='bold', 
#                     bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

#         ax.imshow(img)
    
#     plt.axis('off')
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close() # 重要：释放内存，否则循环多图片会崩

# image = Image.open('images/PF259.jpg')
# image = np.array(image.convert("RGB"))




# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

# masks = mask_generator.generate(image)


# print(len(masks))
# print(masks[0].keys())


# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns_with_labels(masks, borders=True, show_box=True, show_score=True)
# plt.axis('off')
# plt.show() 





# #auto mask
# import glob
# import matplotlib.patches as patches
# import torch
# import torchvision
# print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("CUDA is available:", torch.cuda.is_available())
# import sys
# import gc
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# import os
# # if using Apple MPS, fall back to CPU for unsupported ops
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image


# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
# elif device.type == "mps":
#     print(
#         "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#         "give numerically different outputs and sometimes degraded performance on MPS. "
#         "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#     )

#基于npu的引入
import cv2
import glob
import matplotlib.patches as patches
import torch
import torchvision
import sys
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("Successfully imported torch_npu!")
except ImportError:
    print("Warning: torch_npu not found. Please ensure you are in an Ascend NPU environment.")
    
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
device = torch.device("npu" if torch.npu.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"NPU is available: {torch.npu.is_available()}")
print(f"Current device: {device}")

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

np.random.seed(3)

# 1. 基础配置
input_dir = "../data/img"      # 你的图片原始文件夹
output_dir = "../data/SR_sam2_l"    # 结果保存文件夹
os.makedirs(output_dir, exist_ok=True)


# 3. 加载模型 (在循环外只加载一次)
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    # 1. 增加采样点密度，确保覆盖细小缺陷
    points_per_side=64,             # 从32提高到64，甚至128（如果显存够）
    
    # 2. 降低质量门槛，捕获模糊或低对比度的边缘
    pred_iou_thresh=0.7,           # 降低此值（原0.85），允许模型输出“不那么确定”的细长裂纹
    stability_score_thresh=0.7,    # 降低此值（原0.95），对模糊边缘更宽容
    
    # 3. 开启切片扫描，这是找细小目标的“神技”
    crop_n_layers=1,                # 设为1，模型会把图切开看，小缺陷在切片里显得更大
    crop_n_points_downscale_factor=2,
    
    # 4. 后处理：防止误杀极小坑洞
    min_mask_region_area=50,        # 调小（原100），确保像图2里那种微小点不被删掉
    
    # 5. 重叠控制
    box_nms_thresh=0.7,             # 保持0.7，防止同一个裂缝被碎成太多框
    
    # 6. 多掩码输出
    multimask_output=True           # 必须开启，帮助在模糊边缘寻找最佳形状
)


# 4. 定义保存函数
def save_anns_with_labels(image, anns, save_path, borders=True):
    if len(anns) == 0:
        # 如果没检测到东西，直接存原图
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    # 按面积从大到小排序
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    ax.set_autoscale_on(False)

    # 准备掩码层
    img_mask = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_mask[:, :, 3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        # 随机颜色掩码
        color_mask = np.concatenate([np.random.random(3), [0.5]]) 
        img_mask[m] = color_mask 
        
        # 绘制边界线
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img_mask, contours, -1, (1, 1, 1, 0.6), thickness=1) 

        # 绘制 Box
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
        # 绘制分数 (predicted_iou)
        score = ann['predicted_iou']
        ax.text(x, y - 5, f"{score:.2f}", color='yellow', fontsize=12, weight='bold', 
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    ax.imshow(img_mask)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # 显式关闭 figure 释放内存

def save_anns_with_cv2(image, anns, save_path):
    """
    使用 OpenCV 保存结果，确保像素尺寸与原图完全一致
    """
    if len(anns) == 0:
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return

    # 1. 拷贝原图，避免直接修改原始数据
    # 注意：如果 image 是 RGB，cv2 需要 BGR
    res_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # 2. 准备一个半透明遮罩层
    mask_overlay = res_img.copy()
    
    for ann in anns:
        m = ann['segmentation'] # 这是一个布尔矩阵
        
        # 随机颜色 (B, G, R)
        color = np.random.randint(0, 255, (3,)).tolist()
        
        # 填充遮罩颜色
        mask_overlay[m] = color
        
        # 绘制边界线 (白色)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1)

        # 绘制 Box (黄色)
        x, y, w, h = [int(v) for v in ann['bbox']]
        cv2.rectangle(res_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # 绘制分数 (predicted_iou)
        score = ann['predicted_iou']
        label = f"{score:.2f}"
        # 在方框上方写字
        cv2.putText(res_img, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 3. 将遮罩与原图融合 (0.5 是透明度)
    cv2.addWeighted(mask_overlay, 0.5, res_img, 0.5, 0, res_img)
    
    # 4. 直接保存，像素绝对不会变
    cv2.imwrite(save_path, res_img)

# 5. 循环处理图片
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

print(f"Found {len(image_paths)} images. Starting processing...")

for path in image_paths:
    img_name = os.path.basename(path)
    print(f"Processing: {img_name}")

    try:
        # 加载并转换图片
        raw_image = Image.open(path).convert("RGB")
        image_np = np.array(raw_image)

        # 执行推理
        with torch.no_grad(): # 推理模式，不计算梯度
            masks = mask_generator.generate(image_np)

        # 保存图片
        img_h, img_w = image_np.shape[:2]
        total_area = img_h * img_w

        # 设置一个过滤阈值，例如：过滤掉占总面积 80% 以上的框
        max_area_ratio = 0.8 

        # 在调用绘图或保存前进行过滤
        filtered_masks = [
            m for m in masks 
            if m['area'] < (total_area * max_area_ratio)
        ]

        print(f"过滤前: {len(masks)} 个, 过滤后: {len(filtered_masks)} 个")

# 使用过滤后的 filtered_masks 进行显示
        save_path = os.path.join(output_dir, f"{img_name}")
        save_anns_with_labels(image_np,filtered_masks, save_path)

        # --- 清理显存和内存 ---
        del masks
        del image_np
        del raw_image

        # NPU 显存清理建议
        if device.type == "npu":
            # 强制清空 NPU 缓存池
            torch.npu.empty_cache() 
        
        elif device.type == "cuda":
            torch.cuda.empty_cache() # 清理 GPU 显存池
        elif device.type == "mps":
            # MPS 暂不支持直接 empty_cache，主要依靠 gc
            pass
        
        gc.collect() # 清理 Python 内存垃圾

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print("Done! All images processed.")
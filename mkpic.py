import matplotlib.pyplot as plt
import os

# 1. 准备数据
# data = [["China", "81.18", "58.33", "54.39"]]
# column_labels = ["Category", "AUROC-px (%)", "F1-px (%)", "AP-px (%)"]
data = [["0.7965", "0.5274", "0.5609"]]
column_labels = ["mRecall", "mPrecision", "F1-Score"]
# 2. 创建画布
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

# 3. 创建表格
table = ax.table(
    cellText=data, 
    colLabels=column_labels, 
    loc='center', 
    cellLoc='center'
)

# 4. 美化字体和缩放
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 2.5)

# 5. 核心：修正后的三线表逻辑 (使用 T, B, L, R 简写)
for (row, col), cell in table.get_celld().items():
    # 默认隐藏所有边框
    cell.set_linewidth(0)
    
    # 标题行 (Row 0)
    if row == 0:
        cell.set_text_props(weight='bold') # 标题加粗
        # 标题行需要上边框(T)和下边框(B)
        cell.visible_edges = 'TB' 
        cell.set_linewidth(2) # 设置粗线
        
    # 数据行 (Row 1 及以后)
    elif row == len(data): # 最后一行数据的底部
        cell.visible_edges = 'B'
        cell.set_linewidth(2) # 底部粗线
    
    # 注意：中间的数据行如果需要区分，可以不设置，保持空白

# 6. 保存图片 (DPI设为300，达到打印级别清晰度)
save_path = "academic_table.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"表格已成功保存至: {os.path.abspath(save_path)}")
plt.show()
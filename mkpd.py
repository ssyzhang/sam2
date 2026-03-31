import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 加载你生成的 CSV 文件
csv_path = "e:/cvlab/data0312/PF/final_output/evaluation_metrics.csv"
df = pd.read_csv(csv_path)

# 2. 为了演示和美观，我们选取前 6 个样本进行横向展示
# 如果你想展示全部，建议循环切片处理
df_sample = df.head(6).copy()

# 3. 执行转置 (Transpose)
# 将 FileName 设为索引，然后转置，这样指标就变成了行名
df_horizontal = df_sample.set_index('FileName').T

# 4. 格式化数值：保留 4 位小数
df_horizontal = df_horizontal.map(lambda x: f"{x:.4f}")

# 5. 绘图：生成学术风格图片
fig, ax = plt.subplots(figsize=(12, 4)) # 横向比例
ax.axis('tight')
ax.axis('off')

# 创建表格
# df_horizontal.columns 是文件名，df_horizontal.index 是指标名
table = ax.table(
    cellText=df_horizontal.values,
    rowLabels=df_horizontal.index,   # 左侧显示：IoU, Recall...
    colLabels=df_horizontal.columns, # 上方显示：PF201, PF202...
    loc='center',
    cellLoc='center'
)

# 6. 美化：三线表风格
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.0, 2.5) # 调整单元格高度

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0) # 先隐藏所有线
    
    # 顶线 (标题行上方)
    if row == 0:
        cell.visible_edges = 'T'
        cell.set_linewidth(2)
    
    # 标题下线 (标题行下方)
    if row == 0:
        # 注意：OpenCV/Matplotlib 逻辑中，这里比较特殊
        # 我们给第一行数据也加个上边框
        pass
    
    # 给所有第一行数据单元格加一个顶线作为标题下线
    if row == 1:
        cell.visible_edges = 'T'
        cell.set_linewidth(1.5)

    # 底线 (最后一行下方)
    if row == len(df_horizontal.index):
        cell.visible_edges = 'B'
        cell.set_linewidth(2)
        
    # 左侧标签列的样式
    if col == -1:
        cell.set_text_props(weight='bold')

# 7. 保存
save_path = "e:/cvlab/data0312/PF/final_output/horizontal_metrics_table.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"横向表格已保存至: {save_path}")
plt.show()
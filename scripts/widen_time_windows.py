"""
进一步放宽订单时间窗口

当前时间窗口已经是原始的1.5倍（45-90分钟）
进一步放宽到原始的2.0倍（60-120分钟）
这需要将当前值再乘以 2.0/1.5 = 1.333
"""

import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = ['uniform_grid_100.csv', 'uniform_grid_300.csv', 
         'uniform_grid_500.csv', 'uniform_grid_1000.csv']

# 目标：从1.5倍放宽到2.0倍
# 当前已是1.5倍，需要再乘以 2.0/1.5 = 1.333
multiplier = 2.0 / 1.5

print("=" * 70)
print("放宽订单时间窗口：从1.5倍提升到2.0倍")
print("=" * 70)

for f in files:
    path = os.path.join(project_root, 'data', 'orders', f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # 当前时间窗口
        old_min = df['delivery_window'].min() / 60
        old_max = df['delivery_window'].max() / 60
        
        # 放宽时间窗口
        df['delivery_window'] = df['delivery_window'] * multiplier
        
        # 重新计算相关字段
        df['deadline'] = df['arrival_time'] + df['delivery_window']
        df['latest_delivery_time'] = df['deadline']
        
        # 新时间窗口
        new_min = df['delivery_window'].min() / 60
        new_max = df['delivery_window'].max() / 60
        
        # 保存
        df.to_csv(path, index=False)
        
        print(f"\n{f}:")
        print(f"  时间窗口: {old_min:.1f}-{old_max:.1f}min → {new_min:.1f}-{new_max:.1f}min")

print("\n" + "=" * 70)
print("完成！所有订单文件的时间窗口已放宽到原始的2.0倍")
print("=" * 70)

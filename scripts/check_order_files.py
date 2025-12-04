"""检查所有订单文件的时间分布"""
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = ['uniform_grid_100.csv', 'uniform_grid_300.csv', 'uniform_grid_500.csv', 'uniform_grid_1000.csv']

print("=" * 70)
print("订单文件时间分布检查")
print("=" * 70)

for f in files:
    path = os.path.join(project_root, 'data', 'orders', f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        min_t = df['arrival_time'].min()
        max_t = df['arrival_time'].max()
        span = (max_t - min_t) / 3600
        dw_min = df['delivery_window'].min()
        dw_max = df['delivery_window'].max()
        print(f"\n{f}:")
        print(f"  订单数: {len(df)}")
        print(f"  到达时间: {min_t:.0f}s - {max_t:.0f}s (跨度: {span:.1f}小时)")
        print(f"  时间窗口: {dw_min/60:.1f}min - {dw_max/60:.1f}min")
        
        # 检查是否需要修复
        if min_t > 1000 or max_t > 43200:
            print(f"  ⚠️ 需要修复: 到达时间超出仿真范围(0-43200s)")
        else:
            print(f"  ✅ 时间分布正常")
    else:
        print(f"\n{f}: 文件不存在")

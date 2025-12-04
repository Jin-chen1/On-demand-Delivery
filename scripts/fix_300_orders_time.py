"""
修复300单订单的时间范围，使其适配12小时仿真
到达时间范围: 0s - 30240s (约8.4小时，占12小时仿真的70%)
"""
import pandas as pd
import numpy as np

# 读取订单
df = pd.read_csv('data/orders/uniform_grid_300.csv')
print(f"原始时间范围: {df['arrival_time'].min():.0f}s - {df['arrival_time'].max():.0f}s")

# 将到达时间平移到从0开始
min_time = df['arrival_time'].min()
df['arrival_time'] = df['arrival_time'] - min_time

# 更新时间相关列
df['deadline'] = df['arrival_time'] + df['delivery_window']
df['earliest_pickup_time'] = df['arrival_time'] + df['preparation_time']
df['latest_delivery_time'] = df['deadline']

# 保存
df.to_csv('data/orders/uniform_grid_300.csv', index=False)

print(f"修复后时间范围: {df['arrival_time'].min():.0f}s - {df['arrival_time'].max():.0f}s")
print(f"时间跨度: {(df['arrival_time'].max() - df['arrival_time'].min())/3600:.1f}小时")
print(f"订单数: {len(df)}")
print("文件已更新: data/orders/uniform_grid_300.csv")

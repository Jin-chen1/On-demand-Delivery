"""
统一修复所有订单文件的时间分布

目标：
1. 将订单到达时间映射到 [0, 36000s]（10小时）
2. 放宽时间窗口（乘以1.5倍）
"""

import pandas as pd
import numpy as np
import os
import shutil

def fix_order_times(input_file: str, output_file: str, backup_suffix: str = "_backup"):
    """修复订单时间分布"""
    
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    # 检查是否需要修复
    min_t = df['arrival_time'].min()
    max_t = df['arrival_time'].max()
    
    # 如果已经在合理范围内，只放宽时间窗口
    if min_t < 1000 and max_t <= 43200:
        print(f"  到达时间已在合理范围内，只放宽时间窗口...")
        # 检查时间窗口是否已经放宽
        if df['delivery_window'].min() < 2500:  # 原始窗口最小约1800s
            time_window_multiplier = 1.5
            df['delivery_window'] = df['delivery_window'] * time_window_multiplier
            df['deadline'] = df['arrival_time'] + df['delivery_window']
            df['latest_delivery_time'] = df['deadline']
            df.to_csv(output_file, index=False)
            print(f"  时间窗口已放宽1.5倍")
        else:
            print(f"  时间窗口已经放宽过，跳过")
        return df
    
    # 创建备份
    backup_file = input_file.replace('.csv', f'{backup_suffix}.csv')
    if not os.path.exists(backup_file):
        shutil.copy(input_file, backup_file)
        print(f"  已创建备份: {os.path.basename(backup_file)}")
    
    print(f"  原始到达时间: {min_t:.0f}s - {max_t:.0f}s")
    
    # 目标时间范围：0 到 36000秒（10小时）
    target_start = 0
    target_end = 36000
    
    # 线性映射
    original_span = max_t - min_t
    target_span = target_end - target_start
    scale_factor = target_span / original_span
    
    df['arrival_time'] = (df['arrival_time'] - min_t) * scale_factor + target_start
    
    # 放宽时间窗口（乘以1.5倍）
    time_window_multiplier = 1.5
    df['delivery_window'] = df['delivery_window'] * time_window_multiplier
    
    # 重新计算相关时间字段
    df['deadline'] = df['arrival_time'] + df['delivery_window']
    df['earliest_pickup_time'] = df['arrival_time'] + df['preparation_time']
    df['latest_delivery_time'] = df['deadline']
    
    print(f"  修复后到达时间: {df['arrival_time'].min():.0f}s - {df['arrival_time'].max():.0f}s")
    print(f"  时间窗口范围: {df['delivery_window'].min()/60:.1f}min - {df['delivery_window'].max()/60:.1f}min")
    
    # 保存
    df.to_csv(output_file, index=False)
    print(f"  已保存: {os.path.basename(output_file)}")
    
    return df


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orders_dir = os.path.join(project_root, "data", "orders")
    
    # 需要检查的文件
    files = ['uniform_grid_100.csv', 'uniform_grid_1000.csv']
    
    print("=" * 70)
    print("修复订单文件时间分布")
    print("=" * 70)
    
    for f in files:
        path = os.path.join(orders_dir, f)
        if os.path.exists(path):
            print(f"\n处理 {f}:")
            fix_order_times(path, path)
        else:
            print(f"\n{f}: 文件不存在，跳过")
    
    print("\n" + "=" * 70)
    print("修复完成！重新检查所有文件:")
    print("=" * 70)
    
    # 重新检查
    all_files = ['uniform_grid_100.csv', 'uniform_grid_300.csv', 
                 'uniform_grid_500.csv', 'uniform_grid_1000.csv']
    
    for f in all_files:
        path = os.path.join(orders_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            min_t = df['arrival_time'].min()
            max_t = df['arrival_time'].max()
            span = (max_t - min_t) / 3600
            dw_min = df['delivery_window'].min() / 60
            dw_max = df['delivery_window'].max() / 60
            status = "✅" if max_t <= 43200 else "⚠️"
            print(f"\n{f}: {status}")
            print(f"  订单数: {len(df)}")
            print(f"  到达时间: {min_t:.0f}s - {max_t:.0f}s (跨度: {span:.1f}h)")
            print(f"  时间窗口: {dw_min:.1f}min - {dw_max:.1f}min")


if __name__ == "__main__":
    main()

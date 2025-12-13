"""
重新生成订单数据 - 使用基于距离的合理配送时间窗口

问题：原订单数据的delivery_window是随机生成的(30-60分钟)，不考虑实际配送距离
结果：远距离订单可能只有30分钟窗口，加上备餐时间后根本无法按时送达

修复策略：
1. delivery_window = 基础时间 + 距离相关时间 + 多单缓冲
2. 确保即使骑手携带2-3单，订单也有合理的完成时间
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 骑手速度配置
COURIER_SPEED_MEAN = 15.0  # km/h
COURIER_SPEED_SLOW = 10.0  # km/h (考虑高峰期/天气)

# 时间窗口参数
BASE_WINDOW_MIN = 1200  # 基础窗口20分钟（取货+交付开销）
MULTI_ORDER_BUFFER = 900  # 多单缓冲15分钟（假设作为第2单）
SAFETY_MARGIN = 300  # 安全余量5分钟

# 使用慢速计算，确保高峰期也可行
SPEED_FOR_CALCULATION = COURIER_SPEED_SLOW  # 10 km/h


def calculate_realistic_delivery_window(distance_km: float, preparation_time: float) -> float:
    """
    计算合理的配送时间窗口
    
    公式：
    delivery_window = travel_time + base_overhead + multi_order_buffer + safety_margin
    
    其中：
    - travel_time: 基于距离和慢速计算的行驶时间
    - base_overhead: 取货+交付固定开销
    - multi_order_buffer: 考虑作为骑手第2单的等待时间
    - safety_margin: 安全余量
    """
    # 行驶时间（使用慢速，秒）
    travel_time = (distance_km / SPEED_FOR_CALCULATION) * 3600
    
    # 总窗口时间
    delivery_window = travel_time + BASE_WINDOW_MIN + MULTI_ORDER_BUFFER + SAFETY_MARGIN
    
    # 确保最小窗口（即使距离很近）
    min_window = 2400  # 最小40分钟
    max_window = 5400  # 最大90分钟
    
    return np.clip(delivery_window, min_window, max_window)


def regenerate_orders(input_path: Path, output_path: Path):
    """重新生成订单数据，使用合理的时间窗口"""
    
    print(f"读取原始订单: {input_path}")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    print(f"订单数量: {original_count}")
    
    # 保存原始统计
    print(f"\n--- 原始数据统计 ---")
    print(f"原始delivery_window范围: {df['delivery_window'].min():.0f} - {df['delivery_window'].max():.0f} 秒")
    print(f"原始delivery_window均值: {df['delivery_window'].mean():.0f} 秒 ({df['delivery_window'].mean()/60:.1f} 分钟)")
    
    # 重新计算delivery_window
    new_windows = []
    for _, row in df.iterrows():
        new_window = calculate_realistic_delivery_window(
            row['distance_km'], 
            row['preparation_time']
        )
        new_windows.append(new_window)
    
    df['delivery_window'] = new_windows
    
    # 重新计算latest_delivery_time
    df['latest_delivery_time'] = df['earliest_pickup_time'] + df['delivery_window']
    
    print(f"\n--- 调整后数据统计 ---")
    print(f"新delivery_window范围: {df['delivery_window'].min():.0f} - {df['delivery_window'].max():.0f} 秒")
    print(f"新delivery_window均值: {df['delivery_window'].mean():.0f} 秒 ({df['delivery_window'].mean()/60:.1f} 分钟)")
    
    # 验证可行性
    # 纯配送时间窗口 = delivery_window（已经不含备餐）
    df['required_time'] = (df['distance_km'] / SPEED_FOR_CALCULATION) * 3600 + 120  # 行驶+取货交付
    df['margin'] = df['delivery_window'] - df['required_time']
    
    feasible_rate = (df['margin'] >= 0).mean() * 100
    print(f"\n单订单可行率（慢速{SPEED_FOR_CALCULATION}km/h）: {feasible_rate:.1f}%")
    
    # 考虑作为第2单
    avg_delivery_time = df['required_time'].mean()
    df['margin_as_2nd'] = df['margin'] - avg_delivery_time
    feasible_as_2nd = (df['margin_as_2nd'] >= 0).mean() * 100
    print(f"作为第2单可行率: {feasible_as_2nd:.1f}%")
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 删除临时列
    df = df.drop(columns=['required_time', 'margin', 'margin_as_2nd'])
    
    df.to_csv(output_path, index=False)
    print(f"\n已保存调整后的订单数据: {output_path}")
    
    return df


def main():
    orders_dir = Path(__file__).parent.parent / 'data' / 'orders'
    
    if not orders_dir.exists():
        print(f"错误：订单目录不存在: {orders_dir}")
        sys.exit(1)
    
    print("="*70)
    print("重新生成订单数据 - 使用基于距离的合理配送时间窗口")
    print("="*70)
    
    print(f"\n参数配置：")
    print(f"  计算用速度: {SPEED_FOR_CALCULATION} km/h (使用慢速确保高峰可行)")
    print(f"  基础开销: {BASE_WINDOW_MIN} 秒 ({BASE_WINDOW_MIN/60:.0f} 分钟)")
    print(f"  多单缓冲: {MULTI_ORDER_BUFFER} 秒 ({MULTI_ORDER_BUFFER/60:.0f} 分钟)")
    print(f"  安全余量: {SAFETY_MARGIN} 秒 ({SAFETY_MARGIN/60:.0f} 分钟)")
    
    # 处理所有订单文件
    files_to_process = [
        'uniform_grid_100.csv',
        'uniform_grid_300.csv',
        'uniform_grid_500.csv',
        'uniform_grid_1000.csv',
    ]
    
    for filename in files_to_process:
        input_path = orders_dir / filename
        if not input_path.exists():
            print(f"\n跳过不存在的文件: {filename}")
            continue
        
        # 备份原文件（如果备份不存在）
        backup_path = orders_dir / filename.replace('.csv', '_original.csv')
        if not backup_path.exists():
            import shutil
            shutil.copy(input_path, backup_path)
            print(f"\n已备份原始文件: {backup_path}")
        
        # 重新生成
        print(f"\n{'='*70}")
        regenerate_orders(input_path, input_path)
    
    print(f"\n{'='*70}")
    print("完成！所有订单文件已更新为合理的时间窗口")
    print("原始文件已备份为 *_original.csv")
    print("="*70)
    
    print("\n建议：重新运行 analyze_order_delivery_time.py 验证调整效果")


if __name__ == '__main__':
    main()

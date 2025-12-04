"""
修复 uniform_grid_500.csv 的订单到达时间分布

问题：
1. 原始订单到达时间范围（28812s-79096s）超出12小时仿真范围
2. 环境会将其压缩到（0s-30240s），导致订单密度过高
3. 时间窗口（30-60分钟）过紧，几乎没有容错空间

修复方案：
1. 将订单到达时间线性映射到 [0, 36000s]（10小时，预留2小时处理尾部订单）
2. 适当放宽时间窗口（乘以1.5倍）
3. 保持其他字段不变
"""

import pandas as pd
import numpy as np
import os

def fix_order_times(input_file: str, output_file: str):
    """
    修复订单时间分布
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    print(f"原始数据统计:")
    print(f"  订单数量: {len(df)}")
    print(f"  到达时间范围: {df['arrival_time'].min():.1f}s - {df['arrival_time'].max():.1f}s")
    print(f"  到达时间跨度: {(df['arrival_time'].max() - df['arrival_time'].min())/3600:.2f}小时")
    print(f"  时间窗口范围: {df['delivery_window'].min():.1f}s - {df['delivery_window'].max():.1f}s")
    
    # 目标时间范围：0 到 36000秒（10小时）
    # 预留2小时给最后一批订单的配送
    target_start = 0
    target_end = 36000  # 10小时
    
    # 获取原始时间范围
    original_min = df['arrival_time'].min()
    original_max = df['arrival_time'].max()
    original_span = original_max - original_min
    
    # 线性映射到目标范围
    target_span = target_end - target_start
    scale_factor = target_span / original_span
    
    # 计算新的到达时间
    df['arrival_time_original'] = df['arrival_time']  # 保存原始值用于调试
    df['arrival_time'] = (df['arrival_time'] - original_min) * scale_factor + target_start
    
    # 放宽时间窗口（乘以1.5倍）
    time_window_multiplier = 1.5
    df['delivery_window_original'] = df['delivery_window']
    df['delivery_window'] = df['delivery_window'] * time_window_multiplier
    
    # 重新计算相关时间字段
    # deadline = arrival_time + delivery_window
    df['deadline'] = df['arrival_time'] + df['delivery_window']
    
    # earliest_pickup_time = arrival_time + preparation_time
    df['earliest_pickup_time'] = df['arrival_time'] + df['preparation_time']
    
    # latest_delivery_time = deadline
    df['latest_delivery_time'] = df['deadline']
    
    print(f"\n修复后数据统计:")
    print(f"  到达时间范围: {df['arrival_time'].min():.1f}s - {df['arrival_time'].max():.1f}s")
    print(f"  到达时间跨度: {(df['arrival_time'].max() - df['arrival_time'].min())/3600:.2f}小时")
    print(f"  时间窗口范围: {df['delivery_window'].min():.1f}s - {df['delivery_window'].max():.1f}s")
    print(f"  时间窗口倍数: {time_window_multiplier}x")
    
    # 计算每小时订单分布
    print(f"\n每小时订单分布:")
    for hour in range(12):
        start_sec = hour * 3600
        end_sec = (hour + 1) * 3600
        count = len(df[(df['arrival_time'] >= start_sec) & (df['arrival_time'] < end_sec)])
        bar = '█' * (count // 5) if count > 0 else ''
        print(f"  {hour:2d}:00 - {hour+1:2d}:00: {count:3d}单 {bar}")
    
    # 删除调试用的临时列
    df = df.drop(columns=['arrival_time_original', 'delivery_window_original'])
    
    # 保存修复后的数据
    df.to_csv(output_file, index=False)
    print(f"\n已保存到: {output_file}")
    
    # 验证数据完整性
    print(f"\n数据完整性检查:")
    print(f"  earliest_pickup_time >= arrival_time: {(df['earliest_pickup_time'] >= df['arrival_time']).all()}")
    print(f"  latest_delivery_time > earliest_pickup_time: {(df['latest_delivery_time'] > df['earliest_pickup_time']).all()}")
    print(f"  deadline > arrival_time: {(df['deadline'] > df['arrival_time']).all()}")
    
    return df


def main():
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 输入输出文件路径
    input_file = os.path.join(project_root, "data", "orders", "uniform_grid_500.csv")
    output_file = os.path.join(project_root, "data", "orders", "uniform_grid_500.csv")
    
    # 备份原始文件
    backup_file = os.path.join(project_root, "data", "orders", "uniform_grid_500_backup.csv")
    
    if os.path.exists(input_file):
        # 创建备份
        import shutil
        if not os.path.exists(backup_file):
            shutil.copy(input_file, backup_file)
            print(f"已创建备份: {backup_file}")
        
        # 修复时间分布
        fix_order_times(input_file, output_file)
    else:
        print(f"错误: 文件不存在 {input_file}")


if __name__ == "__main__":
    main()

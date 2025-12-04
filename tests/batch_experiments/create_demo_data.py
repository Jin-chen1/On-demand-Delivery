"""
创建Day 9演示数据
快速生成模拟的实验结果用于测试Day 9可视化功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def create_demo_experiment_results():
    """创建演示用的实验结果数据"""
    
    # 配置参数
    order_volumes = [500, 1000, 1500]
    courier_counts = [10, 15, 20, 25]
    dispatchers = ['greedy', 'ortools', 'alns']
    seeds = [42, 123]
    
    results = []
    
    # 为每个配置组合生成模拟数据
    for num_orders in order_volumes:
        for num_couriers in courier_counts:
            for dispatcher in dispatchers:
                for seed in seeds:
                    # 计算运力比（订单数/骑手数）
                    capacity_ratio = num_orders / num_couriers
                    
                    # 根据调度器类型和运力比模拟性能指标
                    # 贪心算法在高压力下表现最差
                    if dispatcher == 'greedy':
                        base_timeout = 0.15 + (capacity_ratio / 100) * 2.5
                        base_completion = 0.85 - (capacity_ratio / 100) * 2.0
                        base_service_time = 600 + capacity_ratio * 15
                        solve_time = 0.001
                    
                    # OR-Tools表现中等，但在极高压力下也会崩溃
                    elif dispatcher == 'ortools':
                        base_timeout = 0.08 + (capacity_ratio / 100) * 1.8
                        base_completion = 0.92 - (capacity_ratio / 100) * 1.5
                        base_service_time = 550 + capacity_ratio * 12
                        solve_time = 0.8 + np.random.uniform(-0.2, 0.3)
                    
                    # ALNS最优，即使在高压力下也相对稳定
                    else:  # alns
                        base_timeout = 0.05 + (capacity_ratio / 100) * 1.2
                        base_completion = 0.95 - (capacity_ratio / 100) * 1.0
                        base_service_time = 520 + capacity_ratio * 10
                        solve_time = 1.2 + np.random.uniform(-0.3, 0.5)
                    
                    # 添加随机波动
                    timeout_rate = np.clip(base_timeout + np.random.normal(0, 0.02), 0, 1)
                    completion_rate = np.clip(base_completion + np.random.normal(0, 0.03), 0, 1)
                    service_time = max(300, base_service_time + np.random.normal(0, 50))
                    
                    # 其他指标
                    total_orders = num_orders
                    completed_orders = int(total_orders * completion_rate)
                    timeout_orders = int(total_orders * timeout_rate)
                    
                    # 骑手利用率（高压力下利用率更高）
                    utilization = np.clip(0.5 + (capacity_ratio / 100) * 1.5 + np.random.normal(0, 0.05), 0.3, 1.0)
                    
                    # 构建结果记录
                    result = {
                        'experiment_id': f"exp_{len(results)+1:03d}",
                        'task_id': f"{len(results)+1:03d}_orders{num_orders}_couriers{num_couriers}_{dispatcher}_seed{seed}",
                        'num_orders': num_orders,
                        'num_couriers': num_couriers,
                        'dispatcher_type': dispatcher,
                        'random_seed': seed,
                        
                        # 性能指标
                        'total_orders': total_orders,
                        'completed_orders': completed_orders,
                        'timeout_orders': timeout_orders,
                        'completion_rate': completion_rate,
                        'timeout_rate': timeout_rate,
                        'avg_service_time': service_time,
                        'avg_solve_time': solve_time,
                        'avg_courier_utilization': utilization,
                        
                        # 其他统计
                        'total_distance': num_couriers * 80 * np.random.uniform(0.8, 1.2),
                        'total_duration': 3600,
                        'duration_seconds': 3600,  # 仿真持续时间（秒）
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
    
    return pd.DataFrame(results)


def main():
    """主函数"""
    print("="*80)
    print("创建Day 9演示数据")
    print("="*80)
    print()
    
    # 创建输出目录
    output_dir = Path("outputs/day9_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print()
    
    # 生成数据
    print("生成模拟实验结果...")
    results_df = create_demo_experiment_results()
    
    print(f"[OK] Generated {len(results_df)} experiment records")
    print()
    
    # 显示数据概览
    print("Data Overview:")
    print(f"  - Order volumes: {sorted(results_df['num_orders'].unique().tolist())}")
    print(f"  - Courier counts: {sorted(results_df['num_couriers'].unique().tolist())}")
    print(f"  - Dispatchers: {sorted(results_df['dispatcher_type'].unique().tolist())}")
    print(f"  - Random seeds: {sorted(results_df['random_seed'].unique().tolist())}")
    print()
    
    # 保存CSV
    csv_file = output_dir / "raw_results.csv"
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"[OK] CSV file saved: {csv_file}")
    
    # 保存JSON（摘要）
    summary = {
        'total_experiments': len(results_df),
        'order_volumes': sorted(results_df['num_orders'].unique().tolist()),
        'courier_counts': sorted(results_df['num_couriers'].unique().tolist()),
        'dispatchers': sorted(results_df['dispatcher_type'].unique().tolist()),
        'avg_completion_rate': results_df['completion_rate'].mean(),
        'avg_timeout_rate': results_df['timeout_rate'].mean(),
        'generated_at': datetime.now().isoformat()
    }
    
    json_file = output_dir / "experiment_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] JSON summary saved: {json_file}")
    print()
    
    # 显示统计信息
    print("="*80)
    print("Experiment Results Statistics")
    print("="*80)
    print(f"\nAverage Completion Rate: {summary['avg_completion_rate']*100:.1f}%")
    print(f"Average Timeout Rate: {summary['avg_timeout_rate']*100:.1f}%")
    
    print("\nBy Dispatcher:")
    for dispatcher in sorted(results_df['dispatcher_type'].unique()):
        disp_data = results_df[results_df['dispatcher_type'] == dispatcher]
        print(f"  {dispatcher:8s} - Timeout: {disp_data['timeout_rate'].mean()*100:5.1f}%, "
              f"Completion: {disp_data['completion_rate'].mean()*100:5.1f}%")
    
    print("\n="*80)
    print("Demo Data Creation Complete!")
    print("="*80)
    print(f"\nNow you can run Day 9 analysis:")
    print(f"  python tests\\batch_experiments\\run_day9_analysis.py --input-dir {output_dir}")
    print()


if __name__ == "__main__":
    main()

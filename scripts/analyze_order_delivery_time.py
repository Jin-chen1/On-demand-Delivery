"""
订单配送时间分析脚本
分析订单数据中的配送时间窗口是否合理，是否是导致训练超时的根因

分析内容：
1. 订单的可用配送时间窗口 (latest_delivery_time - earliest_pickup_time)
2. 基于距离和骑手速度估算的最小配送时间
3. 判断订单是否可行（时间窗口是否足够）
4. 统计不可行订单比例
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 骑手速度配置（来自config.yaml）
COURIER_SPEED_MEAN = 15.0  # km/h
COURIER_SPEED_MIN = 10.0   # km/h
COURIER_SPEED_MAX = 20.0   # km/h（假设最大值）

# 高峰期拥堵系数（来自scenarios.yaml）
PEAK_CONGESTION_FACTOR = 0.6  # 高峰期速度降至60%
MEAL_CONGESTION_FACTOR = 0.8  # 用餐高峰速度降至80%

# 额外时间开销（取货、交付等）
PICKUP_TIME_OVERHEAD = 60  # 秒，骑手到店后等待/取货时间
DELIVERY_TIME_OVERHEAD = 60  # 秒，交付给顾客的时间


def analyze_orders(csv_path: Path, verbose: bool = True):
    """分析单个订单文件"""
    df = pd.read_csv(csv_path)
    
    results = {
        'file': csv_path.name,
        'total_orders': len(df),
    }
    
    # 计算可用配送时间窗口（从最早可取货到最晚送达）
    df['available_window'] = df['latest_delivery_time'] - df['earliest_pickup_time']
    
    # 也计算从订单到达到最晚送达的总时间窗口
    df['total_window'] = df['latest_delivery_time'] - df['arrival_time']
    
    # 备餐后的有效配送时间（不含备餐）
    df['delivery_only_window'] = df['available_window'] - df['preparation_time']
    
    # 基于距离估算最小配送时间（不同速度场景）
    # 场景1：正常速度
    df['min_travel_time_normal'] = (df['distance_km'] / COURIER_SPEED_MEAN) * 3600  # 秒
    # 场景2：最慢速度
    df['min_travel_time_slow'] = (df['distance_km'] / COURIER_SPEED_MIN) * 3600
    # 场景3：高峰期最慢
    df['min_travel_time_peak'] = (df['distance_km'] / (COURIER_SPEED_MIN * PEAK_CONGESTION_FACTOR)) * 3600
    
    # 加上取货和交付开销
    overhead = PICKUP_TIME_OVERHEAD + DELIVERY_TIME_OVERHEAD
    df['required_time_normal'] = df['min_travel_time_normal'] + overhead
    df['required_time_slow'] = df['min_travel_time_slow'] + overhead
    df['required_time_peak'] = df['min_travel_time_peak'] + overhead
    
    # 判断订单是否可行（配送时间窗口 >= 所需时间）
    df['feasible_normal'] = df['delivery_only_window'] >= df['required_time_normal']
    df['feasible_slow'] = df['delivery_only_window'] >= df['required_time_slow']
    df['feasible_peak'] = df['delivery_only_window'] >= df['required_time_peak']
    
    # 计算时间裕量（正值表示有余量，负值表示不可行）
    df['margin_normal'] = df['delivery_only_window'] - df['required_time_normal']
    df['margin_slow'] = df['delivery_only_window'] - df['required_time_slow']
    df['margin_peak'] = df['delivery_only_window'] - df['required_time_peak']
    
    # 统计
    results['feasible_rate_normal'] = df['feasible_normal'].mean() * 100
    results['feasible_rate_slow'] = df['feasible_slow'].mean() * 100
    results['feasible_rate_peak'] = df['feasible_peak'].mean() * 100
    
    results['infeasible_count_normal'] = (~df['feasible_normal']).sum()
    results['infeasible_count_slow'] = (~df['feasible_slow']).sum()
    results['infeasible_count_peak'] = (~df['feasible_peak']).sum()
    
    # 时间窗口统计
    results['available_window_min'] = df['available_window'].min()
    results['available_window_max'] = df['available_window'].max()
    results['available_window_mean'] = df['available_window'].mean()
    results['available_window_median'] = df['available_window'].median()
    
    results['delivery_only_window_min'] = df['delivery_only_window'].min()
    results['delivery_only_window_max'] = df['delivery_only_window'].max()
    results['delivery_only_window_mean'] = df['delivery_only_window'].mean()
    
    # 距离统计
    results['distance_min'] = df['distance_km'].min()
    results['distance_max'] = df['distance_km'].max()
    results['distance_mean'] = df['distance_km'].mean()
    
    # 备餐时间统计
    results['prep_time_min'] = df['preparation_time'].min()
    results['prep_time_max'] = df['preparation_time'].max()
    results['prep_time_mean'] = df['preparation_time'].mean()
    
    # 所需配送时间统计（正常速度）
    results['required_time_min'] = df['required_time_normal'].min()
    results['required_time_max'] = df['required_time_normal'].max()
    results['required_time_mean'] = df['required_time_normal'].mean()
    
    # 时间裕量统计
    results['margin_min_normal'] = df['margin_normal'].min()
    results['margin_mean_normal'] = df['margin_normal'].mean()
    results['margin_min_slow'] = df['margin_slow'].min()
    results['margin_min_peak'] = df['margin_peak'].min()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"文件: {csv_path.name}")
        print(f"{'='*70}")
        print(f"订单总数: {results['total_orders']}")
        
        print(f"\n--- 距离统计 ---")
        print(f"  最短距离: {results['distance_min']:.2f} km")
        print(f"  最长距离: {results['distance_max']:.2f} km")
        print(f"  平均距离: {results['distance_mean']:.2f} km")
        
        print(f"\n--- 备餐时间统计 ---")
        print(f"  最短备餐: {results['prep_time_min']:.0f} 秒 ({results['prep_time_min']/60:.1f} 分钟)")
        print(f"  最长备餐: {results['prep_time_max']:.0f} 秒 ({results['prep_time_max']/60:.1f} 分钟)")
        print(f"  平均备餐: {results['prep_time_mean']:.0f} 秒 ({results['prep_time_mean']/60:.1f} 分钟)")
        
        print(f"\n--- 可用配送时间窗口（含备餐）---")
        print(f"  最短窗口: {results['available_window_min']:.0f} 秒 ({results['available_window_min']/60:.1f} 分钟)")
        print(f"  最长窗口: {results['available_window_max']:.0f} 秒 ({results['available_window_max']/60:.1f} 分钟)")
        print(f"  平均窗口: {results['available_window_mean']:.0f} 秒 ({results['available_window_mean']/60:.1f} 分钟)")
        
        print(f"\n--- 纯配送时间窗口（不含备餐）---")
        print(f"  最短窗口: {results['delivery_only_window_min']:.0f} 秒 ({results['delivery_only_window_min']/60:.1f} 分钟)")
        print(f"  最长窗口: {results['delivery_only_window_max']:.0f} 秒 ({results['delivery_only_window_max']/60:.1f} 分钟)")
        print(f"  平均窗口: {results['delivery_only_window_mean']:.0f} 秒 ({results['delivery_only_window_mean']/60:.1f} 分钟)")
        
        print(f"\n--- 所需配送时间估算（正常速度 {COURIER_SPEED_MEAN} km/h）---")
        print(f"  最短所需: {results['required_time_min']:.0f} 秒 ({results['required_time_min']/60:.1f} 分钟)")
        print(f"  最长所需: {results['required_time_max']:.0f} 秒 ({results['required_time_max']/60:.1f} 分钟)")
        print(f"  平均所需: {results['required_time_mean']:.0f} 秒 ({results['required_time_mean']/60:.1f} 分钟)")
        
        print(f"\n--- 订单可行性分析 ---")
        print(f"  正常速度({COURIER_SPEED_MEAN} km/h):")
        print(f"    可行订单: {results['feasible_rate_normal']:.1f}% ({results['total_orders'] - results['infeasible_count_normal']}/{results['total_orders']})")
        print(f"    不可行订单: {results['infeasible_count_normal']} 单")
        print(f"    最小时间裕量: {results['margin_min_normal']:.0f} 秒 ({results['margin_min_normal']/60:.1f} 分钟)")
        
        print(f"  最慢速度({COURIER_SPEED_MIN} km/h):")
        print(f"    可行订单: {results['feasible_rate_slow']:.1f}% ({results['total_orders'] - results['infeasible_count_slow']}/{results['total_orders']})")
        print(f"    不可行订单: {results['infeasible_count_slow']} 单")
        print(f"    最小时间裕量: {results['margin_min_slow']:.0f} 秒 ({results['margin_min_slow']/60:.1f} 分钟)")
        
        print(f"  高峰期最慢({COURIER_SPEED_MIN * PEAK_CONGESTION_FACTOR} km/h):")
        print(f"    可行订单: {results['feasible_rate_peak']:.1f}% ({results['total_orders'] - results['infeasible_count_peak']}/{results['total_orders']})")
        print(f"    不可行订单: {results['infeasible_count_peak']} 单")
        print(f"    最小时间裕量: {results['margin_min_peak']:.0f} 秒 ({results['margin_min_peak']/60:.1f} 分钟)")
    
    # 找出最紧张的订单（时间裕量最小的）
    worst_orders = df.nsmallest(5, 'margin_normal')[
        ['order_id', 'distance_km', 'preparation_time', 'delivery_only_window', 
         'required_time_normal', 'margin_normal']
    ]
    
    if verbose:
        print(f"\n--- 最紧张的5个订单（正常速度）---")
        for _, row in worst_orders.iterrows():
            print(f"  订单{int(row['order_id'])}: "
                  f"距离={row['distance_km']:.2f}km, "
                  f"备餐={row['preparation_time']/60:.1f}min, "
                  f"窗口={row['delivery_only_window']/60:.1f}min, "
                  f"需要={row['required_time_normal']/60:.1f}min, "
                  f"裕量={row['margin_normal']/60:.1f}min")
    
    return results, df


def analyze_multi_order_scenario(df: pd.DataFrame, verbose: bool = True):
    """
    分析多订单场景：骑手携带多单时的超时风险
    
    关键洞察：
    - 单独看每个订单可能都可行
    - 但骑手携带2-3单时，后面的订单可能因为前面订单的配送时间而超时
    """
    if verbose:
        print(f"\n{'='*70}")
        print("多订单场景分析（骑手携带多单）")
        print(f"{'='*70}")
    
    # 假设骑手平均携带2-3单
    avg_orders_per_courier = 2.5
    
    # 平均每单配送时间（含行驶+取货+交付）
    avg_delivery_time = df['required_time_normal'].mean()
    
    # 如果骑手先送其他订单，该订单的等待时间
    extra_wait_for_2nd = avg_delivery_time  # 第2单需要等第1单完成
    extra_wait_for_3rd = avg_delivery_time * 2  # 第3单需要等前2单完成
    
    # 重新计算可行性（考虑等待）
    df['margin_as_2nd'] = df['margin_normal'] - extra_wait_for_2nd
    df['margin_as_3rd'] = df['margin_normal'] - extra_wait_for_3rd
    
    feasible_as_2nd = (df['margin_as_2nd'] >= 0).mean() * 100
    feasible_as_3rd = (df['margin_as_3rd'] >= 0).mean() * 100
    
    if verbose:
        print(f"\n假设：")
        print(f"  - 平均每单配送时间（正常速度）: {avg_delivery_time/60:.1f} 分钟")
        print(f"  - 骑手平均携带订单数: {avg_orders_per_courier}")
        
        print(f"\n多订单可行性：")
        print(f"  作为第1单（无需等待）: {df['feasible_normal'].mean()*100:.1f}% 可行")
        print(f"  作为第2单（等待{avg_delivery_time/60:.1f}分钟）: {feasible_as_2nd:.1f}% 可行")
        print(f"  作为第3单（等待{extra_wait_for_3rd/60:.1f}分钟）: {feasible_as_3rd:.1f}% 可行")
        
        print(f"\n⚠️ 关键发现：")
        print(f"  即使单个订单100%可行，但作为骑手的第2-3单时，")
        print(f"  可行率可能降至 {feasible_as_2nd:.1f}% ~ {feasible_as_3rd:.1f}%")
        print(f"  这可能是训练中大量超时的根本原因！")


def main():
    # 订单文件目录
    orders_dir = Path(__file__).parent.parent / 'data' / 'orders'
    
    if not orders_dir.exists():
        print(f"错误：订单目录不存在: {orders_dir}")
        sys.exit(1)
    
    # 找到所有CSV文件
    csv_files = sorted(orders_dir.glob('uniform_grid_*.csv'))
    
    if not csv_files:
        print(f"错误：未找到订单文件")
        sys.exit(1)
    
    print("="*70)
    print("订单配送时间分析")
    print("="*70)
    print(f"\n骑手速度配置：")
    print(f"  平均速度: {COURIER_SPEED_MEAN} km/h")
    print(f"  最慢速度: {COURIER_SPEED_MIN} km/h")
    print(f"  高峰期最慢: {COURIER_SPEED_MIN * PEAK_CONGESTION_FACTOR} km/h")
    print(f"\n额外时间开销：")
    print(f"  取货开销: {PICKUP_TIME_OVERHEAD} 秒")
    print(f"  交付开销: {DELIVERY_TIME_OVERHEAD} 秒")
    
    all_results = []
    all_dfs = []
    
    for csv_path in csv_files:
        results, df = analyze_orders(csv_path)
        all_results.append(results)
        all_dfs.append(df)
    
    # 合并分析当前使用的500单文件
    main_file = orders_dir / 'uniform_grid_500.csv'
    if main_file.exists():
        _, main_df = analyze_orders(main_file, verbose=False)
        analyze_multi_order_scenario(main_df)
    
    # 汇总表格
    print(f"\n{'='*70}")
    print("汇总对比")
    print(f"{'='*70}")
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            '文件': r['file'],
            '订单数': r['total_orders'],
            '平均距离(km)': f"{r['distance_mean']:.2f}",
            '平均窗口(min)': f"{r['delivery_only_window_mean']/60:.1f}",
            '平均所需(min)': f"{r['required_time_mean']/60:.1f}",
            '正常可行%': f"{r['feasible_rate_normal']:.1f}",
            '慢速可行%': f"{r['feasible_rate_slow']:.1f}",
            '高峰可行%': f"{r['feasible_rate_peak']:.1f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 结论
    print(f"\n{'='*70}")
    print("分析结论")
    print(f"{'='*70}")
    
    # 使用当前训练文件的结果
    current_result = next((r for r in all_results if r['file'] == 'uniform_grid_500.csv'), all_results[0])
    
    print(f"\n1. 单订单可行性：")
    print(f"   - 正常速度下 {current_result['feasible_rate_normal']:.1f}% 订单可行")
    print(f"   - 高峰期仅 {current_result['feasible_rate_peak']:.1f}% 订单可行")
    
    if current_result['feasible_rate_peak'] < 90:
        print(f"\n2. ⚠️ 潜在超时原因：")
        print(f"   - 高峰期有 {100 - current_result['feasible_rate_peak']:.1f}% 订单理论上不可能按时送达")
        print(f"   - 这些订单即使完美调度也会超时")
    
    print(f"\n3. 多订单叠加效应：")
    print(f"   - 骑手携带多单时，后续订单需要等待")
    print(f"   - 作为第2单，可行率会显著下降")
    print(f"   - 这是训练中大量超时的主要原因")
    
    print(f"\n4. 建议：")
    print(f"   - 放宽delivery_window（增加latest_delivery_time）")
    print(f"   - 或减少订单密度（降低每骑手订单数）")
    print(f"   - 或增加骑手数量")
    print(f"   - 或优化订单生成参数，确保时间窗口合理")


if __name__ == '__main__':
    main()

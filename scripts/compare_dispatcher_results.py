"""
调度器结果对比分析脚本

自动读取最新的RL、ALNS、ORTools运行结果，生成对比表格
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def find_latest_result(results_dir: Path, prefix: str) -> Path:
    """查找指定前缀的最新结果目录"""
    matching_dirs = [d for d in results_dir.iterdir() 
                    if d.is_dir() and d.name.startswith(prefix)]
    if not matching_dirs:
        return None
    # 按名称排序（包含时间戳），取最新的
    return sorted(matching_dirs, key=lambda x: x.name)[-1]


def load_result(result_dir: Path) -> dict:
    """加载单个结果目录的数据"""
    if result_dir is None or not result_dir.exists():
        return None
    
    result = {
        'path': str(result_dir),
        'name': result_dir.name
    }
    
    # 加载统计信息
    stats_file = result_dir / 'statistics.json'
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            result['statistics'] = json.load(f)
    
    # 加载订单结果
    orders_file = result_dir / 'orders_result.csv'
    if orders_file.exists():
        df = pd.read_csv(orders_file)
        result['orders_df'] = df
        
        # 计算指标
        total = len(df)
        delivered = (df['status'] == 'delivered').sum()
        timeout = df['is_timeout'].sum() if 'is_timeout' in df.columns else 0
        
        # 计算服务时间（仅已送达订单）
        delivered_df = df[df['status'] == 'delivered'].copy()
        if len(delivered_df) > 0 and 'delivery_complete_time' in df.columns:
            delivered_df['service_time'] = delivered_df['delivery_complete_time'] - delivered_df['arrival_time']
            avg_service_time = delivered_df['service_time'].mean()
        else:
            avg_service_time = 0
        
        result['metrics'] = {
            'total_orders': total,
            'completed_orders': delivered,
            'completion_rate': delivered / total * 100 if total > 0 else 0,
            'timeout_orders': timeout,
            'timeout_rate': timeout / total * 100 if total > 0 else 0,
            'avg_service_time_sec': avg_service_time,
            'avg_service_time_min': avg_service_time / 60
        }
    
    return result


def main():
    # 结果目录
    results_dir = Path('outputs/simulation_results')
    
    # 查找最新结果
    print("=" * 80)
    print("调度器结果对比分析")
    print("=" * 80)
    print()
    
    # 定义要查找的调度器
    dispatchers = {
        'RL': 'day13_rl',
        'ALNS': 'day6_alns',
        'ORTools': 'day4_ortools'
    }
    
    results = {}
    for name, prefix in dispatchers.items():
        result_dir = find_latest_result(results_dir, prefix)
        if result_dir:
            print(f"找到 {name}: {result_dir.name}")
            results[name] = load_result(result_dir)
        else:
            print(f"未找到 {name} 结果 (前缀: {prefix})")
    
    print()
    
    # 生成对比表格
    if not results:
        print("没有找到任何结果！")
        return
    
    # 创建对比数据
    comparison_data = []
    for name, result in results.items():
        if result and 'metrics' in result:
            m = result['metrics']
            comparison_data.append({
                '调度器': name,
                '总订单': m['total_orders'],
                '完成订单': m['completed_orders'],
                '完成率(%)': f"{m['completion_rate']:.1f}",
                '超时订单': m['timeout_orders'],
                '超时率(%)': f"{m['timeout_rate']:.1f}",
                '平均服务时间(秒)': f"{m['avg_service_time_sec']:.1f}",
                '平均服务时间(分钟)': f"{m['avg_service_time_min']:.1f}"
            })
    
    # 创建DataFrame
    df = pd.DataFrame(comparison_data)
    
    # 打印表格
    print("=" * 80)
    print("性能对比表")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    # 保存到CSV
    output_dir = Path('outputs/comparison_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f'dispatcher_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"对比结果已保存到: {csv_path}")
    
    # 生成Markdown表格
    md_path = output_dir / f'dispatcher_comparison_{timestamp}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 调度器性能对比\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 数据来源
        f.write("## 数据来源\n\n")
        for name, result in results.items():
            if result:
                f.write(f"- **{name}**: `{result['name']}`\n")
        f.write("\n")
        
        # 性能对比表
        f.write("## 性能对比\n\n")
        f.write("| 调度器 | 完成率 | 超时率 | 平均服务时间 |\n")
        f.write("|--------|--------|--------|-------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['调度器']} | {row['完成率(%)']}% | {row['超时率(%)']}% | {row['平均服务时间(分钟)']}分钟 |\n")
        f.write("\n")
        
        # 分析
        f.write("## 分析\n\n")
        if 'RL' in results and 'ALNS' in results:
            rl_m = results['RL']['metrics']
            alns_m = results['ALNS']['metrics']
            
            timeout_improvement = (alns_m['timeout_rate'] - rl_m['timeout_rate']) / alns_m['timeout_rate'] * 100
            service_improvement = (alns_m['avg_service_time_sec'] - rl_m['avg_service_time_sec']) / alns_m['avg_service_time_sec'] * 100
            
            f.write(f"### RL vs ALNS\n\n")
            f.write(f"- 超时率改善: **{timeout_improvement:.1f}%**\n")
            f.write(f"- 服务时间改善: **{service_improvement:.1f}%**\n")
    
    print(f"Markdown报告已保存到: {md_path}")
    
    # 返回结果供其他脚本使用
    return df, results


if __name__ == '__main__':
    main()

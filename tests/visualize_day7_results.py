"""
Day 7 可视化脚本：生成求解速度对比图表
重点图表：
1. 求解时间对比箱线图
2. 性能指标雷达图
3. 综合对比柱状图
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_comparison_results(results_file: Path):
    """加载对比结果数据"""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_solve_time_comparison(ortools_data, alns_data, output_dir: Path):
    """
    绘制求解时间对比图（核心图表）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 平均求解时间对比
    ax1 = axes[0]
    dispatchers = ['OR-Tools', 'ALNS']
    solve_times = [
        ortools_data.get('avg_solve_time', 0),
        alns_data.get('avg_solve_time', 0)
    ]
    
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(dispatchers, solve_times, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, val in zip(bars, solve_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Average Solve Time (seconds)', fontsize=12)
    ax1.set_title('Average Solving Time Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(solve_times) * 1.2)
    
    # 子图2: 求解成功率对比
    ax2 = axes[1]
    success_rates = [
        ortools_data.get('solve_success_rate', 0) * 100,
        alns_data.get('solve_success_rate', 0) * 100
    ]
    
    bars = ax2.bar(dispatchers, success_rates, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Solve Success Rate Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_solve_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 图表1已保存: fig1_solve_time_comparison.png")


def plot_performance_comparison(ortools_data, alns_data, output_dir: Path):
    """
    绘制性能指标对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    dispatchers = ['OR-Tools', 'ALNS']
    colors = ['#3498db', '#e74c3c']
    
    # 子图1: 超时率对比
    ax1 = axes[0, 0]
    timeout_rates = [
        ortools_data.get('timeout_rate', 0) * 100,
        alns_data.get('timeout_rate', 0) * 100
    ]
    bars = ax1.bar(dispatchers, timeout_rates, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, timeout_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Timeout Rate (%)', fontsize=11)
    ax1.set_title('Timeout Rate Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图2: 平均服务时长对比
    ax2 = axes[0, 1]
    service_times = [
        ortools_data.get('avg_service_time', 0) / 60,  # 转换为分钟
        alns_data.get('avg_service_time', 0) / 60
    ]
    bars = ax2.bar(dispatchers, service_times, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, service_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}m',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Average Service Time (minutes)', fontsize=11)
    ax2.set_title('Average Service Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图3: 单均行驶距离对比
    ax3 = axes[1, 0]
    distances = [
        ortools_data.get('avg_distance_per_order', 0),
        alns_data.get('avg_distance_per_order', 0)
    ]
    bars = ax3.bar(dispatchers, distances, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, distances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}m',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Avg Distance per Order (meters)', fontsize=11)
    ax3.set_title('Average Distance per Order Comparison', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 子图4: 骑手利用率对比
    ax4 = axes[1, 1]
    utilization = [
        ortools_data.get('avg_courier_utilization', 0) * 100,
        alns_data.get('avg_courier_utilization', 0) * 100
    ]
    bars = ax4.bar(dispatchers, utilization, color=colors, alpha=0.7, edgecolor='black')
    for bar, val in zip(bars, utilization):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Courier Utilization (%)', fontsize=11)
    ax4.set_title('Average Courier Utilization Comparison', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 图表2已保存: fig2_performance_comparison.png")


def plot_comprehensive_comparison(ortools_data, alns_data, output_dir: Path):
    """
    绘制综合对比雷达图
    """
    # 准备数据（标准化到0-100范围）
    categories = ['Solve Speed', 'Service Quality', 'Distance Efficiency', 
                  'Success Rate', 'Courier Utilization']
    
    # 计算指标（越高越好）
    ortools_scores = [
        100 - min(ortools_data.get('avg_solve_time', 0) * 10, 100),  # 求解速度
        100 - ortools_data.get('timeout_rate', 0) * 100,  # 服务质量
        100 - min(ortools_data.get('avg_distance_per_order', 0) / 50, 100),  # 距离效率
        ortools_data.get('solve_success_rate', 0) * 100,  # 成功率
        ortools_data.get('avg_courier_utilization', 0) * 100  # 利用率
    ]
    
    alns_scores = [
        100 - min(alns_data.get('avg_solve_time', 0) * 10, 100),
        100 - alns_data.get('timeout_rate', 0) * 100,
        100 - min(alns_data.get('avg_distance_per_order', 0) / 50, 100),
        alns_data.get('solve_success_rate', 0) * 100,
        alns_data.get('avg_courier_utilization', 0) * 100
    ]
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    ortools_scores += ortools_scores[:1]
    alns_scores += alns_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制
    ax.plot(angles, ortools_scores, 'o-', linewidth=2, label='OR-Tools', color='#3498db')
    ax.fill(angles, ortools_scores, alpha=0.25, color='#3498db')
    
    ax.plot(angles, alns_scores, 'o-', linewidth=2, label='ALNS', color='#e74c3c')
    ax.fill(angles, alns_scores, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('Comprehensive Performance Radar Chart\n(Higher is Better)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 图表3已保存: fig3_radar_comparison.png")


def plot_summary_table(ortools_data, alns_data, output_dir: Path):
    """
    生成结果汇总表格图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    metrics = [
        ('Dispatcher Type', 'OR-Tools', 'ALNS'),
        ('', '', ''),
        ('Solving Performance', '', ''),
        ('  Avg Solve Time (s)', f"{ortools_data.get('avg_solve_time', 0):.3f}", 
         f"{alns_data.get('avg_solve_time', 0):.3f}"),
        ('  Success Rate (%)', f"{ortools_data.get('solve_success_rate', 0)*100:.1f}", 
         f"{alns_data.get('solve_success_rate', 0)*100:.1f}"),
        ('  Dispatch Count', f"{ortools_data.get('dispatch_count', 0)}", 
         f"{alns_data.get('dispatch_count', 0)}"),
        ('', '', ''),
        ('Service Quality', '', ''),
        ('  Completion Rate (%)', f"{ortools_data.get('completion_rate', 0)*100:.1f}", 
         f"{alns_data.get('completion_rate', 0)*100:.1f}"),
        ('  Timeout Rate (%)', f"{ortools_data.get('timeout_rate', 0)*100:.1f}", 
         f"{alns_data.get('timeout_rate', 0)*100:.1f}"),
        ('  Avg Service Time (min)', f"{ortools_data.get('avg_service_time', 0)/60:.1f}", 
         f"{alns_data.get('avg_service_time', 0)/60:.1f}"),
        ('', '', ''),
        ('Operational Efficiency', '', ''),
        ('  Total Distance (km)', f"{ortools_data.get('total_distance', 0)/1000:.2f}", 
         f"{alns_data.get('total_distance', 0)/1000:.2f}"),
        ('  Avg Distance per Order (m)', f"{ortools_data.get('avg_distance_per_order', 0):.0f}", 
         f"{alns_data.get('avg_distance_per_order', 0):.0f}"),
        ('  Courier Utilization (%)', f"{ortools_data.get('avg_courier_utilization', 0)*100:.1f}", 
         f"{alns_data.get('avg_courier_utilization', 0)*100:.1f}")
    ]
    
    table = ax.table(cellText=metrics, loc='center', cellLoc='left',
                     colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置样式
    for i, row in enumerate(metrics):
        if i == 0:  # 标题行
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
        elif row[0] in ['Solving Performance', 'Service Quality', 'Operational Efficiency']:
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold', fontsize=11)
        elif row[0] == '':  # 空行
            for j in range(3):
                cell = table[(i, j)]
                cell.set_facecolor('#ffffff')
        else:
            table[(i, 0)].set_facecolor('#f8f9fa')
            table[(i, 1)].set_facecolor('#e3f2fd')
            table[(i, 2)].set_facecolor('#ffebee')
    
    plt.title('Day 7: OR-Tools vs ALNS - Comprehensive Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'fig4_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 图表4已保存: fig4_summary_table.png")


def main():
    """主函数"""
    print("="*70)
    print("Day 7 可视化脚本：生成求解速度对比图表")
    print("="*70)
    
    # 查找最新的结果文件
    outputs_dir = project_root / "outputs" / "day7_comparison"
    
    if not outputs_dir.exists():
        print(f"错误：结果目录不存在: {outputs_dir}")
        print("请先运行 test_day7_comparison.py 生成对比结果")
        return False
    
    # 找到最新的结果目录
    result_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
    
    if not result_dirs:
        print(f"错误：在 {outputs_dir} 中未找到结果目录")
        return False
    
    latest_dir = result_dirs[0]
    results_file = latest_dir / "comparison_results.json"
    
    if not results_file.exists():
        print(f"错误：结果文件不存在: {results_file}")
        return False
    
    print(f"\n加载结果文件: {results_file}")
    
    # 加载数据
    data = load_comparison_results(results_file)
    ortools_data = data['ortools']
    alns_data = data['alns']
    
    # 创建可视化输出目录
    vis_dir = latest_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    print(f"\n开始生成图表...")
    print("-"*70)
    
    # 生成各类图表
    plot_solve_time_comparison(ortools_data, alns_data, vis_dir)
    plot_performance_comparison(ortools_data, alns_data, vis_dir)
    plot_comprehensive_comparison(ortools_data, alns_data, vis_dir)
    plot_summary_table(ortools_data, alns_data, vis_dir)
    
    print("\n" + "="*70)
    print(f"[OK] 所有图表已生成并保存到: {vis_dir}")
    print("="*70)
    print("\n生成的图表:")
    print("  1. fig1_solve_time_comparison.png - 求解时间对比")
    print("  2. fig2_performance_comparison.png - 性能指标对比")
    print("  3. fig3_radar_comparison.png - 综合对比雷达图")
    print("  4. fig4_summary_table.png - 结果汇总表格")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

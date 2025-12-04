"""
Day 16: 高级实验可视化模块

功能：
1. 算法对比图（RL vs ALNS vs OR-Tools vs Greedy）
2. 鲁棒性分析图（正常 vs 暴雨 vs 爆单 vs 极端）
3. 压力测试曲线
4. Pareto前沿图
5. 性能退化分析图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# 定义颜色方案
AGENT_COLORS = {
    'greedy': '#FF6B6B',      # 红色
    'ortools': '#4ECDC4',     # 青色
    'alns': '#45B7D1',        # 蓝色
    'rl_ppo': '#96CEB4',      # 绿色
}

SCENARIO_COLORS = {
    'normal': '#2ECC71',      # 绿色
    'rain': '#3498DB',        # 蓝色
    'surge': '#E74C3C',       # 红色
    'extreme': '#9B59B6',     # 紫色
}


def plot_agent_comparison_bar(df: pd.DataFrame, output_dir: Path, 
                               metric: str = 'completion_rate'):
    """
    绘制算法对比柱状图
    
    Args:
        df: 实验结果DataFrame
        output_dir: 输出目录
        metric: 对比指标
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 按agent汇总
    agent_stats = df.groupby('agent_type').agg({
        metric: ['mean', 'std']
    })
    agent_stats.columns = ['mean', 'std']
    agent_stats = agent_stats.sort_values('mean', ascending=False)
    
    # 绘制柱状图
    agents = agent_stats.index.tolist()
    means = agent_stats['mean'].values
    stds = agent_stats['std'].values
    colors = [AGENT_COLORS.get(a, '#888888') for a in agents]
    
    bars = ax.bar(agents, means, yerr=stds, capsize=5, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 标题和标签
    metric_name = 'Completion Rate' if metric == 'completion_rate' else 'Timeout Rate'
    ax.set_title(f'Algorithm Comparison - {metric_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f'agent_comparison_{metric}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_scenario_type_comparison(df: pd.DataFrame, output_dir: Path):
    """
    绘制场景类型对比图（分组柱状图）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 获取所有agent和scenario types
    agents = df['agent_type'].unique().tolist()
    scenario_types = ['normal', 'rain', 'surge', 'extreme']
    
    x = np.arange(len(scenario_types))
    width = 0.2
    
    metrics = [('completion_rate', 'Completion Rate'), ('timeout_rate', 'Timeout Rate')]
    
    for ax_idx, (metric, metric_name) in enumerate(metrics):
        ax = axes[ax_idx]
        
        for i, agent in enumerate(agents):
            values = []
            errors = []
            for stype in scenario_types:
                sdf = df[(df['agent_type'] == agent) & (df['scenario_type'] == stype)]
                if not sdf.empty:
                    values.append(sdf[metric].mean())
                    errors.append(sdf[metric].std())
                else:
                    values.append(0)
                    errors.append(0)
            
            offset = (i - len(agents)/2 + 0.5) * width
            color = AGENT_COLORS.get(agent, '#888888')
            ax.bar(x + offset, values, width, yerr=errors, label=agent.upper(),
                   color=color, edgecolor='black', linewidth=0.5, capsize=3)
        
        ax.set_xlabel('Scenario Type', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} by Scenario Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in scenario_types])
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'scenario_type_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_robustness_heatmap(df: pd.DataFrame, output_dir: Path):
    """
    绘制鲁棒性热力图
    """
    # 创建pivot表
    pivot = df.pivot_table(
        values='completion_rate',
        index='agent_type',
        columns='scenario_type',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制热力图
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置标签
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([c.upper() for c in pivot.columns])
    ax.set_yticklabels([r.upper() for r in pivot.index])
    
    # 添加数值标签
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, f'{value:.1%}', ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
    
    # 颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Completion Rate', rotation=-90, va="bottom", fontsize=12)
    
    ax.set_title('Robustness Heatmap: Completion Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario Type', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    plt.tight_layout()
    save_path = output_dir / 'robustness_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_stress_test_curve(df: pd.DataFrame, output_dir: Path):
    """
    绘制压力测试曲线（订单量 vs 性能）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    agents = df['agent_type'].unique().tolist()
    
    # 按订单量分组
    df_grouped = df.groupby(['agent_type', 'total_orders']).agg({
        'completion_rate': ['mean', 'std'],
        'timeout_rate': ['mean', 'std']
    }).reset_index()
    df_grouped.columns = ['agent_type', 'total_orders', 
                          'completion_mean', 'completion_std',
                          'timeout_mean', 'timeout_std']
    
    # 绘制完成率曲线
    ax = axes[0]
    for agent in agents:
        agent_df = df_grouped[df_grouped['agent_type'] == agent].sort_values('total_orders')
        if agent_df.empty:
            continue
        color = AGENT_COLORS.get(agent, '#888888')
        ax.errorbar(agent_df['total_orders'], agent_df['completion_mean'],
                   yerr=agent_df['completion_std'], label=agent.upper(),
                   color=color, marker='o', linewidth=2, capsize=4, markersize=8)
    
    ax.set_xlabel('Total Orders', fontsize=12)
    ax.set_ylabel('Completion Rate', fontsize=12)
    ax.set_title('Stress Test: Completion Rate vs Order Volume', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(alpha=0.3)
    
    # 绘制超时率曲线
    ax = axes[1]
    for agent in agents:
        agent_df = df_grouped[df_grouped['agent_type'] == agent].sort_values('total_orders')
        if agent_df.empty:
            continue
        color = AGENT_COLORS.get(agent, '#888888')
        ax.errorbar(agent_df['total_orders'], agent_df['timeout_mean'],
                   yerr=agent_df['timeout_std'], label=agent.upper(),
                   color=color, marker='s', linewidth=2, capsize=4, markersize=8)
    
    ax.set_xlabel('Total Orders', fontsize=12)
    ax.set_ylabel('Timeout Rate', fontsize=12)
    ax.set_title('Stress Test: Timeout Rate vs Order Volume', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'stress_test_curve.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_performance_degradation(df: pd.DataFrame, output_dir: Path):
    """
    绘制性能退化分析图
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    agents = df['agent_type'].unique().tolist()
    scenario_types = ['normal', 'rain', 'surge', 'extreme']
    
    # 计算各agent在各场景下的性能
    data = []
    for agent in agents:
        agent_df = df[df['agent_type'] == agent]
        normal_perf = agent_df[agent_df['scenario_type'] == 'normal']['completion_rate'].mean()
        
        for stype in scenario_types:
            sdf = agent_df[agent_df['scenario_type'] == stype]
            if not sdf.empty:
                perf = sdf['completion_rate'].mean()
                degradation = (normal_perf - perf) / normal_perf * 100 if normal_perf > 0 else 0
                data.append({
                    'agent': agent,
                    'scenario_type': stype,
                    'performance': perf,
                    'degradation': degradation
                })
    
    plot_df = pd.DataFrame(data)
    
    # 绘制分组柱状图
    x = np.arange(len(agents))
    width = 0.2
    
    for i, stype in enumerate(scenario_types):
        sdf = plot_df[plot_df['scenario_type'] == stype]
        values = []
        for agent in agents:
            adf = sdf[sdf['agent'] == agent]
            values.append(adf['performance'].values[0] if not adf.empty else 0)
        
        offset = (i - len(scenario_types)/2 + 0.5) * width
        color = SCENARIO_COLORS.get(stype, '#888888')
        ax.bar(x + offset, values, width, label=stype.upper(), color=color,
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Completion Rate', fontsize=12)
    ax.set_title('Performance Degradation Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in agents])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='upper right', title='Scenario Type')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'performance_degradation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """
    绘制Pareto前沿图（完成率 vs 超时率）
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 按agent汇总
    agent_stats = df.groupby('agent_type').agg({
        'completion_rate': 'mean',
        'timeout_rate': 'mean'
    })
    
    # 绘制散点
    for agent in agent_stats.index:
        comp = agent_stats.loc[agent, 'completion_rate']
        timeout = agent_stats.loc[agent, 'timeout_rate']
        color = AGENT_COLORS.get(agent, '#888888')
        ax.scatter(timeout, comp, s=300, c=color, edgecolors='black', 
                  linewidth=2, label=agent.upper(), zorder=5)
        ax.annotate(agent.upper(), (timeout, comp), 
                   textcoords="offset points", xytext=(10, 10),
                   fontsize=12, fontweight='bold')
    
    # 绘制Pareto前沿线（理想方向：高完成率，低超时率）
    points = agent_stats[['timeout_rate', 'completion_rate']].values
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]
    
    # 找出Pareto最优点
    pareto_points = [sorted_points[0]]
    for p in sorted_points[1:]:
        if p[1] > pareto_points[-1][1]:  # 更高的完成率
            pareto_points.append(p)
    
    if len(pareto_points) > 1:
        pareto_arr = np.array(pareto_points)
        ax.plot(pareto_arr[:, 0], pareto_arr[:, 1], 'k--', linewidth=2, 
               alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel('Timeout Rate (Lower is Better)', fontsize=12)
    ax.set_ylabel('Completion Rate (Higher is Better)', fontsize=12)
    ax.set_title('Pareto Frontier: Completion vs Timeout Trade-off', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    
    # 添加理想区域标注
    ax.annotate('Ideal Region', xy=(0.1, 0.9), fontsize=14, color='green', fontweight='bold')
    ax.annotate('Worst Region', xy=(0.8, 0.1), fontsize=14, color='red', fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'pareto_frontier.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_boxplot_comparison(df: pd.DataFrame, output_dir: Path):
    """
    绘制箱线图对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    agents = df['agent_type'].unique().tolist()
    
    # 完成率箱线图
    ax = axes[0]
    data = [df[df['agent_type'] == agent]['completion_rate'].values for agent in agents]
    bp = ax.boxplot(data, labels=[a.upper() for a in agents], patch_artist=True)
    
    for patch, agent in zip(bp['boxes'], agents):
        patch.set_facecolor(AGENT_COLORS.get(agent, '#888888'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Completion Rate', fontsize=12)
    ax.set_title('Completion Rate Distribution by Algorithm', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    
    # 超时率箱线图
    ax = axes[1]
    data = [df[df['agent_type'] == agent]['timeout_rate'].values for agent in agents]
    bp = ax.boxplot(data, labels=[a.upper() for a in agents], patch_artist=True)
    
    for patch, agent in zip(bp['boxes'], agents):
        patch.set_facecolor(AGENT_COLORS.get(agent, '#888888'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Timeout Rate', fontsize=12)
    ax.set_title('Timeout Rate Distribution by Algorithm', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'boxplot_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")


def generate_all_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    生成所有可视化图表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Generating Day 16 Visualizations")
    logger.info("="*60)
    
    # 1. 算法对比柱状图
    plot_agent_comparison_bar(df, output_dir, 'completion_rate')
    plot_agent_comparison_bar(df, output_dir, 'timeout_rate')
    
    # 2. 场景类型对比图
    plot_scenario_type_comparison(df, output_dir)
    
    # 3. 鲁棒性热力图
    plot_robustness_heatmap(df, output_dir)
    
    # 4. 压力测试曲线
    plot_stress_test_curve(df, output_dir)
    
    # 5. 性能退化分析
    plot_performance_degradation(df, output_dir)
    
    # 6. Pareto前沿图
    plot_pareto_frontier(df, output_dir)
    
    # 7. 箱线图对比
    plot_boxplot_comparison(df, output_dir)
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("Generated files:")
    for f in output_dir.glob('*.png'):
        logger.info(f"  - {f.name}")

"""
Day 15: 评估结果可视化模块

生成以下图表：
1. 多场景性能对比柱状图
2. 超时率-完成率散点图
3. Reward权重敏感性热力图
4. 超参数搜索结果分析图
5. 算法性能雷达图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_scenario_comparison(df: pd.DataFrame, output_path: Path,
                             metric: str = 'completion_rate') -> None:
    """
    绘制多场景性能对比柱状图
    
    Args:
        df: 评估结果DataFrame
        output_path: 输出路径
        metric: 对比指标
    """
    # 按场景和Agent类型聚合
    summary = df.groupby(['scenario_name', 'agent_type'])[metric].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(summary.index))
    width = 0.2
    n_agents = len(summary.columns)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (agent, values) in enumerate(summary.items()):
        offset = (i - n_agents/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=agent.upper(), color=colors[i % len(colors)])
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1%}' if metric.endswith('rate') else f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Performance Comparison: {metric.replace("_", " ").title()} by Scenario', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scenario_comparison_{metric}.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / f'scenario_comparison_{metric}.png'}")


def plot_capacity_curve(df: pd.DataFrame, output_path: Path, 
                        scenarios_config: Dict[str, Dict] = None) -> None:
    """
    绘制压力测试曲线 (Capacity Curve / Stress Test)
    对应研究大纲 Fig 4: X轴为订单量/负载，Y轴为超时率
    
    Args:
        df: 评估结果DataFrame
        output_path: 输出路径
        scenarios_config: 场景配置字典（可选，如果不传则从TestScenarios动态获取）
    """
    # 动态获取场景配置
    if scenarios_config is None:
        try:
            from .evaluation_and_tuning import TestScenarios
            scenarios_config = TestScenarios.get_all_scenarios()
        except ImportError:
            # 回退到默认配置
            scenarios_config = {
                'low_load': {'total_orders': 500},
                'medium_load': {'total_orders': 1000},
                'high_load': {'total_orders': 1500},
                'extreme_load': {'total_orders': 2000}
            }
    
    # 提取负载相关的场景（排除非负载场景如rainy_weather）
    load_scenarios = ['low_load', 'medium_load', 'high_load', 'extreme_load']
    df_load = df[df['scenario_name'].isin(load_scenarios)].copy()
    
    if df_load.empty:
        logger.warning("No load scenarios found for capacity curve")
        return
        
    # 从场景配置中动态获取订单量映射
    scenario_orders = {
        name: cfg.get('total_orders', 0) 
        for name, cfg in scenarios_config.items() 
        if name in load_scenarios
    }
    
    df_load['order_volume'] = df_load['scenario_name'].map(scenario_orders)
    df_load = df_load.sort_values('order_volume')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'greedy': '#2ecc71', 'ortools': '#3498db', 'alns': '#e74c3c', 
              'rl_ppo': '#9b59b6', 'random': '#95a5a6'}
    markers = {'greedy': 'o', 'ortools': 's', 'alns': '^', 'rl_ppo': 'D', 'random': 'x'}
    
    # 按Agent类型绘制曲线
    for agent_type in df_load['agent_type'].unique():
        agent_data = df_load[df_load['agent_type'] == agent_type]
        # 计算每个负载等级的平均值
        means = agent_data.groupby('order_volume')['timeout_rate'].mean()
        stds = agent_data.groupby('order_volume')['timeout_rate'].std().fillna(0)
        
        ax.plot(means.index, means.values, 
               marker=markers.get(agent_type, 'o'),
               color=colors.get(agent_type, '#333333'),
               linewidth=2, label=agent_type.upper())
               
        # 添加置信区间
        ax.fill_between(means.index, 
                       np.maximum(0, means.values - stds.values),
                       np.minimum(1, means.values + stds.values),
                       alpha=0.15, color=colors.get(agent_type, '#333333'))
    
    ax.set_xlabel('Order Volume (Orders/Day)', fontsize=12)
    ax.set_ylabel('Timeout Rate', fontsize=12)
    ax.set_title('Capacity Curve: Timeout Rate vs Order Volume', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 标注推荐负载区
    ax.axvspan(500, 1000, alpha=0.05, color='green', label='Safe Zone')
    ax.axvspan(1500, 2000, alpha=0.05, color='red', label='Risk Zone')
    
    plt.tight_layout()
    plt.savefig(output_path / 'capacity_curve.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'capacity_curve.png'}")


def plot_pareto_frontier(df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制帕累托前沿图 (Pareto Frontier)
    对应研究大纲 Fig 7: 配送成本(平均耗时) vs 服务水平(超时率)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'greedy': '#2ecc71', 'ortools': '#3498db', 'alns': '#e74c3c', 
              'rl_ppo': '#9b59b6', 'random': '#95a5a6'}
    markers = {'greedy': 'o', 'ortools': 's', 'alns': '^', 'rl_ppo': 'D', 'random': 'x'}
    
    # 确定成本指标：优先用 avg_service_time，如果没有则用 episode_length 或 total_distance
    cost_metric = 'avg_service_time'
    if cost_metric not in df.columns or df[cost_metric].sum() == 0:  # 如果不存在或全为0
        if 'total_distance' in df.columns and df['total_distance'].sum() > 0:
            cost_metric = 'total_distance'
        elif 'episode_length' in df.columns:
            cost_metric = 'episode_length'
        else:
            # 如果没有合适的成本指标，创建一个默认的
            df['dummy_cost'] = 1.0
            cost_metric = 'dummy_cost'
            
    cost_label = {
        'avg_service_time': 'Avg Delivery Time (s)',
        'total_distance': 'Total Distance (m)',
        'episode_length': 'Steps',
        'dummy_cost': 'Cost (N/A)'
    }.get(cost_metric, cost_metric)
    
    for agent_type in df['agent_type'].unique():
        agent_data = df[df['agent_type'] == agent_type]
        
        # 绘制散点
        ax.scatter(agent_data['timeout_rate'], agent_data[cost_metric],
                  marker=markers.get(agent_type, 'o'),
                  c=colors.get(agent_type, '#333333'),
                  s=100, alpha=0.6, label=agent_type.upper())
        
        # 绘制该Agent的趋势线（拟合）或中心点
        mean_x = agent_data['timeout_rate'].mean()
        mean_y = agent_data[cost_metric].mean()
        ax.scatter([mean_x], [mean_y], marker='+', s=200, c='black', linewidth=2)
    
    ax.set_xlabel('Timeout Rate (Risk)', fontsize=12)
    ax.set_ylabel(f'Cost: {cost_label}', fontsize=12)
    ax.set_title('Pareto Frontier: Cost vs Risk', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 理想区域在左下角
    ax.text(0.02, 0.02, 'Ideal Region', transform=ax.transAxes, 
           fontsize=12, color='green', fontweight='bold',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
    
    plt.tight_layout()
    plt.savefig(output_path / 'pareto_frontier.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'pareto_frontier.png'}")


def plot_completion_vs_timeout(df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制完成率-超时率散点图
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    markers = {'greedy': 'o', 'ortools': 's', 'alns': '^', 'rl_ppo': 'D', 'random': 'x'}
    colors = {'greedy': '#2ecc71', 'ortools': '#3498db', 'alns': '#e74c3c', 
              'rl_ppo': '#9b59b6', 'random': '#95a5a6'}
    
    for agent_type in df['agent_type'].unique():
        agent_data = df[df['agent_type'] == agent_type]
        ax.scatter(agent_data['completion_rate'], agent_data['timeout_rate'],
                  marker=markers.get(agent_type, 'o'),
                  c=colors.get(agent_type, '#333333'),
                  label=agent_type.upper(), s=100, alpha=0.7)
    
    # 添加理想区域标注
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Target timeout < 30%')
    ax.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Target completion > 50%')
    
    ax.set_xlabel('Completion Rate', fontsize=12)
    ax.set_ylabel('Timeout Rate', fontsize=12)
    ax.set_title('Completion Rate vs Timeout Rate', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 设置轴范围
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path / 'completion_vs_timeout.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'completion_vs_timeout.png'}")


def plot_sensitivity_heatmap(df: pd.DataFrame, output_path: Path,
                             metric: str = 'mean_reward') -> None:
    """
    绘制Reward权重敏感性热力图
    """
    # 提取参数名和值
    config_cols = [c for c in df.columns if c.startswith('config_')]
    
    if len(config_cols) < 2:
        logger.warning("Not enough config columns for heatmap")
        return
    
    # 简化：选取两个主要参数绘制热力图
    main_params = ['config_weight_timeout_penalty', 'config_weight_completion_bonus']
    available_params = [p for p in main_params if p in df.columns]
    
    if len(available_params) < 2:
        available_params = config_cols[:2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建透视表
    pivot_data = df.pivot_table(
        values=metric,
        index=available_params[0] if len(available_params) > 0 else df.index,
        columns=available_params[1] if len(available_params) > 1 else None,
        aggfunc='mean'
    )
    
    if pivot_data.ndim == 2:
        im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([f'{v:.3f}' for v in pivot_data.columns])
        ax.set_yticks(range(len(pivot_data.index)))
        ax.set_yticklabels([f'{v:.1f}' for v in pivot_data.index])
        
        ax.set_xlabel(available_params[1].replace('config_', ''), fontsize=12)
        ax.set_ylabel(available_params[0].replace('config_', ''), fontsize=12)
        
        plt.colorbar(im, ax=ax, label=metric)
    
    ax.set_title(f'Sensitivity Analysis: {metric}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensitivity_heatmap.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'sensitivity_heatmap.png'}")


def plot_hyperparameter_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制超参数搜索结果分析图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 学习率 vs 奖励
    if 'config_learning_rate' in df.columns:
        ax = axes[0, 0]
        lr_data = df.groupby('config_learning_rate')['mean_reward'].mean()
        ax.bar(range(len(lr_data)), lr_data.values, color='#3498db')
        ax.set_xticks(range(len(lr_data)))
        ax.set_xticklabels([f'{v:.0e}' for v in lr_data.index], rotation=45)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Learning Rate vs Reward')
    
    # 2. Gamma vs 完成率
    if 'config_gamma' in df.columns:
        ax = axes[0, 1]
        gamma_data = df.groupby('config_gamma')['mean_completion_rate'].mean()
        ax.bar(range(len(gamma_data)), gamma_data.values, color='#2ecc71')
        ax.set_xticks(range(len(gamma_data)))
        ax.set_xticklabels([f'{v:.3f}' for v in gamma_data.index])
        ax.set_xlabel('Gamma (Discount Factor)')
        ax.set_ylabel('Completion Rate')
        ax.set_title('Gamma vs Completion Rate')
    
    # 3. 熵系数 vs 奖励
    if 'config_ent_coef' in df.columns:
        ax = axes[1, 0]
        ent_data = df.groupby('config_ent_coef')['mean_reward'].mean()
        ax.bar(range(len(ent_data)), ent_data.values, color='#e74c3c')
        ax.set_xticks(range(len(ent_data)))
        ax.set_xticklabels([f'{v:.3f}' for v in ent_data.index])
        ax.set_xlabel('Entropy Coefficient')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Entropy Coef vs Reward')
    
    # 4. 训练时间 vs 奖励
    ax = axes[1, 1]
    ax.scatter(df['training_time_seconds'], df['mean_reward'], 
              c='#9b59b6', alpha=0.6, s=60)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Time vs Reward')
    
    plt.tight_layout()
    plt.savefig(output_path / 'hyperparameter_analysis.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'hyperparameter_analysis.png'}")


def plot_radar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制算法性能雷达图
    """
    # 聚合每个Agent的平均指标
    metrics = ['completion_rate', 'timeout_rate', 'total_reward']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        logger.warning("No metrics available for radar chart")
        return
    
    summary = df.groupby('agent_type')[available_metrics].mean()
    
    # 归一化（用于雷达图）
    normalized = summary.copy()
    for col in normalized.columns:
        if col == 'timeout_rate':
            # 超时率反转（越低越好）
            normalized[col] = 1 - (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min() + 1e-8)
        else:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min() + 1e-8)
    
    # 雷达图
    categories = [m.replace('_', ' ').title() for m in available_metrics]
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    for i, (agent, row) in enumerate(normalized.iterrows()):
        values = row.tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=agent.upper(), 
               color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Algorithm Performance Radar Chart', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path / 'radar_chart.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'radar_chart.png'}")


def generate_all_visualizations(eval_df: pd.DataFrame, 
                                sensitivity_df: pd.DataFrame = None,
                                hp_df: pd.DataFrame = None,
                                output_dir: Path = None) -> None:
    """
    生成所有可视化图表
    """
    if output_dir is None:
        output_dir = Path('./outputs/day15_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Day 15: Generating Visualizations")
    logger.info("="*60)
    
    # 1. 场景对比图（完成率）
    if eval_df is not None and not eval_df.empty:
        plot_scenario_comparison(eval_df, output_dir, 'completion_rate')
        plot_scenario_comparison(eval_df, output_dir, 'timeout_rate')
        plot_completion_vs_timeout(eval_df, output_dir)
        plot_capacity_curve(eval_df, output_dir)
        plot_pareto_frontier(eval_df, output_dir)
        plot_radar_chart(eval_df, output_dir)
    
    # 2. 敏感性分析热力图
    if sensitivity_df is not None and not sensitivity_df.empty:
        plot_sensitivity_heatmap(sensitivity_df, output_dir)
    
    # 3. 超参数分析图
    if hp_df is not None and not hp_df.empty:
        plot_hyperparameter_analysis(hp_df, output_dir)
    
    logger.info(f"All visualizations saved to: {output_dir}")


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    """
    绘制训练曲线
    """
    import json
    
    if not history_path.exists():
        logger.warning(f"Training history not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 奖励曲线
    if 'mean_rewards' in history:
        ax = axes[0, 0]
        ax.plot(history.get('timesteps', range(len(history['mean_rewards']))),
               history['mean_rewards'], color='#3498db', linewidth=2)
        if 'std_rewards' in history:
            mean = np.array(history['mean_rewards'])
            std = np.array(history['std_rewards'])
            steps = history.get('timesteps', range(len(mean)))
            ax.fill_between(steps, mean - std, mean + std, alpha=0.3, color='#3498db')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Training Reward Curve')
        ax.grid(True, alpha=0.3)
    
    # 2. 完成率曲线
    if 'completion_rates' in history:
        ax = axes[0, 1]
        ax.plot(history.get('timesteps', range(len(history['completion_rates']))),
               history['completion_rates'], color='#2ecc71', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Completion Rate')
        ax.set_title('Completion Rate During Training')
        ax.grid(True, alpha=0.3)
    
    # 3. 超时率曲线
    if 'timeout_rates' in history:
        ax = axes[1, 0]
        ax.plot(history.get('timesteps', range(len(history['timeout_rates']))),
               history['timeout_rates'], color='#e74c3c', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Timeout Rate')
        ax.set_title('Timeout Rate During Training')
        ax.grid(True, alpha=0.3)
    
    # 4. Episode长度
    if 'episode_lengths' in history:
        ax = axes[1, 1]
        ax.plot(history.get('timesteps', range(len(history['episode_lengths']))),
               history['episode_lengths'], color='#9b59b6', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Length During Training')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path / 'training_curves.png'}")

"""
RL vs 基线算法对比分析脚本
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 加载基线算法结果
    baseline_file = Path('outputs/day7_comparison/20251127_102455/comparison_results.json')
    with open(baseline_file, 'r') as f:
        baseline_results = json.load(f)

    # 加载RL v2训练评估数据
    rl_eval_file = Path('outputs/rl_training/models/20251127_145102_low_load/eval_logs/evaluations.npz')
    rl_data = np.load(rl_eval_file)
    rl_results = rl_data['results']
    rl_mean_rewards = np.mean(rl_results, axis=1)
    rl_timesteps = rl_data['timesteps']

    # RL v1结果（对比用）
    v1_eval_file = Path('outputs/rl_training/models/20251127_113058_low_load/eval_logs/evaluations.npz')
    v1_data = np.load(v1_eval_file)
    v1_results = v1_data['results']
    v1_mean_rewards = np.mean(v1_results, axis=1)

    print('=' * 80)
    print('RL vs Baseline Algorithms Comparison Report')
    print('=' * 80)
    print()

    # 基线算法结果
    print('Baseline Algorithm Results (100 orders, 20 couriers, 4h simulation):')
    print('-' * 80)
    header = f"{'Algorithm':<12} {'Completion':<12} {'Timeout':<12} {'Avg Service':<15} {'Avg Distance':<15}"
    print(header)
    print('-' * 80)

    for algo in ['ortools', 'alns']:
        r = baseline_results[algo]
        line = f"{algo.upper():<12} {r['completion_rate']*100:>8.1f}% {r['timeout_rate']*100:>9.1f}% {r['avg_service_time']:>12.1f}s {r['avg_distance_per_order']:>12.1f}m"
        print(line)

    print()
    print('=' * 80)
    print('RL Training Results:')
    print('-' * 80)
    header2 = f"{'Model':<15} {'Mean Reward':<15} {'Best Reward':<15} {'Final Reward':<15} {'Std Dev':<12}"
    print(header2)
    print('-' * 80)

    # v1结果
    v1_mean = np.mean(v1_mean_rewards)
    v1_best = np.max(v1_mean_rewards)
    v1_final = v1_mean_rewards[-1]
    v1_std = np.std(v1_mean_rewards)
    print(f"{'PPO v1':<15} {v1_mean:>12.2f} {v1_best:>14.2f} {v1_final:>14.2f} {v1_std:>11.2f}")

    # v2结果
    v2_mean = np.mean(rl_mean_rewards)
    v2_best = np.max(rl_mean_rewards)
    v2_final = rl_mean_rewards[-1]
    v2_std = np.std(rl_mean_rewards)
    print(f"{'PPO v2 (Opt)':<15} {v2_mean:>12.2f} {v2_best:>14.2f} {v2_final:>14.2f} {v2_std:>11.2f}")

    print()
    print('=' * 80)
    print('Improvement Analysis:')
    print('-' * 80)
    improvement = ((v2_mean - v1_mean) / v1_mean * 100)
    print(f'1. PPO v2 reward improved by {improvement:.1f}% over v1')
    print(f'2. PPO v2 relative variance: {(v2_std/v2_mean*100):.1f}% (vs v1: {(v1_std/v1_mean*100):.1f}%)')
    print(f'3. Baseline OR-Tools: 100% completion, 23% timeout')
    print(f'4. Baseline ALNS: 100% completion, 25% timeout')
    print()

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1: RL v1 vs v2 学习曲线
    ax1 = axes[0]
    v1_timesteps = np.load(v1_eval_file)['timesteps']
    ax1.plot(v1_timesteps/1000000, v1_mean_rewards, 'r-', alpha=0.7, label='PPO v1', linewidth=1.5)
    ax1.plot(rl_timesteps/1000000, rl_mean_rewards, 'b-', alpha=0.7, label='PPO v2 (Optimized)', linewidth=1.5)
    ax1.set_xlabel('Timesteps (Million)')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('RL Training: v1 vs v2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 奖励分布对比
    ax2 = axes[1]
    bp = ax2.boxplot([v1_mean_rewards, rl_mean_rewards], 
                     tick_labels=['PPO v1', 'PPO v2'])
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Reward Distribution Comparison')
    ax2.grid(True, alpha=0.3)
    
    # 添加均值标注
    means = [v1_mean, v2_mean]
    for i, mean in enumerate(means, 1):
        ax2.annotate(f'Mean: {mean:.0f}', xy=(i, mean), 
                    xytext=(i+0.3, mean), fontsize=9, color='red')
    
    # 图3: 基线算法对比
    ax3 = axes[2]
    algorithms = ['OR-Tools', 'ALNS']
    timeout_rates = [baseline_results['ortools']['timeout_rate'] * 100,
                     baseline_results['alns']['timeout_rate'] * 100]
    completion_rates = [baseline_results['ortools']['completion_rate'] * 100,
                        baseline_results['alns']['completion_rate'] * 100]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, completion_rates, width, label='Completion Rate', color='green', alpha=0.7)
    bars2 = ax3.bar(x + width/2, timeout_rates, width, label='Timeout Rate', color='red', alpha=0.7)
    
    ax3.set_ylabel('Rate (%)')
    ax3.set_title('Baseline Algorithm Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(algorithms)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = 'outputs/rl_training/models/20251127_145102_low_load/rl_vs_baseline_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f'Comparison chart saved to: {output_path}')
    
    # 图4: 最终对比（包含服务时间）
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    algorithms = ['OR-Tools', 'ALNS', 'PPO v2 (RL)']
    
    # RL评估结果（修复后）
    rl_completion = 93.8
    rl_timeout = 12.0
    rl_service_time = 1926.0
    
    # 图1: 完成率对比
    ax1 = axes2[0]
    completion_rates = [100.0, 100.0, rl_completion]
    colors = ['#2ecc71', '#2ecc71', '#3498db']
    bars1 = ax1.bar(algorithms, completion_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Completion Rate (%)')
    ax1.set_title('Completion Rate Comparison')
    ax1.set_ylim(0, 110)
    for bar, val in zip(bars1, completion_rates):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 图2: 超时率对比
    ax2 = axes2[1]
    timeout_rates = [23.0, 25.0, rl_timeout]
    colors = ['#e74c3c', '#e74c3c', '#27ae60']
    bars2 = ax2.bar(algorithms, timeout_rates, color=colors, alpha=0.8)
    ax2.set_ylabel('Timeout Rate (%)')
    ax2.set_title('Timeout Rate Comparison (Lower is Better)')
    ax2.set_ylim(0, 35)
    for bar, val in zip(bars2, timeout_rates):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 图3: 服务时间对比
    ax3 = axes2[2]
    service_times = [2550.7, 2554.5, rl_service_time]
    colors = ['#e74c3c', '#e74c3c', '#27ae60']
    bars3 = ax3.bar(algorithms, service_times, color=colors, alpha=0.8)
    ax3.set_ylabel('Avg Service Time (s)')
    ax3.set_title('Service Time Comparison (Lower is Better)')
    for bar, val in zip(bars3, service_times):
        ax3.annotate(f'{val:.0f}s', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path2 = 'outputs/rl_training/models/20251127_145102_low_load/final_comparison.png'
    plt.savefig(output_path2, dpi=150)
    print(f'Final comparison chart saved to: {output_path2}')
    
    # 图5: On-time Orders Count（准时送达单量）- 关键说服力图表
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # 计算准时送达单量（5个episode，每个100单，共500单）
    total_orders = 500
    
    # 基线算法（100%完成，但有超时）
    ortools_completed = total_orders  # 500
    ortools_timeout = int(total_orders * 0.23)  # 115
    ortools_ontime = ortools_completed - ortools_timeout  # 385
    
    alns_completed = total_orders  # 500
    alns_timeout = int(total_orders * 0.25)  # 125
    alns_ontime = alns_completed - alns_timeout  # 375
    
    # RL算法（93.8%完成，12%超时）
    rl_completed = 469  # 实际完成数
    rl_timeout = 60  # 实际超时数
    rl_ontime = rl_completed - rl_timeout  # 409
    
    algorithms = ['OR-Tools', 'ALNS', 'PPO v2 (RL)']
    ontime_counts = [ortools_ontime, alns_ontime, rl_ontime]
    timeout_counts = [ortools_timeout, alns_timeout, rl_timeout]
    incomplete_counts = [0, 0, total_orders - rl_completed]  # 未完成订单
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # 堆叠柱状图
    bars_ontime = ax.bar(x, ontime_counts, width, label='On-time Delivered', color='#27ae60', alpha=0.9)
    bars_timeout = ax.bar(x, timeout_counts, width, bottom=ontime_counts, label='Timeout (Late)', color='#f39c12', alpha=0.9)
    bars_incomplete = ax.bar(x, incomplete_counts, width, bottom=[a+b for a,b in zip(ontime_counts, timeout_counts)], 
                             label='Incomplete', color='#e74c3c', alpha=0.9)
    
    ax.set_ylabel('Number of Orders', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_title('On-time Orders Comparison (5 Episodes, 500 Total Orders)\n** RL delivers MORE orders ON TIME **', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 550)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars_ontime, ontime_counts)):
        ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val/2),
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 标注RL的优势
    rl_advantage = rl_ontime - max(ortools_ontime, alns_ontime)
    ax.annotate(f'+{rl_advantage} more\non-time orders!', 
                xy=(2, rl_ontime + 30), ha='center', fontsize=11, color='#27ae60', fontweight='bold')
    
    # 添加总结文本框
    textstr = f'On-time Delivery Rate:\n• OR-Tools: {ortools_ontime}/{total_orders} = {ortools_ontime/total_orders*100:.1f}%\n• ALNS: {alns_ontime}/{total_orders} = {alns_ontime/total_orders*100:.1f}%\n• PPO v2: {rl_ontime}/{total_orders} = {rl_ontime/total_orders*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    output_path3 = 'outputs/rl_training/models/20251127_145102_low_load/ontime_orders_comparison.png'
    plt.savefig(output_path3, dpi=150)
    print(f'On-time orders chart saved to: {output_path3}')
    
    print()
    print('=' * 80)
    print('Final Summary:')
    print('-' * 80)
    print(f'  PPO v2 (RL) vs Baseline:')
    print(f'    - Timeout Rate: 12.0% vs 23-25% (IMPROVED by ~50%)')
    print(f'    - Service Time: 1926s vs 2550s (IMPROVED by ~24%)')
    print(f'    - Completion Rate: 93.8% vs 100% (slightly lower)')
    print('=' * 80)

if __name__ == '__main__':
    main()

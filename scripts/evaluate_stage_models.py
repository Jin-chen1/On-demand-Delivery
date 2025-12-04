"""
评估各阶段RL模型的性能

对训练过程中保存的各阶段模型进行独立评估，
获取完成率和超时率等关键指标。
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from src.rl.rl_environment import DeliveryRLEnvironment


def load_config(config_path: str = "config/rl_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_stage_config(full_config: dict, stage_name: str):
    """获取指定阶段的配置"""
    # curriculum在 rl -> training -> curriculum -> curriculum_stages
    stages = full_config['rl']['training']['curriculum']['curriculum_stages']
    for stage in stages:
        if stage['name'] == stage_name:
            return stage
    raise ValueError(f"未找到阶段: {stage_name}")


def create_eval_env(full_config: dict, stage_config: dict):
    """创建评估环境"""
    # 基础仿真配置（从simulation部分获取）
    base_sim_config = full_config.get('simulation', {})
    
    # 构建阶段特定配置
    sim_config = {
        **base_sim_config,  # 继承基础配置
        'orders_file': stage_config['orders_file'],
        'num_couriers': stage_config['num_couriers'],
        'simulation_duration': stage_config['simulation_duration'],
        'total_orders': stage_config['total_orders'],
        'enable_auto_dispatch': False,  # RL模式禁用自动调度
    }
    
    # RL配置
    rl_config = {
        'state_encoder': full_config['rl'].get('state_encoder', {}),
        'reward_calculator': full_config['rl'].get('reward_calculator', {}),
        'action_mode': full_config['rl'].get('action_mode', 'discrete'),
    }
    
    return DeliveryRLEnvironment(sim_config, rl_config)


def evaluate_model(model_path: str, env, n_episodes: int = 5):
    """评估单个模型"""
    print(f"  加载模型: {model_path}")
    model = PPO.load(model_path)
    
    results = {
        'episode_rewards': [],
        'completion_rates': [],
        'timeout_rates': [],
        'episode_lengths': []
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # 收集统计
        stats = env.get_episode_statistics()
        results['episode_rewards'].append(episode_reward)
        results['completion_rates'].append(stats.get('completion_rate', 0))
        results['timeout_rates'].append(stats.get('timeout_rate', 0))
        results['episode_lengths'].append(steps)
        
        print(f"    Episode {ep + 1}: 完成率={stats.get('completion_rate', 0):.1%}, "
              f"超时率={stats.get('timeout_rate', 0):.1%}, 奖励={episode_reward:.2f}")
    
    # 汇总
    summary = {
        'mean_reward': float(np.mean(results['episode_rewards'])),
        'std_reward': float(np.std(results['episode_rewards'])),
        'mean_completion_rate': float(np.mean(results['completion_rates'])),
        'std_completion_rate': float(np.std(results['completion_rates'])),
        'mean_timeout_rate': float(np.mean(results['timeout_rates'])),
        'std_timeout_rate': float(np.std(results['timeout_rates'])),
    }
    
    return summary


def main():
    """主函数"""
    # 模型目录
    model_dir = project_root / "outputs" / "rl_training" / "models" / "20251201_134403_low_load"
    
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        return
    
    # 加载配置
    config_path = project_root / "config" / "rl_config.yaml"
    full_config = load_config(config_path)
    
    # 阶段定义
    stages = [
        ('stage_1_warmup', 'warmup'),
        ('stage_2_transition', 'transition'),
        ('stage_3_low_load', 'low_load'),
        ('stage_4_medium_load', 'medium_load'),
        ('stage_5_high_load', 'high_load'),
        ('stage_6_extreme_load', 'extreme_load'),
    ]
    
    # 评估结果
    all_results = {}
    
    print("=" * 70)
    print("各阶段模型评估")
    print("=" * 70)
    
    for model_file, stage_name in stages:
        model_path = model_dir / f"{model_file}.zip"
        
        if not model_path.exists():
            print(f"\n跳过 {stage_name}: 模型文件不存在")
            continue
        
        print(f"\n评估阶段: {stage_name}")
        print("-" * 50)
        
        try:
            # 获取阶段配置
            stage_config = get_stage_config(full_config, stage_name)
            print(f"  订单数: {stage_config['total_orders']}, 骑手数: {stage_config['num_couriers']}")
            
            # 创建环境
            env = create_eval_env(full_config, stage_config)
            
            # 评估模型
            summary = evaluate_model(str(model_path), env, n_episodes=3)
            
            all_results[stage_name] = summary
            
            print(f"  平均完成率: {summary['mean_completion_rate']:.1%} ± {summary['std_completion_rate']:.1%}")
            print(f"  平均超时率: {summary['mean_timeout_rate']:.1%} ± {summary['std_timeout_rate']:.1%}")
            print(f"  平均奖励: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
            
            env.close()
            
        except Exception as e:
            print(f"  评估失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出汇总表格
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    print(f"{'阶段':<15} {'完成率':<15} {'超时率':<15} {'平均奖励':<15}")
    print("-" * 70)
    
    for stage_name in ['warmup', 'transition', 'low_load', 'medium_load', 'high_load', 'extreme_load']:
        if stage_name in all_results:
            r = all_results[stage_name]
            print(f"{stage_name:<15} "
                  f"{r['mean_completion_rate']*100:>5.1f}% ± {r['std_completion_rate']*100:>4.1f}%  "
                  f"{r['mean_timeout_rate']*100:>5.1f}% ± {r['std_timeout_rate']*100:>4.1f}%  "
                  f"{r['mean_reward']:>8.2f}")
    
    # 保存结果
    output_path = model_dir / "stage_evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()

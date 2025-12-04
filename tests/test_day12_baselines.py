"""
Day 12测试：Baseline Wrappers
在RL环境中封装Greedy、OR-Tools、ALNS作为基线Agent，
确保它们能在相同的Gym接口下运行，收集Benchmark数据

测试内容：
1. 各Agent初始化测试
2. predict()接口正确性验证
3. 单Agent Episode运行测试
4. 多Agent Benchmark对比
5. 结果保存和输出
"""

import sys
from pathlib import Path
import logging
import time
import json
import csv
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

# 仿真环境配置
SIMULATION_CONFIG = {
    'data_dir': 'data/processed',
    'orders_file': 'data/orders/orders.csv',
    'total_orders': 100,
    'num_couriers': 10,
    'simulation_duration': 7200,
    'dispatch_interval': 30,
    'dispatcher_type': 'greedy',  # 默认调度器（BaselineAgent会覆盖）
    'dispatcher_config': {},
    'courier_config': {
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 5}
    }
}

# RL配置
RL_CONFIG = {
    'state_encoder': {
        'max_pending_orders': 20,
        'max_couriers': 15,
        'grid_size': 5
    },
    'reward_calculator': {
        'reward_type': 'dense',
        'weight_timeout_penalty': 10.0,
        'weight_distance_cost': 0.001
    },
    'action_mode': 'discrete'
}

# Benchmark配置
BENCHMARK_CONFIG = {
    'agent_types': ['random', 'greedy', 'ortools', 'alns'],
    'num_episodes': 2,
    'max_steps': 200,
    'output_dir': 'outputs/day12_benchmark'
}


# ============================================================
# 测试函数
# ============================================================

def test_agent_initialization():
    """测试1：Agent初始化"""
    print("\n" + "="*60)
    print("测试1：Agent初始化")
    print("="*60)
    
    from src.rl.rl_environment import DeliveryRLEnvironment
    from src.rl.baseline_agents import (
        RandomAgent, 
        BaselineAgent, 
        create_baseline_agent
    )
    
    # 创建环境
    env = DeliveryRLEnvironment(SIMULATION_CONFIG, RL_CONFIG)
    
    results = {}
    
    # 测试RandomAgent
    try:
        agent = RandomAgent(env.action_space)
        assert agent.agent_type == 'random'
        results['random'] = '✓ 初始化成功'
        print(f"  RandomAgent: {results['random']}")
    except Exception as e:
        results['random'] = f'✗ 失败: {e}'
        print(f"  RandomAgent: {results['random']}")
    
    # 测试各种BaselineAgent
    for agent_type in ['greedy', 'ortools', 'alns']:
        try:
            agent = BaselineAgent(agent_type, env)
            assert agent.agent_type == agent_type
            results[agent_type] = '✓ 初始化成功'
            print(f"  {agent_type.capitalize()}Agent: {results[agent_type]}")
        except Exception as e:
            results[agent_type] = f'✗ 失败: {e}'
            print(f"  {agent_type.capitalize()}Agent: {results[agent_type]}")
    
    # 测试工厂函数
    try:
        for agent_type in ['random', 'greedy', 'ortools', 'alns']:
            agent = create_baseline_agent(agent_type, env)
            assert agent is not None
        results['factory'] = '✓ 工厂函数正常'
        print(f"  工厂函数: {results['factory']}")
    except Exception as e:
        results['factory'] = f'✗ 失败: {e}'
        print(f"  工厂函数: {results['factory']}")
    
    env.close()
    return results


def test_predict_interface():
    """测试2：predict()接口正确性"""
    print("\n" + "="*60)
    print("测试2：predict()接口正确性")
    print("="*60)
    
    from src.rl.rl_environment import DeliveryRLEnvironment
    from src.rl.baseline_agents import create_baseline_agent
    
    env = DeliveryRLEnvironment(SIMULATION_CONFIG, RL_CONFIG)
    obs, info = env.reset(seed=42)
    
    results = {}
    
    for agent_type in ['random', 'greedy', 'ortools', 'alns']:
        try:
            agent = create_baseline_agent(agent_type, env)
            
            # 调用predict
            action, state = agent.predict(obs, deterministic=True)
            
            # 验证返回值
            assert action is not None, "action不应为None"
            
            # 对于离散动作空间，验证动作在有效范围内
            if hasattr(env.action_space, 'n'):
                assert 0 <= action < env.action_space.n, f"动作超出范围: {action}"
            
            results[agent_type] = f'✓ action={action}'
            print(f"  {agent_type.capitalize()}: {results[agent_type]}")
            
        except Exception as e:
            results[agent_type] = f'✗ 失败: {e}'
            print(f"  {agent_type.capitalize()}: {results[agent_type]}")
    
    env.close()
    return results


def test_single_episode():
    """测试3：单Agent Episode运行"""
    print("\n" + "="*60)
    print("测试3：单Agent Episode运行")
    print("="*60)
    
    from src.rl.rl_environment import DeliveryRLEnvironment
    from src.rl.baseline_agents import create_baseline_agent, run_baseline_episode
    
    results = {}
    
    # 只测试greedy（最快）
    agent_type = 'greedy'
    
    try:
        env = DeliveryRLEnvironment(SIMULATION_CONFIG, RL_CONFIG)
        agent = create_baseline_agent(agent_type, env)
        
        print(f"  运行 {agent_type} Episode...")
        start_time = time.time()
        
        episode_result = run_baseline_episode(
            agent, env, 
            max_steps=100, 
            verbose=False
        )
        
        elapsed = time.time() - start_time
        
        print(f"  完成! 耗时: {elapsed:.2f}秒")
        print(f"  - 步数: {episode_result['step_count']}")
        print(f"  - 总奖励: {episode_result['total_reward']:.2f}")
        print(f"  - 完成率: {episode_result['completion_rate']:.1%}")
        print(f"  - 超时率: {episode_result['timeout_rate']:.1%}")
        
        results[agent_type] = episode_result
        env.close()
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        results[agent_type] = {'error': str(e)}
    
    return results


def test_multi_agent_benchmark():
    """测试4：多Agent Benchmark对比"""
    print("\n" + "="*60)
    print("测试4：多Agent Benchmark对比")
    print("="*60)
    
    from src.rl.baseline_agents import benchmark_agents
    
    # 使用较小的配置进行快速测试
    test_sim_config = SIMULATION_CONFIG.copy()
    test_sim_config['total_orders'] = 50
    test_sim_config['simulation_duration'] = 3600
    
    test_rl_config = RL_CONFIG.copy()
    
    try:
        results = benchmark_agents(
            env_config=test_sim_config,
            rl_config=test_rl_config,
            agent_types=['random', 'greedy'],  # 快速测试只用两个
            num_episodes=1,
            max_steps=100,
            verbose=False
        )
        
        print(f"\n  Benchmark完成，共 {len(results)} 条记录")
        
        # 汇总结果
        print("\n  汇总:")
        for r in results:
            print(f"    {r['agent_type']}: "
                  f"完成率={r['completion_rate']:.1%}, "
                  f"超时率={r['timeout_rate']:.1%}, "
                  f"奖励={r['total_reward']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_full_benchmark():
    """测试5：完整Benchmark运行并保存结果"""
    print("\n" + "="*60)
    print("测试5：完整Benchmark运行")
    print("="*60)
    
    from src.rl.baseline_agents import benchmark_agents
    
    # 创建输出目录
    output_dir = Path(BENCHMARK_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  输出目录: {output_dir}")
    print(f"  Agent类型: {BENCHMARK_CONFIG['agent_types']}")
    print(f"  Episode数: {BENCHMARK_CONFIG['num_episodes']}")
    print(f"  最大步数: {BENCHMARK_CONFIG['max_steps']}")
    
    start_time = time.time()
    
    try:
        results = benchmark_agents(
            env_config=SIMULATION_CONFIG,
            rl_config=RL_CONFIG,
            agent_types=BENCHMARK_CONFIG['agent_types'],
            num_episodes=BENCHMARK_CONFIG['num_episodes'],
            max_steps=BENCHMARK_CONFIG['max_steps'],
            verbose=True
        )
        
        total_time = time.time() - start_time
        print(f"\n  Benchmark完成，总耗时: {total_time:.1f}秒")
        
        # 保存结果到CSV
        csv_path = output_dir / 'benchmark_results.csv'
        save_results_to_csv(results, csv_path)
        print(f"  结果已保存到: {csv_path}")
        
        # 保存汇总到JSON
        summary = create_summary(results)
        json_path = output_dir / 'benchmark_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  汇总已保存到: {json_path}")
        
        # 打印汇总表
        print_summary_table(summary)
        
        return results
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_results_to_csv(results: list, filepath: Path):
    """保存结果到CSV文件"""
    if not results:
        return
    
    # 获取所有可能的字段
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # 排序字段
    priority_keys = ['agent_type', 'episode', 'total_reward', 'step_count',
                    'completion_rate', 'timeout_rate', 'total_orders',
                    'completed_orders', 'timeout_orders']
    sorted_keys = [k for k in priority_keys if k in all_keys]
    sorted_keys += sorted(k for k in all_keys if k not in priority_keys)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, '') for k in sorted_keys})


def create_summary(results: list) -> dict:
    """创建结果汇总"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'simulation': SIMULATION_CONFIG,
            'rl': RL_CONFIG,
            'benchmark': BENCHMARK_CONFIG
        },
        'agents': {}
    }
    
    # 按agent类型分组
    agent_results = {}
    for r in results:
        agent_type = r['agent_type']
        if agent_type not in agent_results:
            agent_results[agent_type] = []
        agent_results[agent_type].append(r)
    
    # 计算每个agent的统计
    for agent_type, runs in agent_results.items():
        rewards = [r['total_reward'] for r in runs]
        completion_rates = [r['completion_rate'] for r in runs]
        timeout_rates = [r['timeout_rate'] for r in runs]
        step_times = [r.get('avg_step_time_ms', 0) for r in runs]
        
        summary['agents'][agent_type] = {
            'num_episodes': len(runs),
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards))
            },
            'completion_rate': {
                'mean': float(np.mean(completion_rates)),
                'std': float(np.std(completion_rates))
            },
            'timeout_rate': {
                'mean': float(np.mean(timeout_rates)),
                'std': float(np.std(timeout_rates))
            },
            'avg_step_time_ms': {
                'mean': float(np.mean(step_times)),
                'std': float(np.std(step_times))
            }
        }
    
    return summary


def print_summary_table(summary: dict):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("Benchmark 结果汇总")
    print("="*80)
    
    # 表头
    print(f"{'Agent':<12} {'Reward':<20} {'Completion':<15} {'Timeout':<15} {'Time(ms)':<12}")
    print("-"*80)
    
    # 按完成率排序
    agents = summary.get('agents', {})
    sorted_agents = sorted(
        agents.items(), 
        key=lambda x: x[1]['completion_rate']['mean'], 
        reverse=True
    )
    
    for agent_type, stats in sorted_agents:
        reward = f"{stats['reward']['mean']:.2f}±{stats['reward']['std']:.2f}"
        completion = f"{stats['completion_rate']['mean']:.1%}±{stats['completion_rate']['std']:.1%}"
        timeout = f"{stats['timeout_rate']['mean']:.1%}±{stats['timeout_rate']['std']:.1%}"
        time_ms = f"{stats['avg_step_time_ms']['mean']:.1f}"
        
        print(f"{agent_type:<12} {reward:<20} {completion:<15} {timeout:<15} {time_ms:<12}")
    
    print("="*80)


# ============================================================
# 主函数
# ============================================================

def main():
    """主测试函数"""
    print("\n" + "#"*60)
    print("# Day 12: Baseline Wrappers 测试")
    print("#"*60)
    
    all_passed = True
    
    # 测试1：Agent初始化
    try:
        test_agent_initialization()
    except Exception as e:
        print(f"测试1失败: {e}")
        all_passed = False
    
    # 测试2：predict接口
    try:
        test_predict_interface()
    except Exception as e:
        print(f"测试2失败: {e}")
        all_passed = False
    
    # 测试3：单Episode运行
    try:
        test_single_episode()
    except Exception as e:
        print(f"测试3失败: {e}")
        all_passed = False
    
    # 测试4：多Agent对比（快速）
    try:
        test_multi_agent_benchmark()
    except Exception as e:
        print(f"测试4失败: {e}")
        all_passed = False
    
    # 测试5：完整Benchmark
    print("\n" + "-"*60)
    print("是否运行完整Benchmark? (可能需要几分钟)")
    print("按Enter运行，输入'skip'跳过: ", end='')
    
    try:
        user_input = input().strip().lower()
        if user_input != 'skip':
            run_full_benchmark()
    except EOFError:
        # 非交互模式，运行完整benchmark
        run_full_benchmark()
    
    # 总结
    print("\n" + "#"*60)
    if all_passed:
        print("# 所有测试通过! Day 12 完成!")
    else:
        print("# 部分测试失败，请检查错误信息")
    print("#"*60)
    
    return all_passed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Day 12 Baseline Wrappers测试')
    parser.add_argument('--full', action='store_true', help='运行完整benchmark')
    parser.add_argument('--skip-full', action='store_true', help='跳过完整benchmark')
    args = parser.parse_args()
    
    # 如果指定了--skip-full，修改main函数行为
    if args.skip_full:
        # 只运行基础测试
        print("\n" + "#"*60)
        print("# Day 12: Baseline Wrappers 测试 (快速模式)")
        print("#"*60)
        test_agent_initialization()
        test_predict_interface()
        test_single_episode()
        test_multi_agent_benchmark()
        print("\n# 基础测试通过! 使用 --full 运行完整benchmark")
    elif args.full:
        # 运行完整benchmark
        main()
        # 自动运行完整benchmark，不等待输入
    else:
        main()

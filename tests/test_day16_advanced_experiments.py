"""
Day 16: 高级实验测试脚本 (Advanced Experiments)

测试功能：
1. 最终对比实验（RL vs ALNS vs OR-Tools vs Greedy）
2. 多场景鲁棒性测试（暴雨、爆单）
3. 可视化生成
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_scenario_definitions():
    """测试场景定义"""
    print("\n" + "="*70)
    print("Day 16 Test 1: Scenario Definitions")
    print("="*70)
    
    from src.rl.advanced_experiments import RobustnessScenarios
    
    # 获取所有场景
    all_scenarios = RobustnessScenarios.get_all_scenarios()
    print(f"\nTotal scenarios: {len(all_scenarios)}")
    
    # 按类型统计
    type_counts = {}
    for name, cfg in all_scenarios.items():
        stype = cfg.get('type', 'unknown')
        type_counts[stype] = type_counts.get(stype, 0) + 1
        print(f"  - {name}: {cfg.get('description', '')}")
    
    print(f"\nScenarios by type:")
    for stype, count in type_counts.items():
        print(f"  - {stype}: {count}")
    
    # 验证关键字段
    required_fields = ['name', 'type', 'total_orders', 'num_couriers', 'speed_multiplier']
    all_valid = True
    for name, cfg in all_scenarios.items():
        for field in required_fields:
            if field not in cfg:
                print(f"  WARNING: {name} missing field '{field}'")
                all_valid = False
    
    # 获取快速场景
    quick_scenarios = RobustnessScenarios.get_quick_scenarios()
    print(f"\nQuick scenarios: {len(quick_scenarios)}")
    for name, cfg in quick_scenarios.items():
        print(f"  - {name}: {cfg['total_orders']} orders")
    
    if all_valid:
        print("\n✓ Scenario definitions test PASSED")
        return True
    else:
        print("\n✗ Scenario definitions test FAILED")
        return False


def test_experimenter_initialization():
    """测试实验管理器初始化"""
    print("\n" + "="*70)
    print("Day 16 Test 2: Experimenter Initialization")
    print("="*70)
    
    from src.rl.advanced_experiments import AdvancedExperimenter
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    try:
        # 快速模式初始化
        experimenter = AdvancedExperimenter(
            config_path=str(config_path),
            output_dir=str(PROJECT_ROOT / 'outputs' / 'day16_test'),
            quick_mode=True
        )
        
        print(f"\nExperimenter initialized:")
        print(f"  - Output dir: {experimenter.output_dir}")
        print(f"  - Scenarios: {len(experimenter.scenarios)}")
        print(f"  - Available agents: {experimenter.available_agents}")
        print(f"  - Quick mode: {experimenter.quick_mode}")
        
        print("\n✓ Experimenter initialization test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Experimenter initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_experiment():
    """测试快速实验运行"""
    print("\n" + "="*70)
    print("Day 16 Test 3: Quick Experiment Run")
    print("="*70)
    
    from src.rl.advanced_experiments import AdvancedExperimenter
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    output_dir = PROJECT_ROOT / 'outputs' / 'day16_quick_test'
    
    try:
        experimenter = AdvancedExperimenter(
            config_path=str(config_path),
            output_dir=str(output_dir),
            quick_mode=True
        )
        
        # 只测试greedy在一个场景下运行1次
        print("\nRunning quick test (greedy, normal_low, 1 run)...")
        df = experimenter.run_comparison_experiment(
            agent_types=['greedy'],
            scenario_names=['normal_low'],
            n_runs=1
        )
        
        print(f"\nResults shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        if not df.empty:
            print(f"\nResult summary:")
            print(f"  - Completion rate: {df['completion_rate'].iloc[0]:.1%}")
            print(f"  - Timeout rate: {df['timeout_rate'].iloc[0]:.1%}")
            print(f"  - Total orders: {df['total_orders'].iloc[0]}")
            print(f"  - Completed: {df['completed_orders'].iloc[0]}")
            
            # 检查输出文件
            print(f"\nOutput files:")
            for f in output_dir.iterdir():
                print(f"  - {f.name}")
            
            print("\n✓ Quick experiment test PASSED")
            return True
        else:
            print("\n✗ Quick experiment test FAILED: Empty results")
            return False
            
    except Exception as e:
        print(f"\n✗ Quick experiment test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_agent_comparison():
    """测试多算法对比"""
    print("\n" + "="*70)
    print("Day 16 Test 4: Multi-Agent Comparison")
    print("="*70)
    
    from src.rl.advanced_experiments import AdvancedExperimenter
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    output_dir = PROJECT_ROOT / 'outputs' / 'day16_multi_agent_test'
    
    try:
        experimenter = AdvancedExperimenter(
            config_path=str(config_path),
            output_dir=str(output_dir),
            quick_mode=True
        )
        
        # 测试多算法对比（greedy和ortools）
        print("\nRunning multi-agent comparison (greedy, ortools)...")
        print("  Scenarios: normal_low, rain_low")
        print("  Runs: 2")
        
        df = experimenter.run_comparison_experiment(
            agent_types=['greedy', 'ortools'],
            scenario_names=['normal_low', 'rain_low'],
            n_runs=2
        )
        
        print(f"\nResults shape: {df.shape}")
        
        # 按agent汇总
        print("\nPerformance by Agent:")
        agent_summary = df.groupby('agent_type').agg({
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std']
        }).round(4)
        print(agent_summary)
        
        # 按场景汇总
        print("\nPerformance by Scenario:")
        scenario_summary = df.groupby('scenario_name').agg({
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std']
        }).round(4)
        print(scenario_summary)
        
        print("\n✓ Multi-agent comparison test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Multi-agent comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robustness_scenarios():
    """测试鲁棒性场景"""
    print("\n" + "="*70)
    print("Day 16 Test 5: Robustness Scenarios (Rain & Surge)")
    print("="*70)
    
    from src.rl.advanced_experiments import AdvancedExperimenter
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    output_dir = PROJECT_ROOT / 'outputs' / 'day16_robustness_test'
    
    try:
        experimenter = AdvancedExperimenter(
            config_path=str(config_path),
            output_dir=str(output_dir),
            quick_mode=True
        )
        
        # 测试暴雨和爆单场景
        print("\nRunning robustness test scenarios...")
        print("  Scenarios: normal_medium, rain_medium, surge_50")
        print("  Agent: greedy")
        print("  Runs: 2")
        
        df = experimenter.run_comparison_experiment(
            agent_types=['greedy'],
            scenario_names=['normal_medium', 'rain_medium', 'surge_50'],
            n_runs=2
        )
        
        print(f"\nResults shape: {df.shape}")
        
        # 按场景类型分析
        print("\nPerformance by Scenario Type:")
        for stype in ['normal', 'rain', 'surge']:
            type_df = df[df['scenario_type'] == stype]
            if not type_df.empty:
                avg_comp = type_df['completion_rate'].mean()
                avg_timeout = type_df['timeout_rate'].mean()
                print(f"  {stype}: completion={avg_comp:.1%}, timeout={avg_timeout:.1%}")
        
        # 计算性能退化
        normal_perf = df[df['scenario_type'] == 'normal']['completion_rate'].mean()
        rain_perf = df[df['scenario_type'] == 'rain']['completion_rate'].mean()
        surge_perf = df[df['scenario_type'] == 'surge']['completion_rate'].mean()
        
        if normal_perf > 0:
            rain_degradation = (normal_perf - rain_perf) / normal_perf * 100
            surge_degradation = (normal_perf - surge_perf) / normal_perf * 100
            print(f"\nPerformance Degradation:")
            print(f"  Rain: {rain_degradation:.1f}%")
            print(f"  Surge: {surge_degradation:.1f}%")
        
        print("\n✓ Robustness scenarios test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Robustness scenarios test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_day16_experiment():
    """运行完整Day 16实验"""
    print("\n" + "="*70)
    print("Day 16: FULL ADVANCED EXPERIMENT")
    print("="*70)
    
    from src.rl.advanced_experiments import AdvancedExperimenter
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / 'outputs' / f'day16_full_{timestamp}'
    
    try:
        experimenter = AdvancedExperimenter(
            config_path=str(config_path),
            output_dir=str(output_dir),
            quick_mode=False  # 完整模式
        )
        
        print(f"\nExperiment Configuration:")
        print(f"  - Output: {output_dir}")
        print(f"  - Scenarios: {len(experimenter.scenarios)}")
        print(f"  - Agents: {experimenter.available_agents}")
        
        # 选择关键场景进行测试
        key_scenarios = [
            'normal_low', 'normal_medium', 'normal_high',
            'rain_low', 'rain_medium', 'rain_high',
            'surge_50', 'surge_100',
            'extreme_rain_surge'
        ]
        
        print(f"\nRunning experiments on {len(key_scenarios)} key scenarios...")
        print(f"Agents: {experimenter.available_agents}")
        print(f"Runs per config: 3")
        
        df = experimenter.run_comparison_experiment(
            agent_types=['greedy', 'ortools', 'alns'],
            scenario_names=key_scenarios,
            n_runs=3
        )
        
        # 打印汇总
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        
        # 按Agent汇总
        print("\n## Performance by Agent (All Scenarios)")
        agent_summary = df.groupby('agent_type').agg({
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std'],
            'avg_service_time': 'mean'
        }).round(4)
        print(agent_summary)
        
        # 按场景类型汇总
        print("\n## Performance by Scenario Type")
        for stype in ['normal', 'rain', 'surge', 'extreme']:
            type_df = df[df['scenario_type'] == stype]
            if type_df.empty:
                continue
            print(f"\n### {stype.upper()}")
            type_agent_summary = type_df.groupby('agent_type').agg({
                'completion_rate': ['mean', 'std'],
                'timeout_rate': ['mean', 'std']
            }).round(4)
            print(type_agent_summary)
        
        # 找出最佳算法
        print("\n## Best Agent by Scenario")
        for scenario in df['scenario_name'].unique():
            sdata = df[df['scenario_name'] == scenario].groupby('agent_type')['completion_rate'].mean()
            if not sdata.empty:
                best = sdata.idxmax()
                print(f"  {scenario}: {best} ({sdata.max():.1%})")
        
        # 鲁棒性分析
        print("\n## Robustness Analysis (Performance Degradation)")
        for agent_type in df['agent_type'].unique():
            agent_df = df[df['agent_type'] == agent_type]
            normal_perf = agent_df[agent_df['scenario_type'] == 'normal']['completion_rate'].mean()
            stress_perf = agent_df[agent_df['scenario_type'].isin(['rain', 'surge', 'extreme'])]['completion_rate'].mean()
            if normal_perf > 0:
                degradation = (normal_perf - stress_perf) / normal_perf * 100
                print(f"  {agent_type}: {degradation:.1f}% degradation under stress")
        
        print(f"\n\nResults saved to: {output_dir}")
        print("  - experiment_results.csv")
        print("  - experiment_results.json")
        print("  - summary_report.md")
        
        return df
        
    except Exception as e:
        print(f"\n✗ Full experiment FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("="*70)
    print("Day 16: Advanced Experiments - Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # 测试1: 场景定义
    results['scenario_definitions'] = test_scenario_definitions()
    
    # 测试2: 实验管理器初始化
    results['experimenter_init'] = test_experimenter_initialization()
    
    # 测试3: 快速实验
    results['quick_experiment'] = test_quick_experiment()
    
    # 测试4: 多算法对比
    results['multi_agent'] = test_multi_agent_comparison()
    
    # 测试5: 鲁棒性场景
    results['robustness'] = test_robustness_scenarios()
    
    # 汇总结果
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r)
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("\n" + "="*70)
        print("All Day 16 tests PASSED!")
        print("="*70)
        
        # 询问是否运行完整实验
        print("\nWould you like to run the full experiment?")
        print("(This will take longer but produce complete results)")
        
        return 0
    else:
        print(f"\n{len(results) - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

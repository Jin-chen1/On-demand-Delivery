"""
Day 15: 模型评估与超参数调优测试脚本

测试功能：
1. 多场景模型评估
2. Reward权重敏感性分析
3. 超参数网格搜索
4. 可视化生成
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

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


def test_model_evaluation():
    """测试模型评估功能"""
    print("\n" + "="*70)
    print("Day 15 Test: Model Evaluation")
    print("="*70)
    
    from src.rl.evaluation_and_tuning import ModelEvaluator, TestScenarios
    
    # 测试场景获取
    scenarios = TestScenarios.get_all_scenarios()
    print(f"\nAvailable test scenarios: {len(scenarios)}")
    for name, config in scenarios.items():
        print(f"  - {name}: {config.get('description', '')}")
    
    quick_scenarios = TestScenarios.get_quick_scenarios()
    print(f"\nQuick test scenarios: {len(quick_scenarios)}")
    
    # 初始化评估器（快速模式）
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return False
    
    print(f"\nInitializing ModelEvaluator with quick_mode=True...")
    evaluator = ModelEvaluator(
        config_path=str(config_path),
        model_path=None,  # 不加载RL模型，只测试基线
        output_dir=str(PROJECT_ROOT / 'outputs' / 'day15_test_eval'),
        quick_mode=True
    )
    
    print(f"  Output directory: {evaluator.output_dir}")
    print(f"  Scenarios loaded: {len(evaluator.scenarios)}")
    
    # 运行快速评估（只测试greedy，1个场景，1个episode）
    print("\nRunning quick evaluation (greedy only, low_load, 1 episode)...")
    try:
        df = evaluator.run_evaluation(
            agent_types=['greedy'],
            scenario_names=['low_load'],
            n_episodes=1
        )
        
        print(f"\nEvaluation results shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        if not df.empty:
            print(f"\nResults summary:")
            print(f"  Completion rate: {df['completion_rate'].mean():.1%}")
            print(f"  Timeout rate: {df['timeout_rate'].mean():.1%}")
            print("\n✓ Model evaluation test PASSED")
            return True
        else:
            print("Warning: Empty results")
            return False
            
    except Exception as e:
        print(f"✗ Model evaluation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_weight_tuning():
    """测试Reward权重调优功能"""
    print("\n" + "="*70)
    print("Day 15 Test: Reward Weight Sensitivity Analysis")
    print("="*70)
    
    try:
        from src.rl.hyperparameter_tuning import RewardWeightTuner
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    
    print(f"\nInitializing RewardWeightTuner with quick_mode=True...")
    tuner = RewardWeightTuner(
        config_path=str(config_path),
        output_dir=str(PROJECT_ROOT / 'outputs' / 'day15_test_tuning'),
        quick_mode=True
    )
    
    print(f"  Output directory: {tuner.output_dir}")
    
    # 获取搜索空间
    search_space = tuner.get_weight_search_space()
    print(f"\nWeight search space:")
    for param, values in search_space.items():
        print(f"  {param}: {values}")
    
    # 注意：实际训练需要较长时间，这里只验证接口
    print("\n[Note] Full sensitivity analysis requires training, skipping in quick test")
    print("✓ Reward weight tuning interface test PASSED")
    return True


def test_hyperparameter_tuning():
    """测试超参数调优功能"""
    print("\n" + "="*70)
    print("Day 15 Test: Hyperparameter Grid Search")
    print("="*70)
    
    try:
        from src.rl.hyperparameter_tuning import HyperparameterTuner
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    
    print(f"\nInitializing HyperparameterTuner with quick_mode=True...")
    tuner = HyperparameterTuner(
        config_path=str(config_path),
        output_dir=str(PROJECT_ROOT / 'outputs' / 'day15_test_hp'),
        quick_mode=True
    )
    
    print(f"  Output directory: {tuner.output_dir}")
    
    # 获取搜索网格
    grid = tuner.get_hyperparameter_grid()
    print(f"\nHyperparameter grid:")
    for param, values in grid.items():
        print(f"  {param}: {values}")
    
    # 计算组合数
    import itertools
    total_configs = 1
    for values in grid.values():
        total_configs *= len(values)
    print(f"\nTotal configurations: {total_configs}")
    
    print("\n[Note] Full grid search requires training, skipping in quick test")
    print("✓ Hyperparameter tuning interface test PASSED")
    return True


def test_visualization():
    """测试可视化功能"""
    print("\n" + "="*70)
    print("Day 15 Test: Visualization")
    print("="*70)
    
    try:
        from src.rl.visualization_day15 import (
            plot_scenario_comparison,
            plot_completion_vs_timeout,
            plot_capacity_curve,
            plot_pareto_frontier,
            plot_radar_chart,
            generate_all_visualizations
        )
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # 创建模拟数据
    np.random.seed(42)
    
    mock_data = []
    scenarios = ['low_load', 'medium_load', 'high_load']
    agents = ['greedy', 'ortools', 'alns']
    
    for scenario in scenarios:
        for agent in agents:
            for ep in range(3):
                # 模拟不同算法性能差异
                base_completion = {'greedy': 0.3, 'ortools': 0.4, 'alns': 0.5}
                base_timeout = {'greedy': 0.6, 'ortools': 0.5, 'alns': 0.4}
                
                # 场景难度影响
                difficulty = {'low_load': 1.0, 'medium_load': 0.8, 'high_load': 0.6}
                
                mock_data.append({
                    'scenario_name': scenario,
                    'agent_type': agent,
                    'episode': ep + 1,
                    'completion_rate': base_completion[agent] * difficulty[scenario] + np.random.randn() * 0.05,
                    'timeout_rate': base_timeout[agent] * (2 - difficulty[scenario]) + np.random.randn() * 0.05,
                    'total_reward': np.random.randn() * 50 + base_completion[agent] * 100,
                    'avg_service_time': 1200 + np.random.randn() * 200,
                    'total_distance': 50000 + np.random.randn() * 5000,
                    'episode_length': 3600
                })
    
    df = pd.DataFrame(mock_data)
    
    print(f"\nMock data created: {df.shape}")
    print(f"Scenarios: {df['scenario_name'].unique().tolist()}")
    print(f"Agents: {df['agent_type'].unique().tolist()}")
    
    # 测试可视化函数
    output_dir = PROJECT_ROOT / 'outputs' / 'day15_test_viz'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations to: {output_dir}")
    
    try:
        plot_scenario_comparison(df, output_dir, 'completion_rate')
        print("  ✓ Scenario comparison plot generated")
        
        plot_completion_vs_timeout(df, output_dir)
        print("  ✓ Completion vs timeout plot generated")
        
        plot_capacity_curve(df, output_dir)
        print("  ✓ Capacity curve plot generated")
        
        plot_pareto_frontier(df, output_dir)
        print("  ✓ Pareto frontier plot generated")
        
        plot_radar_chart(df, output_dir)
        print("  ✓ Radar chart generated")
        
        print("\n✓ Visualization test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整评估流程"""
    print("\n" + "="*70)
    print("Day 15 Test: Full Evaluation Pipeline")
    print("="*70)
    
    from src.rl.evaluation_and_tuning import ModelEvaluator
    
    config_path = PROJECT_ROOT / 'config' / 'rl_config.yaml'
    output_dir = PROJECT_ROOT / 'outputs' / 'day15_full_test'
    
    print(f"\nInitializing full pipeline test...")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_dir}")
    
    evaluator = ModelEvaluator(
        config_path=str(config_path),
        model_path=None,
        output_dir=str(output_dir),
        quick_mode=True
    )
    
    # 运行多基线、多场景评估
    print("\nRunning evaluation with multiple baselines and scenarios...")
    print("  Agents: greedy, ortools")
    print("  Scenarios: low_load, medium_load")
    print("  Episodes: 2")
    
    try:
        df = evaluator.run_evaluation(
            agent_types=['greedy', 'ortools'],
            scenario_names=['low_load', 'medium_load'],
            n_episodes=2
        )
        
        print(f"\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        # 按场景和Agent汇总
        summary = df.groupby(['scenario_name', 'agent_type']).agg({
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std']
        }).round(4)
        
        print(f"\n{summary.to_string()}")
        
        # 找出每个场景的最佳算法
        print("\n\nBest agent per scenario (by completion rate):")
        for scenario in df['scenario_name'].unique():
            sdata = df[df['scenario_name'] == scenario].groupby('agent_type')['completion_rate'].mean()
            best = sdata.idxmax()
            print(f"  {scenario}: {best} ({sdata.max():.1%})")
        
        # 检查输出文件
        print(f"\n\nOutput files:")
        for f in output_dir.iterdir():
            print(f"  - {f.name}")
        
        print("\n✓ Full pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*70)
    print("Day 15: Model Evaluation & Hyperparameter Tuning - Test Suite")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # 测试1: 模型评估
    results['model_evaluation'] = test_model_evaluation()
    
    # 测试2: Reward权重调优
    results['reward_tuning'] = test_reward_weight_tuning()
    
    # 测试3: 超参数调优
    results['hp_tuning'] = test_hyperparameter_tuning()
    
    # 测试4: 可视化
    results['visualization'] = test_visualization()
    
    # 测试5: 完整流程
    results['full_pipeline'] = test_full_pipeline()
    
    # 汇总结果
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed == 0:
        print("\n" + "="*70)
        print("All Day 15 tests PASSED!")
        print("="*70)
        return 0
    else:
        print(f"\n{failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

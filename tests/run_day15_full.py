"""
Day 15: 完整评估与调优脚本

运行完整的Day 15任务：
1. 在测试集上评估模型性能（多场景）
2. 调整Reward权重（敏感性分析）
3. 超参数网格搜索优化
4. 生成可视化报告

使用方法:
    # 快速测试模式
    python tests/run_day15_full.py --quick
    
    # 完整评估模式
    python tests/run_day15_full.py --full
    
    # 仅评估基线
    python tests/run_day15_full.py --baselines-only
    
    # 指定模型路径进行评估
    python tests/run_day15_full.py --model-path outputs/day14_curriculum/final_model.zip
"""

import sys
import os
import argparse
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


def run_model_evaluation(config_path: str, model_path: str, output_dir: Path,
                         quick_mode: bool, agent_types: list, n_episodes: int):
    """运行模型评估"""
    from src.rl.evaluation_and_tuning import ModelEvaluator
    
    print("\n" + "="*70)
    print("PHASE 1: Model Evaluation on Test Scenarios")
    print("="*70)
    
    evaluator = ModelEvaluator(
        config_path=config_path,
        model_path=model_path,
        output_dir=str(output_dir / 'evaluation'),
        quick_mode=quick_mode
    )
    
    print(f"\nScenarios to evaluate: {list(evaluator.scenarios.keys())}")
    print(f"Agents to evaluate: {agent_types}")
    print(f"Episodes per configuration: {n_episodes}")
    
    # 运行评估
    df = evaluator.run_evaluation(
        agent_types=agent_types,
        n_episodes=n_episodes
    )
    
    # 打印汇总
    print("\n" + "-"*50)
    print("EVALUATION SUMMARY")
    print("-"*50)
    
    summary = df.groupby(['scenario_name', 'agent_type']).agg({
        'completion_rate': 'mean',
        'timeout_rate': 'mean'
    }).round(4)
    print(summary.to_string())
    
    return df


def run_sensitivity_analysis(config_path: str, output_dir: Path,
                             quick_mode: bool, training_steps: int):
    """运行Reward权重敏感性分析"""
    from src.rl.hyperparameter_tuning import RewardWeightTuner
    
    print("\n" + "="*70)
    print("PHASE 2: Reward Weight Sensitivity Analysis")
    print("="*70)
    
    tuner = RewardWeightTuner(
        config_path=config_path,
        output_dir=str(output_dir / 'sensitivity'),
        quick_mode=quick_mode
    )
    
    print(f"\nSearch space:")
    for param, values in tuner.get_weight_search_space().items():
        print(f"  {param}: {values}")
    
    print(f"\nTraining steps per configuration: {training_steps}")
    
    # 运行敏感性分析
    df = tuner.run_sensitivity_analysis(
        training_steps=training_steps,
        n_eval_episodes=3
    )
    
    # 找出最佳配置
    best_idx = df['mean_reward'].idxmax()
    best = df.loc[best_idx]
    
    print("\n" + "-"*50)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("-"*50)
    print(f"Best configuration: {best['config_id']}")
    print(f"  Mean reward: {best['mean_reward']:.2f}")
    print(f"  Completion rate: {best['mean_completion_rate']:.1%}")
    print(f"  Timeout rate: {best['mean_timeout_rate']:.1%}")
    
    return df


def run_hyperparameter_search(config_path: str, output_dir: Path,
                              quick_mode: bool, training_steps: int,
                              max_configs: int):
    """运行超参数网格搜索"""
    from src.rl.hyperparameter_tuning import HyperparameterTuner
    
    print("\n" + "="*70)
    print("PHASE 3: Hyperparameter Grid Search")
    print("="*70)
    
    tuner = HyperparameterTuner(
        config_path=config_path,
        output_dir=str(output_dir / 'hyperparameter'),
        quick_mode=quick_mode
    )
    
    grid = tuner.get_hyperparameter_grid()
    print(f"\nHyperparameter grid:")
    for param, values in grid.items():
        print(f"  {param}: {values}")
    
    print(f"\nTraining steps per configuration: {training_steps}")
    print(f"Max configurations to test: {max_configs}")
    
    # 运行网格搜索
    df = tuner.run_grid_search(
        training_steps=training_steps,
        n_eval_episodes=3,
        max_configs=max_configs
    )
    
    # 找出最佳配置
    best_idx = df['mean_reward'].idxmax()
    best = df.loc[best_idx]
    
    print("\n" + "-"*50)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("-"*50)
    print(f"Best configuration: {best['config_id']}")
    print(f"  Mean reward: {best['mean_reward']:.2f}")
    print(f"  Completion rate: {best['mean_completion_rate']:.1%}")
    
    return df


def generate_visualizations(output_dir: Path, eval_df, sensitivity_df, hp_df):
    """生成可视化报告"""
    from src.rl.visualization_day15 import generate_all_visualizations
    
    print("\n" + "="*70)
    print("PHASE 4: Generating Visualizations")
    print("="*70)
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    generate_all_visualizations(
        eval_df=eval_df,
        sensitivity_df=sensitivity_df,
        hp_df=hp_df,
        output_dir=viz_dir
    )
    
    print(f"\nVisualizations saved to: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description='Day 15: Model Evaluation & Hyperparameter Tuning')
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (reduced data)')
    parser.add_argument('--full', action='store_true',
                       help='Full evaluation mode')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Only evaluate baseline algorithms')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--config', type=str, default='config/rl_config.yaml',
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip sensitivity analysis and HP search')
    
    args = parser.parse_args()
    
    # 确定模式
    quick_mode = args.quick or (not args.full)
    
    # 配置路径
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / 'outputs' / 'day15_full' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Day 15: Model Evaluation & Hyperparameter Tuning")
    print("="*70)
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_path or 'None (baselines only)'}")
    print("="*70)
    
    # 确定Agent类型
    if args.baselines_only:
        agent_types = ['greedy', 'ortools', 'alns']
    else:
        agent_types = ['greedy', 'ortools', 'alns']
        if args.model_path:
            agent_types.insert(0, 'rl_ppo')
    
    # 参数设置
    if quick_mode:
        n_episodes = 2
        training_steps = 10000
        max_configs = 6
    else:
        n_episodes = 5
        training_steps = 50000
        max_configs = 15
    
    # Phase 1: 模型评估
    eval_df = run_model_evaluation(
        config_path=str(config_path),
        model_path=args.model_path,
        output_dir=output_dir,
        quick_mode=quick_mode,
        agent_types=agent_types,
        n_episodes=n_episodes
    )
    
    sensitivity_df = None
    hp_df = None
    
    if not args.skip_tuning:
        # Phase 2: 敏感性分析
        try:
            sensitivity_df = run_sensitivity_analysis(
                config_path=str(config_path),
                output_dir=output_dir,
                quick_mode=quick_mode,
                training_steps=training_steps
            )
        except Exception as e:
            print(f"Sensitivity analysis skipped: {e}")
        
        # Phase 3: 超参数搜索
        try:
            hp_df = run_hyperparameter_search(
                config_path=str(config_path),
                output_dir=output_dir,
                quick_mode=quick_mode,
                training_steps=training_steps,
                max_configs=max_configs
            )
        except Exception as e:
            print(f"Hyperparameter search skipped: {e}")
    
    # Phase 4: 可视化
    try:
        generate_visualizations(output_dir, eval_df, sensitivity_df, hp_df)
    except Exception as e:
        print(f"Visualization generation skipped: {e}")
    
    # 最终报告
    print("\n" + "="*70)
    print("Day 15 COMPLETED")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nKey files:")
    for f in output_dir.rglob('*'):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

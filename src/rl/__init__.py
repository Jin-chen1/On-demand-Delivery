"""
强化学习模块
提供Gym-like环境接口，用于训练调度策略
"""

# 统一检测Stable-Baselines3是否可用（避免多处重复定义）
try:
    from stable_baselines3 import PPO, DQN, A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .rl_environment import DeliveryRLEnvironment
from .state_representation import StateEncoder
from .reward_function import RewardCalculator
from .baseline_agents import (
    RandomAgent,
    BaselineAgent,
    create_baseline_agent,
    run_baseline_episode,
    benchmark_agents
)
from .curriculum_learning import (
    CurriculumManager,
    CurriculumStage,
    StagePerformance,
    CurriculumLearningCallback,
    AdvanceCondition,
    create_curriculum_stages_from_config,
    print_curriculum_overview
)

# Day 15: 评估与调优模块
from .evaluation_and_tuning import (
    ModelEvaluator,
    TestScenarios,
    EvaluationResult
)
from .hyperparameter_tuning import (
    RewardWeightTuner,
    HyperparameterTuner,
    TuningResult
)
from .utils import extract_simulation_metrics

__all__ = [
    # SB3可用性标志
    'SB3_AVAILABLE',
    'DeliveryRLEnvironment',
    'StateEncoder', 
    'RewardCalculator',
    'RandomAgent',
    'BaselineAgent',
    'create_baseline_agent',
    'run_baseline_episode',
    'benchmark_agents',
    # Day 14: 课程学习
    'CurriculumManager',
    'CurriculumStage',
    'StagePerformance',
    'CurriculumLearningCallback',
    'AdvanceCondition',
    'create_curriculum_stages_from_config',
    'print_curriculum_overview',
    # Day 15: 评估与调优
    'ModelEvaluator',
    'TestScenarios',
    'EvaluationResult',
    'RewardWeightTuner',
    'HyperparameterTuner',
    'TuningResult',
    # 公共工具函数
    'extract_simulation_metrics'
]

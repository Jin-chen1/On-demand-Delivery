"""
Day 14: 课程学习模块 (Curriculum Learning) - 完整版

实现从低负载到高负载的难度递增训练策略，帮助RL Agent逐步适应复杂场景。

核心功能：
1. 多阶段课程设计（订单量、骑手数、时间窗等多维度）
2. 自动难度调整（基于Agent性能）
3. 平滑过渡机制
4. 课程进度追踪和可视化
5. 回退机制（性能下降时降低难度）

使用说明：
┌─────────────────────────────────────────────────────────────────────────┐
│ 本模块提供"完整版"课程学习实现，包含：                                  │
│   - CurriculumManager: 课程阶段管理器                                   │
│   - CurriculumLearningCallback: SB3训练回调                             │
│                                                                          │
│ 注意：当前RLTrainer主流程使用的是"简化版"课程学习：                     │
│   - RLTrainer.train_with_curriculum() + CurriculumAdvanceCallback       │
│   - 直接读取YAML配置的curriculum_stages                                 │
│   - 支持达标跳转和加时赛机制                                            │
│                                                                          │
│ 如果要切换到本模块的完整版实现，需要：                                  │
│   1. 修改RLTrainer.train()调用CurriculumManager                         │
│   2. 对齐YAML配置字段（完整版使用不同的配置格式）                       │
│   3. 注意lambda闭包在SubprocVecEnv下的pickle问题                        │
└─────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class DifficultyDimension(Enum):
    """难度维度枚举"""
    ORDER_VOLUME = "order_volume"
    COURIER_COUNT = "courier_count"
    TIME_WINDOW = "time_window"
    PEAK_INTENSITY = "peak_intensity"


class AdvanceCondition(Enum):
    """进阶条件类型"""
    FIXED_STEPS = "fixed_steps"
    PERFORMANCE_THRESHOLD = "performance"
    ADAPTIVE = "adaptive"


@dataclass
class CurriculumStage:
    """课程阶段定义"""
    name: str
    description: str
    total_orders: int
    num_couriers: int
    simulation_duration: int = 7200
    dispatch_interval: float = 30.0
    time_window_multiplier: float = 1.0
    peak_rate_multiplier: float = 1.0
    timesteps: int = 100000
    learning_rate: Optional[float] = None
    advance_condition: AdvanceCondition = AdvanceCondition.FIXED_STEPS
    min_completion_rate: float = 0.5
    max_timeout_rate: float = 0.5
    min_avg_reward: float = -100.0
    difficulty_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'total_orders': self.total_orders,
            'num_couriers': self.num_couriers,
            'simulation_duration': self.simulation_duration,
            'time_window_multiplier': self.time_window_multiplier,
            'peak_rate_multiplier': self.peak_rate_multiplier,
            'timesteps': self.timesteps,
            'advance_condition': self.advance_condition.value,
            'difficulty_score': self.difficulty_score
        }


@dataclass
class StagePerformance:
    """阶段性能记录"""
    stage_name: str
    timesteps_trained: int = 0
    episodes_completed: int = 0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    best_reward: float = float('-inf')
    completion_rate: float = 0.0
    timeout_rate: float = 1.0
    reward_history: List[float] = field(default_factory=list)
    completion_history: List[float] = field(default_factory=list)
    timeout_history: List[float] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def update(self, reward: float, completion_rate: float, timeout_rate: float):
        self.reward_history.append(reward)
        self.completion_history.append(completion_rate)
        self.timeout_history.append(timeout_rate)
        self.avg_reward = np.mean(self.reward_history[-100:])
        self.std_reward = np.std(self.reward_history[-100:])
        self.best_reward = max(self.best_reward, reward)
        self.completion_rate = np.mean(self.completion_history[-100:])
        self.timeout_rate = np.mean(self.timeout_history[-100:])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'stage_name': self.stage_name,
            'timesteps_trained': self.timesteps_trained,
            'episodes_completed': self.episodes_completed,
            'avg_reward': float(self.avg_reward),
            'best_reward': float(self.best_reward) if self.best_reward != float('-inf') else None,
            'completion_rate': float(self.completion_rate),
            'timeout_rate': float(self.timeout_rate)
        }


class CurriculumManager:
    """
    课程学习管理器
    管理训练课程的进度、难度调整和性能追踪
    """
    
    def __init__(self, 
                 stages: Optional[List[CurriculumStage]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stages = stages if stages else self._create_default_stages()
        self.current_stage_idx = 0
        self.total_timesteps_trained = 0
        self.total_episodes = 0
        
        # 性能记录
        self.stage_performances: Dict[str, StagePerformance] = {}
        for stage in self.stages:
            self.stage_performances[stage.name] = StagePerformance(stage_name=stage.name)
        
        # 回退机制配置
        self.enable_fallback = self.config.get('enable_fallback', True)
        self.fallback_threshold = self.config.get('fallback_threshold', 0.3)
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config.get('max_consecutive_failures', 3)
        
        # 平滑过渡配置
        self.enable_smooth_transition = self.config.get('enable_smooth_transition', True)
        self.transition_episodes = self.config.get('transition_episodes', 10)
        
        logger.info(f"课程管理器初始化完成，共 {len(self.stages)} 个阶段")
    
    def _create_default_stages(self) -> List[CurriculumStage]:
        """创建默认课程阶段（从低负载到高负载）"""
        return [
            CurriculumStage(
                name="warmup",
                description="热身阶段：低订单量、充足运力",
                total_orders=300, num_couriers=20,
                time_window_multiplier=1.5, peak_rate_multiplier=0.8,
                timesteps=100000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.6, max_timeout_rate=0.4,
                difficulty_score=0.2
            ),
            CurriculumStage(
                name="low_load",
                description="低负载阶段：订单/骑手比=25",
                total_orders=500, num_couriers=20,
                time_window_multiplier=1.2, peak_rate_multiplier=1.0,
                timesteps=150000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.5, max_timeout_rate=0.5,
                difficulty_score=0.35
            ),
            CurriculumStage(
                name="medium_low_load",
                description="中低负载阶段：订单/骑手比=35",
                total_orders=700, num_couriers=20,
                time_window_multiplier=1.1, peak_rate_multiplier=1.0,
                timesteps=200000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.45, max_timeout_rate=0.55,
                difficulty_score=0.5
            ),
            CurriculumStage(
                name="medium_load",
                description="中负载阶段：订单/骑手比=50（MVP基准）",
                total_orders=1000, num_couriers=20,
                time_window_multiplier=1.0, peak_rate_multiplier=1.0,
                timesteps=250000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.4, max_timeout_rate=0.6,
                difficulty_score=0.65
            ),
            CurriculumStage(
                name="medium_high_load",
                description="中高负载阶段：订单/骑手比=62.5",
                total_orders=1250, num_couriers=20,
                time_window_multiplier=1.0, peak_rate_multiplier=1.2,
                timesteps=250000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.35, max_timeout_rate=0.65,
                difficulty_score=0.75
            ),
            CurriculumStage(
                name="high_load",
                description="高负载阶段：订单/骑手比=75（压力测试）",
                total_orders=1500, num_couriers=20,
                time_window_multiplier=0.9, peak_rate_multiplier=1.5,
                timesteps=300000,
                advance_condition=AdvanceCondition.PERFORMANCE_THRESHOLD,
                min_completion_rate=0.3, max_timeout_rate=0.7,
                difficulty_score=0.85
            ),
            CurriculumStage(
                name="extreme_load",
                description="极端负载阶段：订单/骑手比=100",
                total_orders=2000, num_couriers=20,
                time_window_multiplier=0.8, peak_rate_multiplier=2.0,
                timesteps=300000,
                advance_condition=AdvanceCondition.FIXED_STEPS,
                min_completion_rate=0.25, max_timeout_rate=0.75,
                difficulty_score=1.0
            )
        ]
    
    @property
    def current_stage(self) -> CurriculumStage:
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]
    
    @property
    def current_performance(self) -> StagePerformance:
        return self.stage_performances[self.current_stage.name]
    
    @property
    def is_completed(self) -> bool:
        return self.current_stage_idx >= len(self.stages)
    
    @property
    def progress_percentage(self) -> float:
        total_timesteps = sum(s.timesteps for s in self.stages)
        return min(100.0, self.total_timesteps_trained / total_timesteps * 100)
    
    def get_stage_config(self) -> Dict[str, Any]:
        """获取当前阶段的环境配置"""
        stage = self.current_stage
        return {
            'total_orders': stage.total_orders,
            'num_couriers': stage.num_couriers,
            'simulation_duration': stage.simulation_duration,
            'dispatch_interval': stage.dispatch_interval,
            'time_window_multiplier': stage.time_window_multiplier,
            'order_generation': {'peak_rate_multiplier': stage.peak_rate_multiplier}
        }
    
    def record_episode_result(self, reward: float, completion_rate: float, 
                              timeout_rate: float, timesteps: int = 0) -> None:
        """记录Episode结果"""
        perf = self.current_performance
        perf.update(reward, completion_rate, timeout_rate)
        perf.episodes_completed += 1
        perf.timesteps_trained += timesteps
        self.total_episodes += 1
        self.total_timesteps_trained += timesteps
    
    def should_advance(self) -> bool:
        """检查是否应该进入下一阶段"""
        if self.is_completed:
            return False
        
        stage = self.current_stage
        perf = self.current_performance
        
        if stage.advance_condition == AdvanceCondition.FIXED_STEPS:
            return perf.timesteps_trained >= stage.timesteps
        
        elif stage.advance_condition == AdvanceCondition.PERFORMANCE_THRESHOLD:
            if perf.timesteps_trained < stage.timesteps * 0.5:
                return False
            meets_completion = perf.completion_rate >= stage.min_completion_rate
            meets_timeout = perf.timeout_rate <= stage.max_timeout_rate
            if meets_completion and meets_timeout:
                logger.info(f"阶段 {stage.name} 达到性能阈值")
                return True
            if perf.timesteps_trained >= stage.timesteps:
                logger.info(f"阶段 {stage.name} 达到最大步数")
                return True
            return False
        
        return False
    
    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.is_completed:
            return False
        
        self.current_performance.end_time = datetime.now().isoformat()
        self.current_stage_idx += 1
        self.consecutive_failures = 0
        
        if self.current_stage_idx < len(self.stages):
            self.current_performance.start_time = datetime.now().isoformat()
            logger.info("="*60)
            logger.info(f"进入课程阶段 {self.current_stage_idx + 1}/{len(self.stages)}: {self.current_stage.name}")
            logger.info(f"  订单数: {self.current_stage.total_orders}, 骑手数: {self.current_stage.num_couriers}")
            logger.info("="*60)
            return True
        else:
            logger.info("所有课程阶段已完成！")
            return False
    
    def should_fallback(self) -> bool:
        """
        检查是否需要回退到上一阶段
        
        回退条件：当前阶段的平均奖励相比上一阶段下降超过阈值，且连续失败次数达到上限
        
        注意：负奖励场景的处理策略
        - 如果上一阶段avg_reward <= 0，ratio设为1.0，不触发回退
        - 这是因为负奖励场景下，奖励比值没有明确的物理意义
        - 如果需要在全程负奖励场景下启用回退，建议改用绝对差值或其他指标
        """
        if not self.enable_fallback or self.current_stage_idx == 0:
            return False
        
        perf = self.current_performance
        if perf.episodes_completed < 10:
            return False
        
        prev_perf = self.stage_performances[self.stages[self.current_stage_idx - 1].name]
        
        # 计算奖励比值
        # 注意：当prev_perf.avg_reward <= 0时，ratio设为1.0，不触发回退
        # 这是一个策略选择，适用于奖励通常为正的场景
        if prev_perf.avg_reward > 0:
            ratio = perf.avg_reward / prev_perf.avg_reward
        else:
            # 负奖励或零奖励场景：跳过基于比值的回退检测
            # 如需处理全程负奖励，可改用: ratio = 1.0 if perf.avg_reward >= prev_perf.avg_reward else 0.0
            ratio = 1.0
        
        if ratio < (1 - self.fallback_threshold):
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                return True
        else:
            self.consecutive_failures = 0
        return False
    
    def fallback_stage(self) -> bool:
        """回退到上一阶段"""
        if self.current_stage_idx == 0:
            return False
        logger.warning(f"从阶段 {self.current_stage.name} 回退")
        self.current_stage_idx -= 1
        self.consecutive_failures = 0
        return True
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """获取课程学习摘要"""
        return {
            'current_stage': self.current_stage.name,
            'current_stage_idx': self.current_stage_idx,
            'total_stages': len(self.stages),
            'progress_percentage': self.progress_percentage,
            'total_timesteps_trained': self.total_timesteps_trained,
            'is_completed': self.is_completed,
            'stages': [s.to_dict() for s in self.stages],
            'performances': {n: p.to_dict() for n, p in self.stage_performances.items()}
        }
    
    def save_progress(self, filepath: Path) -> None:
        """保存课程进度"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        progress = {
            'current_stage_idx': self.current_stage_idx,
            'total_timesteps_trained': self.total_timesteps_trained,
            'total_episodes': self.total_episodes,
            'stage_performances': {n: p.to_dict() for n, p in self.stage_performances.items()},
            'saved_at': datetime.now().isoformat()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        logger.info(f"课程进度已保存: {filepath}")
    
    def load_progress(self, filepath: Path) -> bool:
        """加载课程进度"""
        filepath = Path(filepath)
        if not filepath.exists():
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        self.current_stage_idx = progress['current_stage_idx']
        self.total_timesteps_trained = progress['total_timesteps_trained']
        self.total_episodes = progress['total_episodes']
        logger.info(f"课程进度已加载，当前阶段: {self.current_stage.name}")
        return True


# ============================================================
# 课程学习训练回调（用于Stable-Baselines3）
# ============================================================

# 使用统一的SB3_AVAILABLE标志，避免重复定义
# 注意：由于循环导入问题，这里不能从__init__.py导入，需要直接检测
try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_CALLBACK_AVAILABLE = True
except ImportError:
    _SB3_CALLBACK_AVAILABLE = False
    BaseCallback = object


class CurriculumLearningCallback(BaseCallback):
    """
    课程学习回调（Day 14完整版，目前未默认启用）
    
    在训练过程中监控性能，自动调整课程阶段。
    支持回退机制和平滑过渡。
    
    注意：当前train_rl_agent.py中的train_with_curriculum()使用的是简化版实现，
    如需使用本回调，请参考train_with_curriculum.py中的CurriculumTrainer。
    
    使用方式：
    ```python
    from src.rl.curriculum_learning import CurriculumManager, CurriculumLearningCallback
    
    curriculum = CurriculumManager()
    callback = CurriculumLearningCallback(
        curriculum_manager=curriculum,
        env_creator_fn=trainer.create_env,
        check_freq=1000
    )
    model.learn(total_timesteps=100000, callback=callback)
    ```
    """
    
    def __init__(self, 
                 curriculum_manager: CurriculumManager,
                 env_creator_fn,
                 check_freq: int = 1000,
                 verbose: int = 1):
        """
        初始化课程学习回调
        
        Args:
            curriculum_manager: 课程管理器实例
            env_creator_fn: 环境创建函数，接受配置参数
            check_freq: 检查频率（步数）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.env_creator_fn = env_creator_fn
        self.check_freq = check_freq
        
        self.episode_rewards = []
        self.episode_completion_rates = []
        self.episode_timeout_rates = []
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 收集Episode统计
        # 注意：DeliveryRLEnvironment.step()返回的info中没有'episode'键
        # 而是在episode结束时添加'episode_stats'键
        # 因此检测条件应该是'episode_stats'而非'episode'
        infos = self.locals.get('infos', [{}])
        if infos and len(infos) > 0:
            info = infos[0]
            if 'episode_stats' in info:
                stats = info['episode_stats']
                self.curriculum.record_episode_result(
                    reward=stats.get('total_reward', 0),
                    completion_rate=stats.get('completion_rate', 0),
                    timeout_rate=stats.get('timeout_rate', 1),
                    timesteps=stats.get('episode_steps', 0)
                )
        
        # 定期检查是否需要进阶或回退
        if self.n_calls % self.check_freq == 0:
            self._check_curriculum_progress()
        
        return True
    
    def _check_curriculum_progress(self):
        """检查课程进度，决定是否切换阶段"""
        # 检查是否需要回退
        if self.curriculum.should_fallback():
            self.curriculum.fallback_stage()
            self._update_environment()
            return
        
        # 检查是否可以进阶
        if self.curriculum.should_advance():
            if self.curriculum.advance_stage():
                self._update_environment()
    
    def _update_environment(self):
        """更新训练环境到新的课程阶段"""
        new_config = self.curriculum.get_stage_config()
        
        if self.verbose > 0:
            logger.info(f"更新环境配置: {self.curriculum.current_stage.name}")
            logger.info(f"  订单数: {new_config['total_orders']}")
            logger.info(f"  骑手数: {new_config['num_couriers']}")
        
        # 创建新环境并更新模型
        # 注意：使用工厂函数而非捕获已创建的env实例，确保每次创建新env
        # 这样在多环境或资源隔离场景下更安全
        #
        # 警告：当前使用lambda闭包创建环境，在DummyVecEnv下是安全的
        # 但如果改用SubprocVecEnv（Windows），需要将env_creator提升到模块顶层
        # 参考 train_rl_agent.py 中的 _make_env_factory 实现
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # 保存配置到局部变量，避免闭包捕获self
            env_creator = self.env_creator_fn
            config = new_config
            
            # 注意：这里使用DummyVecEnv，lambda闭包是安全的
            # 如果改用SubprocVecEnv，需要重构为模块级工厂函数
            vec_env = DummyVecEnv([lambda: env_creator(**config)])
            self.model.set_env(vec_env)
        except Exception as e:
            logger.error(f"更新环境失败: {e}")
    
    def _on_training_end(self):
        """训练结束时保存进度"""
        summary = self.curriculum.get_curriculum_summary()
        logger.info("="*60)
        logger.info("课程学习训练结束")
        logger.info(f"  最终阶段: {summary['current_stage']}")
        logger.info(f"  总训练步数: {summary['total_timesteps_trained']:,}")
        logger.info(f"  进度: {summary['progress_percentage']:.1f}%")
        logger.info("="*60)


def create_curriculum_stages_from_config(config: Dict[str, Any]) -> List[CurriculumStage]:
    """
    从配置文件创建课程阶段列表
    
    Args:
        config: 配置字典，包含curriculum_stages列表
        
    Returns:
        课程阶段列表
    """
    stages_config = config.get('curriculum_stages', [])
    stages = []
    
    for sc in stages_config:
        advance_cond = sc.get('advance_condition', 'fixed_steps')
        if advance_cond == 'performance':
            advance_cond = AdvanceCondition.PERFORMANCE_THRESHOLD
        elif advance_cond == 'adaptive':
            advance_cond = AdvanceCondition.ADAPTIVE
        else:
            advance_cond = AdvanceCondition.FIXED_STEPS
        
        stage = CurriculumStage(
            name=sc.get('name', 'unnamed'),
            description=sc.get('description', ''),
            total_orders=sc.get('total_orders', 500),
            num_couriers=sc.get('num_couriers', 20),
            simulation_duration=sc.get('simulation_duration', 7200),
            dispatch_interval=sc.get('dispatch_interval', 30.0),
            time_window_multiplier=sc.get('time_window_multiplier', 1.0),
            peak_rate_multiplier=sc.get('peak_rate_multiplier', 1.0),
            timesteps=sc.get('timesteps', 100000),
            learning_rate=sc.get('learning_rate'),
            advance_condition=advance_cond,
            min_completion_rate=sc.get('min_completion_rate', 0.5),
            max_timeout_rate=sc.get('max_timeout_rate', 0.5),
            difficulty_score=sc.get('difficulty_score', 0.5)
        )
        stages.append(stage)
    
    return stages


def print_curriculum_overview(manager: CurriculumManager) -> None:
    """打印课程概览"""
    print("\n" + "="*70)
    print("课程学习设置概览")
    print("="*70)
    print(f"{'阶段':<20} {'订单数':<10} {'骑手数':<10} {'难度':<10} {'步数':<15}")
    print("-"*70)
    
    for i, stage in enumerate(manager.stages):
        marker = ">>>" if i == manager.current_stage_idx else "   "
        print(f"{marker} {stage.name:<17} {stage.total_orders:<10} {stage.num_couriers:<10} "
              f"{stage.difficulty_score:<10.2f} {stage.timesteps:<15,}")
    
    print("-"*70)
    print(f"总训练步数: {sum(s.timesteps for s in manager.stages):,}")
    print(f"当前进度: {manager.progress_percentage:.1f}%")
    print("="*70 + "\n")

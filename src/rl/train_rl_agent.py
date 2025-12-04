"""
Day 13: PPO训练流水线
基于Stable-Baselines3搭建完整的PPO训练流水线

功能：
1. TensorBoard监控
2. 自定义回调函数（训练监控、早停、动态调整）
3. 课程学习支持（从低负载到高负载）
4. 模型保存与恢复
5. 基线对比评估

┌─────────────────────────────────────────────────────────────────────────┐
│                        课程学习实现说明                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ 项目中有两套课程学习实现，各有适用场景：                                │
│                                                                          │
│ 【简化版】本模块 - RLTrainer.train_with_curriculum()                    │
│   ├── 特点：直接读取YAML配置，支持达标跳转和加时赛                      │
│   ├── 适用：快速实验、基础训练、论文复现                                │
│   ├── 配置：rl_config.yaml → training.curriculum.curriculum_stages     │
│   └── 状态：✅ 当前主流程使用                                           │
│                                                                          │
│ 【完整版】src/rl/curriculum_learning.py                                 │
│   ├── 特点：回退机制、平滑过渡、多维度难度评分                          │
│   ├── 适用：需要精细控制课程策略的高级场景                              │
│   ├── 类：CurriculumManager + CurriculumLearningCallback                │
│   └── 状态：🔶 预留扩展，未接入主流程                                   │
│                                                                          │
│ 注意：两套实现相互独立，修改一套不会影响另一套！                        │
│ 如需使用完整版，请参考curriculum_learning.py中的使用说明。              │
└─────────────────────────────────────────────────────────────────────────┘

依赖：pip install stable-baselines3[extra] tensorboard
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable

# RL训练库 - 使用统一的SB3_AVAILABLE标志
from . import SB3_AVAILABLE

if SB3_AVAILABLE:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        BaseCallback, 
        EvalCallback, 
        CheckpointCallback,
        CallbackList
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
else:
    print("警告：Stable-Baselines3未安装，RL训练功能不可用")
    print("安装命令：pip install stable-baselines3[extra]")

from gymnasium import spaces
from .rl_environment import DeliveryRLEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 模块级环境工厂函数（用于SubprocVecEnv的pickle兼容）
# ============================================================
# Windows多进程兼容性说明：
# - Windows使用spawn模式启动子进程，要求传递给SubprocVecEnv的函数必须可pickle
# - lambda函数通常不能被pickle，因此环境工厂函数必须在模块顶层定义
# - 当前实现：
#   - _make_env_factory在模块顶层，可被pickle ✅
#   - train_with_curriculum使用DummyVecEnv（单进程），无pickle问题 ✅
#   - CurriculumLearningCallback._update_environment使用DummyVecEnv ✅
# - 如果未来需要使用SubprocVecEnv进行真正的多进程训练，需要确保：
#   - 所有环境创建函数都在模块顶层
#   - 不使用lambda或闭包捕获复杂对象
# ============================================================

def _make_env_factory(sim_config: Dict[str, Any], rl_config: Dict[str, Any], 
                      log_dir: Optional[str] = None, rank: int = 0):
    """
    创建环境工厂函数（模块级，可被pickle）
    
    Windows兼容性：
    - 此函数必须在模块顶层定义，否则Windows的spawn模式无法pickle
    - 内部的_init函数虽然是闭包，但由于外层函数在模块顶层，整体可pickle
    
    Args:
        sim_config: 仿真配置
        rl_config: RL配置
        log_dir: 日志目录
        rank: 环境编号（用于区分多环境）
    
    Returns:
        环境创建函数
    """
    def _init():
        env = DeliveryRLEnvironment(
            simulation_config=sim_config,
            rl_config=rl_config
        )
        # 使用Monitor包装以记录episode信息
        monitor_path = f"{log_dir}/env_{rank}" if log_dir else None
        return Monitor(env, filename=monitor_path, allow_early_resets=True)
    return _init


# ============================================================
# 自定义回调函数
# ============================================================

class TrainingMonitorCallback(BaseCallback):
    """
    训练监控回调
    
    功能：
    1. 定期记录训练指标到TensorBoard
    2. 打印训练进度
    3. 早停检测
    4. 保存训练历史
    """
    
    def __init__(self, 
                 check_freq: int = 1000,
                 log_dir: Optional[str] = None,
                 early_stop_patience: int = 10,
                 min_improvement: float = 0.01,
                 verbose: int = 1):
        """
        初始化训练监控回调
        
        Args:
            check_freq: 检查频率（步数）
            log_dir: 日志目录
            early_stop_patience: 早停耐心值（评估次数）
            min_improvement: 最小改进阈值
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = Path(log_dir) if log_dir else None
        self.early_stop_patience = early_stop_patience
        self.min_improvement = min_improvement
        
        # 训练历史
        self.history = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'mean_rewards': [],
            'std_rewards': [],
            'completion_rates': [],
            'timeout_rates': []
        }
        
        # 早停相关
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.should_stop = False
        
        # 时间统计
        self.start_time = None
        self.last_check_time = None
    
    def _on_training_start(self) -> None:
        """训练开始时调用"""
        self.start_time = time.time()
        self.last_check_time = self.start_time
        logger.info("="*60)
        logger.info("PPO训练开始")
        logger.info("="*60)
    
    def _on_step(self) -> bool:
        """每步调用"""
        if self.n_calls % self.check_freq == 0:
            self._log_progress()
        
        return not self.should_stop
    
    def _log_progress(self) -> None:
        """记录训练进度"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        interval = current_time - self.last_check_time
        self.last_check_time = current_time
        
        # 获取最近的episode信息
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep['l'] for ep in self.model.ep_info_buffer]
            
            mean_reward = np.mean(ep_rewards)
            std_reward = np.std(ep_rewards)
            mean_length = np.mean(ep_lengths)
            
            # 记录历史
            self.history['timesteps'].append(self.num_timesteps)
            self.history['mean_rewards'].append(mean_reward)
            self.history['std_rewards'].append(std_reward)
            
            # 计算FPS
            fps = self.check_freq / max(interval, 0.001)
            
            # 打印进度
            if self.verbose > 0:
                logger.info(
                    f"[Step {self.num_timesteps:,}] "
                    f"Reward: {mean_reward:.2f}±{std_reward:.2f} | "
                    f"EpLen: {mean_length:.0f} | "
                    f"FPS: {fps:.0f} | "
                    f"Time: {elapsed/60:.1f}min"
                )
            
            # TensorBoard记录
            if self.logger:
                self.logger.record('train/mean_reward', mean_reward)
                self.logger.record('train/std_reward', std_reward)
                self.logger.record('train/mean_ep_length', mean_length)
                self.logger.record('time/fps', fps)
                self.logger.record('time/elapsed_minutes', elapsed / 60)
            
            # 记录业务指标到TensorBoard（从环境的episode_stats中提取）
            # 注意：VecEnv下需要从infos中获取，这里尝试从最近的episode中提取
            try:
                infos = self.locals.get('infos', [])
                for info in infos:
                    if 'episode_stats' in info:
                        stats = info['episode_stats']
                        completion_rate = stats.get('completion_rate', 0)
                        timeout_rate = stats.get('timeout_rate', 0)
                        avg_service_time = stats.get('avg_service_time', 0)
                        
                        # 记录到历史
                        self.history['completion_rates'].append(completion_rate)
                        self.history['timeout_rates'].append(timeout_rate)
                        
                        # 记录到TensorBoard
                        if self.logger:
                            self.logger.record('business/completion_rate', completion_rate)
                            self.logger.record('business/timeout_rate', timeout_rate)
                            self.logger.record('business/avg_service_time', avg_service_time)
                        break  # 只记录第一个有效的episode_stats
            except Exception as e:
                pass  # 静默处理，不影响训练
            
            # 更新最佳奖励记录（与早停逻辑分离）
            self._update_best_mean_reward(mean_reward)
            
            # 早停检查（已禁用，改用课程学习的达标跳转机制）
            # self._check_early_stop(mean_reward)
    
    def _update_best_mean_reward(self, mean_reward: float) -> None:
        """更新最佳平均奖励记录（与早停逻辑分离）"""
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
    
    def _check_early_stop(self, mean_reward: float) -> None:
        """检查是否应该早停"""
        if mean_reward > self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = mean_reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        if self.no_improvement_count >= self.early_stop_patience:
            logger.warning(
                f"早停触发！连续{self.early_stop_patience}次评估无改进 "
                f"(最佳奖励: {self.best_mean_reward:.2f})"
            )
            self.should_stop = True
    
    def _on_training_end(self) -> None:
        """训练结束时调用"""
        total_time = time.time() - self.start_time
        logger.info("="*60)
        logger.info("PPO训练结束")
        logger.info(f"  总步数: {self.num_timesteps:,}")
        logger.info(f"  总耗时: {total_time/60:.1f}分钟")
        logger.info(f"  最佳平均奖励: {self.best_mean_reward:.2f}")
        logger.info("="*60)
        
        # 保存训练历史
        if self.log_dir:
            history_path = self.log_dir / 'training_history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"训练历史已保存: {history_path}")


class EpisodeMetricsCallback(BaseCallback):
    """
    Episode指标回调（可选，目前未默认启用）
    
    记录每个Episode的详细指标（完成率、超时率等）
    
    使用方式：在_create_callbacks中手动添加，或在自定义训练流程中使用：
    ```python
    eval_env = trainer.create_env()
    metrics_callback = EpisodeMetricsCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5
    )
    callbacks.append(metrics_callback)
    ```
    """
    
    def __init__(self, 
                 eval_env: Any = None,
                 eval_freq: int = 5000,
                 n_eval_episodes: int = 5,
                 verbose: int = 1):
        """
        初始化Episode指标回调
        
        Args:
            eval_env: 评估环境
            eval_freq: 评估频率
            n_eval_episodes: 每次评估的Episode数
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # 评估历史
        self.eval_history = {
            'timesteps': [],
            'completion_rates': [],
            'timeout_rates': [],
            'mean_rewards': []
        }
    
    def _on_step(self) -> bool:
        """每步调用"""
        if self.eval_env is not None and self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True
    
    def _evaluate(self) -> None:
        """执行评估"""
        if self.verbose > 0:
            logger.info(f"[Step {self.num_timesteps}] 执行评估...")
        
        completion_rates = []
        timeout_rates = []
        rewards = []
        
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            # 收集Episode统计
            stats = self.eval_env.get_episode_statistics()
            completion_rates.append(stats.get('completion_rate', 0))
            timeout_rates.append(stats.get('timeout_rate', 0))
            rewards.append(episode_reward)
        
        # 计算平均值
        mean_completion = np.mean(completion_rates)
        mean_timeout = np.mean(timeout_rates)
        mean_reward = np.mean(rewards)
        
        # 记录历史
        self.eval_history['timesteps'].append(self.num_timesteps)
        self.eval_history['completion_rates'].append(mean_completion)
        self.eval_history['timeout_rates'].append(mean_timeout)
        self.eval_history['mean_rewards'].append(mean_reward)
        
        # TensorBoard记录
        if self.logger:
            self.logger.record('eval/completion_rate', mean_completion)
            self.logger.record('eval/timeout_rate', mean_timeout)
            self.logger.record('eval/mean_reward', mean_reward)
        
        if self.verbose > 0:
            logger.info(
                f"  评估结果: 完成率={mean_completion:.1%}, "
                f"超时率={mean_timeout:.1%}, 奖励={mean_reward:.2f}"
            )


# ============================================================
# 课程达标跳转回调
# ============================================================

class CurriculumAdvanceCallback(BaseCallback):
    """
    课程达标跳转回调
    
    当模型性能达到当前阶段的阈值时，提前结束当前阶段，进入下一阶段
    """
    
    def __init__(self,
                 eval_env: Any,
                 min_completion_rate: float = 0.5,
                 max_timeout_rate: float = 0.5,
                 eval_freq: int = 5000,
                 n_eval_episodes: int = 3,
                 min_timesteps: int = 10000,
                 verbose: int = 1):
        """
        初始化课程达标跳转回调
        
        Args:
            eval_env: 评估环境
            min_completion_rate: 最小完成率阈值（达到此值才能跳转）
            max_timeout_rate: 最大超时率阈值（低于此值才能跳转）
            eval_freq: 评估频率（步数）
            n_eval_episodes: 每次评估的Episode数
            min_timesteps: 最少训练步数（防止过早跳转）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.min_completion_rate = min_completion_rate
        self.max_timeout_rate = max_timeout_rate
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.min_timesteps = min_timesteps
        
        # 达标状态
        self.stage_completed = False
        self.completion_reason = ""
        self.best_completion_rate = 0.0
        self.best_timeout_rate = 1.0
        
        # 评估历史
        self.eval_history = []
    
    def _on_step(self) -> bool:
        """每步调用，返回False会停止训练"""
        # 定期评估
        if self.n_calls % self.eval_freq == 0 and self.num_timesteps >= self.min_timesteps:
            should_advance = self._evaluate_and_check()
            if should_advance:
                return False  # 停止当前阶段训练
        return True
    
    def _evaluate_and_check(self) -> bool:
        """
        评估模型并检查是否达标
        
        Returns:
            是否应该跳转到下一阶段
        """
        completion_rates = []
        timeout_rates = []
        
        for ep in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            
            # 收集统计
            stats = self.eval_env.get_episode_statistics()
            completion_rates.append(stats.get('completion_rate', 0))
            timeout_rates.append(stats.get('timeout_rate', 1))
        
        # 计算平均值
        mean_completion = np.mean(completion_rates)
        mean_timeout = np.mean(timeout_rates)
        
        # 记录历史
        self.eval_history.append({
            'timesteps': self.num_timesteps,
            'completion_rate': mean_completion,
            'timeout_rate': mean_timeout
        })
        
        # 更新最佳记录
        if mean_completion > self.best_completion_rate:
            self.best_completion_rate = mean_completion
        if mean_timeout < self.best_timeout_rate:
            self.best_timeout_rate = mean_timeout
        
        # TensorBoard记录
        if self.logger:
            self.logger.record('curriculum/completion_rate', mean_completion)
            self.logger.record('curriculum/timeout_rate', mean_timeout)
            self.logger.record('curriculum/target_completion', self.min_completion_rate)
            self.logger.record('curriculum/target_timeout', self.max_timeout_rate)
        
        if self.verbose > 0:
            logger.info(
                f"[Step {self.num_timesteps:,}] 课程评估: "
                f"完成率={mean_completion:.1%} (目标≥{self.min_completion_rate:.1%}), "
                f"超时率={mean_timeout:.1%} (目标≤{self.max_timeout_rate:.1%})"
            )
        
        # 检查是否达标
        if mean_completion >= self.min_completion_rate and mean_timeout <= self.max_timeout_rate:
            self.stage_completed = True
            self.completion_reason = (
                f"达标跳转！完成率={mean_completion:.1%}≥{self.min_completion_rate:.1%}, "
                f"超时率={mean_timeout:.1%}≤{self.max_timeout_rate:.1%}"
            )
            logger.info(f"🎉 {self.completion_reason}")
            return True
        
        return False


# ============================================================
# 简化版课程学习管理器（仅用于train_with_curriculum内部）
# 注意：Day 14的完整版CurriculumManager在curriculum_learning.py中
# ============================================================

class SimpleCurriculumManager:
    """
    简化版课程学习管理器（仅用于train_with_curriculum内部）
    
    管理从低负载到高负载的训练课程。
    如需更完整的功能（回退、平滑过渡等），请使用curriculum_learning.py中的CurriculumManager。
    """
    
    def __init__(self, stages: List[Dict[str, Any]]):
        """
        初始化课程管理器
        
        Args:
            stages: 课程阶段列表，每个阶段包含配置和训练步数
        """
        self.stages = stages
        self.current_stage_idx = 0
        self.total_timesteps_trained = 0
    
    @property
    def current_stage(self) -> Dict[str, Any]:
        """获取当前阶段"""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]
    
    @property
    def is_completed(self) -> bool:
        """是否完成所有阶段"""
        return self.current_stage_idx >= len(self.stages)
    
    def advance(self) -> bool:
        """
        进入下一阶段
        
        Returns:
            是否还有更多阶段
        """
        self.current_stage_idx += 1
        if self.current_stage_idx < len(self.stages):
            logger.info(f"进入课程阶段 {self.current_stage_idx + 1}/{len(self.stages)}")
            logger.info(f"  名称: {self.current_stage['name']}")
            logger.info(f"  订单数: {self.current_stage['total_orders']}")
            logger.info(f"  训练步数: {self.current_stage['timesteps']:,}")
            return True
        return False
    
    def get_stage_config(self) -> Dict[str, Any]:
        """获取当前阶段的环境配置"""
        stage = self.current_stage
        return {
            'total_orders': stage.get('total_orders', 500),
            'num_couriers': stage.get('num_couriers', 20),
            'simulation_duration': stage.get('simulation_duration', 7200)
        }


# ============================================================
# 主训练器类
# ============================================================

class RLTrainer:
    """
    PPO训练流水线
    
    功能：
    1. 加载配置并创建环境
    2. 配置TensorBoard监控
    3. 支持课程学习
    4. 模型保存与恢复
    5. 训练后评估
    """
    
    def __init__(self, config_path: str, scenario: str = None):
        """
        初始化训练器
        
        Args:
            config_path: RL配置文件路径
            scenario: 使用的场景名称（可选，覆盖默认配置）
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.scenario = scenario
        
        # 提取配置
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        self.training_config = self.rl_config.get('training', {})
        self.scenarios_config = self.config.get('scenarios', {})
        
        # 如果指定了场景，使用场景配置覆盖默认配置
        if scenario and scenario in self.scenarios_config:
            scenario_config = self.scenarios_config[scenario]
            self.sim_config.update(scenario_config)
            logger.info(f"使用场景配置: {scenario}")
        
        # 输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_suffix = f"_{scenario}" if scenario else ""
        self.output_dir = Path(self.rl_config.get('model_save_path', './outputs/rl_training')) / f"{timestamp}{scenario_suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard日志目录
        self.tensorboard_dir = self.output_dir / 'tensorboard'
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存使用的配置
        self._save_config()
        
        logger.info(f"训练器初始化完成")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  TensorBoard日志: {self.tensorboard_dir}")
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件已加载: {self.config_path}")
        return config
    
    def _save_config(self) -> None:
        """保存当前使用的配置到输出目录"""
        config_save_path = self.output_dir / 'config_used.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"配置已保存: {config_save_path}")
    
    def create_env(self, **kwargs):
        """
        创建RL环境
        
        Returns:
            RL环境实例
        """
        # 合并配置
        sim_config = {**self.sim_config, **kwargs}
        
        env = DeliveryRLEnvironment(
            simulation_config=sim_config,
            rl_config=self.rl_config
        )
        
        # 添加Monitor包装器（关键修复：用于记录Episode统计）
        # 没有Monitor，ep_info_buffer将为空，导致无法计算平均奖励
        if SB3_AVAILABLE:
            env = Monitor(env, filename=None, allow_early_resets=True)
        
        return env
    
    def train(self, force_single_scenario: bool = False):
        """
        执行训练流程
        
        Args:
            force_single_scenario: 强制使用单场景训练，忽略配置中的use_curriculum_learning
        """
        if not SB3_AVAILABLE:
            logger.error("Stable-Baselines3未安装，无法训练")
            return
        
        # 检查配置是否启用课程学习（除非强制单场景）
        training_strategy = self.training_config.get('training_strategy', {})
        use_curriculum = training_strategy.get('use_curriculum_learning', False)
        
        if use_curriculum and not force_single_scenario:
            logger.info("配置启用了课程学习，自动切换到 train_with_curriculum()")
            return self.train_with_curriculum()
        
        logger.info("="*60)
        logger.info("开始RL训练（单场景模式）")
        logger.info("="*60)
        
        # 1. 创建环境
        logger.info("\n步骤1: 创建训练环境")
        
        # 优先从 training_strategy.num_parallel_envs 读取，兼容旧配置
        training_strategy = self.training_config.get('training_strategy', {})
        num_parallel_envs = training_strategy.get('num_parallel_envs', 
                                                   self.training_config.get('num_parallel_envs', 1))
        
        if num_parallel_envs > 1:
            # 并行环境
            # 使用模块级工厂函数，确保Windows的spawn模式可以pickle
            import platform
            log_dir_str = str(self.output_dir) if self.output_dir else None
            env_fns = [
                _make_env_factory(self.sim_config, self.rl_config, log_dir_str, rank=i)
                for i in range(num_parallel_envs)
            ]
            
            # Windows上SubprocVecEnv可能仍有问题，提供DummyVecEnv回退
            if platform.system() == 'Windows':
                try:
                    env = SubprocVecEnv(env_fns)
                    logger.info(f"  创建了 {num_parallel_envs} 个并行环境 (SubprocVecEnv)")
                except Exception as e:
                    logger.warning(f"SubprocVecEnv创建失败: {e}，回退到DummyVecEnv")
                    env = DummyVecEnv(env_fns)
                    logger.info(f"  创建了 {num_parallel_envs} 个环境 (DummyVecEnv回退)")
            else:
                env = SubprocVecEnv(env_fns)
                logger.info(f"  创建了 {num_parallel_envs} 个并行环境 (SubprocVecEnv)")
        else:
            # 单环境
            env = DummyVecEnv([self.create_env])
            logger.info("  创建了单个训练环境")
        
        # 2. 检查环境（调试用）
        logger.info("\n步骤2: 检查环境兼容性")
        try:
            single_env = self.create_env()
            check_env(single_env)
            logger.info("  ✓ 环境检查通过")
            single_env.close()
        except Exception as e:
            logger.warning(f"  环境检查警告: {str(e)}")
        
        # 3. 创建PPO模型
        logger.info("\n步骤3: 初始化PPO算法")
        model = self._create_ppo_model(env)
        
        logger.info("  算法: PPO")
        logger.info(f"  策略网络: {self.training_config.get('policy', {}).get('net_arch')}")
        
        # 配置TensorBoard logger（与课程学习一致）
        new_logger = configure(str(self.tensorboard_dir), ["stdout", "tensorboard"])
        model.set_logger(new_logger)
        logger.info(f"  TensorBoard日志: {self.tensorboard_dir}")
        
        # 4. 配置回调
        logger.info("\n步骤4: 配置训练回调")
        callbacks = self._create_callbacks(env)
        
        # 5. 开始训练
        logger.info("\n步骤5: 开始训练")
        total_timesteps = self.training_config.get('total_timesteps', 1000000)
        logger.info(f"  总步数: {total_timesteps:,}")
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            logger.info("\n训练完成！")
            
            # 6. 保存最终模型
            final_model_path = self.output_dir / "final_model"
            model.save(final_model_path)
            logger.info(f"最终模型已保存: {final_model_path}")
            
        except KeyboardInterrupt:
            logger.info("\n训练被用户中断")
            interrupt_model_path = self.output_dir / "interrupted_model"
            model.save(interrupt_model_path)
            logger.info(f"中断模型已保存: {interrupt_model_path}")
        
        # 7. 关闭环境
        env.close()
        logger.info("环境已关闭")
    
    def _create_ppo_model(self, env):
        """创建PPO模型"""
        ppo_config = self.training_config.get('ppo', {})
        policy_config = self.training_config.get('policy', {})
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.training_config.get('learning_rate', 3e-4),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=self.training_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.01),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            normalize_advantage=ppo_config.get('normalize_advantage', True),
            policy_kwargs=dict(
                net_arch=policy_config.get('net_arch', [256, 256])
            ),
            verbose=1,
            seed=self.rl_config.get('seed', self.config.get('seed', 42))
        )
        
        return model
    
    
    def _create_callbacks(self, env):
        """创建训练回调"""
        callbacks = []
        
        eval_config = self.rl_config.get('evaluation', {})
        
        # 1. 训练监控回调（自定义）
        monitor_callback = TrainingMonitorCallback(
            check_freq=1000,
            log_dir=str(self.output_dir),
            early_stop_patience=150,  # 150次检查无改进则停止（约150k步）
            min_improvement=10.0,  # 奖励值在2000+范围，提高阈值避免过早停止
            verbose=1
        )
        callbacks.append(monitor_callback)
        logger.info("  ✓ 训练监控回调（每1000步）")
        
        # 2. 标准评估回调
        if eval_config.get('eval_freq', 0) > 0:
            eval_env = DummyVecEnv([self.create_env])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.output_dir / "best_model"),
                log_path=str(self.output_dir / "eval_logs"),
                eval_freq=eval_config.get('eval_freq', 10000),
                n_eval_episodes=eval_config.get('n_eval_episodes', 10),
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            logger.info(f"  ✓ 评估回调（每 {eval_config['eval_freq']} 步）")
        
        # 3. 检查点回调
        if eval_config.get('save_freq', 0) > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=eval_config.get('save_freq', 50000),
                save_path=str(self.output_dir / "checkpoints"),
                name_prefix="rl_model"
            )
            callbacks.append(checkpoint_callback)
            logger.info(f"  ✓ 检查点回调（每 {eval_config['save_freq']} 步）")
        
        return CallbackList(callbacks) if callbacks else None
    
    def train_with_curriculum(self):
        """
        使用课程学习进行训练（达标即跳转模式）
        
        从低负载场景开始，逐步增加难度。
        当模型性能达到当前阶段的阈值时，立即跳转到下一阶段。
        
        注意：本方法是简化版课程学习实现，直接读取config中的curriculum_stages。
        如需更完整的课程学习功能（回退机制、平滑过渡、难度维度等），
        请使用 src/rl/train_with_curriculum.py 中的 CurriculumTrainer，
        它基于 src/rl/curriculum_learning.py 中的 CurriculumManager。
        
        两套实现的区别：
        - 本方法：简单、直接读取YAML配置、支持达标跳转和加时赛
        - CurriculumTrainer：完整框架、支持回退/平滑过渡/难度评分等高级功能
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3未安装，无法训练")
        
        # 课程学习使用PPO算法
        
        # 获取课程学习配置
        curriculum_config = self.training_config.get('curriculum', {})
        stages = curriculum_config.get('curriculum_stages', [])
        
        if not stages:
            logger.warning("未配置课程学习阶段，使用默认训练")
            return self.train()
        
        logger.info("="*60)
        logger.info("开始课程学习训练（达标即跳转模式）")
        logger.info(f"共 {len(stages)} 个阶段")
        logger.info("="*60)
        
        # 创建初始模型
        model = None
        total_stages_completed = 0
        current_env = None  # 跟踪当前环境，用于阶段切换时关闭
        
        for stage_idx, stage in enumerate(stages):
            # 打印阶段信息
            logger.info(f"\n{'='*60}")
            logger.info(f"课程阶段 {stage_idx + 1}/{len(stages)}: {stage['name']}")
            logger.info(f"{'='*60}")
            logger.info(f"  描述: {stage.get('description', 'N/A')}")
            logger.info(f"  订单文件: {stage.get('orders_file', self.sim_config.get('orders_file'))}")
            logger.info(f"  订单数: {stage['total_orders']}")
            logger.info(f"  骑手数: {stage.get('num_couriers', 20)}")
            logger.info(f"  最大训练步数: {stage['timesteps']:,}")
            logger.info(f"  达标条件: 完成率≥{stage.get('min_completion_rate', 0.5):.0%}, "
                       f"超时率≤{stage.get('max_timeout_rate', 0.5):.0%}")
            
            # 创建该阶段的环境配置
            stage_num_couriers = stage.get('num_couriers', 20)
            
            # 验证骑手数不超过max_couriers（动作空间上限）
            # 这是课程学习的关键约束：所有阶段必须使用相同的动作空间维度
            max_couriers = self.rl_config.get('state_encoder', {}).get('max_couriers', 50)
            if stage_num_couriers > max_couriers:
                raise ValueError(
                    f"课程阶段'{stage['name']}'的num_couriers={stage_num_couriers}超过了"
                    f"state_encoder.max_couriers={max_couriers}。"
                    f"请减少该阶段的骑手数，或增大max_couriers配置。"
                )
            
            stage_sim_config = {
                **self.sim_config,
                'total_orders': stage['total_orders'],
                'num_couriers': stage_num_couriers,
                'simulation_duration': stage.get('simulation_duration', self.sim_config.get('simulation_duration', 43200)),
                # 使用阶段特定的订单文件（如果配置了的话）
                'orders_file': stage.get('orders_file', self.sim_config.get('orders_file'))
            }
            
            # 关闭上一阶段的环境（避免SubprocVecEnv子进程泄露）
            if current_env is not None:
                try:
                    current_env.close()
                    logger.debug("已关闭上一阶段的训练环境")
                except Exception as e:
                    logger.warning(f"关闭上一阶段环境时出错: {e}")
            
            # 支持多环境并行（与train()一致）
            training_strategy = self.training_config.get('training_strategy', {})
            num_parallel_envs = training_strategy.get('num_parallel_envs', 
                                                       self.training_config.get('num_parallel_envs', 1))
            
            # 使用模块级工厂函数，确保Windows的spawn模式可以pickle
            import platform
            log_dir_str = str(self.output_dir) if self.output_dir else None
            
            if num_parallel_envs > 1:
                env_fns = [
                    _make_env_factory(stage_sim_config, self.rl_config, log_dir_str, rank=i)
                    for i in range(num_parallel_envs)
                ]
                
                # Windows上SubprocVecEnv可能有问题，提供DummyVecEnv回退
                if platform.system() == 'Windows':
                    try:
                        env = SubprocVecEnv(env_fns)
                        logger.info(f"  创建了 {num_parallel_envs} 个并行环境 (SubprocVecEnv)")
                    except Exception as e:
                        logger.warning(f"SubprocVecEnv创建失败: {e}，回退到DummyVecEnv")
                        env = DummyVecEnv(env_fns)
                        logger.info(f"  创建了 {num_parallel_envs} 个环境 (DummyVecEnv回退)")
                else:
                    env = SubprocVecEnv(env_fns)
                    logger.info(f"  创建了 {num_parallel_envs} 个并行环境 (SubprocVecEnv)")
            else:
                # 单环境使用工厂函数
                env = DummyVecEnv([_make_env_factory(stage_sim_config, self.rl_config, log_dir_str, rank=0)])
            
            # 更新当前环境引用（用于下一阶段关闭）
            current_env = env
            
            # 创建评估环境（用于达标检测，不需要Monitor）
            eval_env = DeliveryRLEnvironment(
                simulation_config=stage_sim_config,
                rl_config=self.rl_config
            )
            
            # 创建或更新模型
            if model is None:
                model = self._create_ppo_model(env)
                # 配置TensorBoard
                new_logger = configure(str(self.tensorboard_dir), ["stdout", "tensorboard"])
                model.set_logger(new_logger)
            else:
                # 更新环境
                model.set_env(env)
            
            # 创建回调列表
            callbacks = []
            
            # 1. 训练监控回调（不含早停）
            monitor_callback = TrainingMonitorCallback(
                check_freq=1000,
                log_dir=str(self.output_dir),
                early_stop_patience=9999,  # 实际上禁用早停
                min_improvement=0.01,
                verbose=1
            )
            callbacks.append(monitor_callback)
            
            # 2. 课程达标跳转回调（核心）
            # 从配置读取评估参数（影响达标判断的稳定性）
            eval_freq = curriculum_config.get('eval_freq', 5000)
            n_eval_episodes = curriculum_config.get('n_eval_episodes', 5)  # 默认5，减少波动
            
            curriculum_callback = CurriculumAdvanceCallback(
                eval_env=eval_env,
                min_completion_rate=stage.get('min_completion_rate', 0.5),
                max_timeout_rate=stage.get('max_timeout_rate', 0.5),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                min_timesteps=10000,  # 至少训练1万步才检测
                verbose=1
            )
            callbacks.append(curriculum_callback)
            
            # 3. 检查点回调
            checkpoint_callback = CheckpointCallback(
                save_freq=20000,
                save_path=str(self.output_dir / "checkpoints"),
                name_prefix=f"stage_{stage_idx + 1}"
            )
            callbacks.append(checkpoint_callback)
            
            callback_list = CallbackList(callbacks)
            
            # 训练该阶段（含弹性延长机制）
            stage_start_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
            max_retries = curriculum_config.get('max_retries', 2)  # 从配置读取
            retry_count = 0
            extra_timesteps = curriculum_config.get('extra_timesteps', 50000)  # 从配置读取
            failure_strategy = curriculum_config.get('failure_strategy', 'stop')  # 失败策略
            
            while retry_count <= max_retries:
                current_timesteps = stage['timesteps'] if retry_count == 0 else extra_timesteps
                
                if retry_count > 0:
                    logger.info(f"⚠️ 阶段 {stage_idx + 1} 未达标，进入第 {retry_count} 次加时赛 (+{extra_timesteps}步)...")
                
                try:
                    model.learn(
                        total_timesteps=current_timesteps,
                        callback=callback_list,
                        reset_num_timesteps=False,
                        progress_bar=True
                    )
                    
                    # 1. 检查是否达标
                    if curriculum_callback.stage_completed:
                        logger.info(f"✅ 阶段 {stage_idx + 1} 达标完成: {curriculum_callback.completion_reason}")
                        break # 退出重试循环，进入下一阶段
                    
                    # 2. 如果未达标，检查是否值得加时
                    best_rate = curriculum_callback.best_completion_rate
                    target_rate = stage.get('min_completion_rate', 0.5)
                    overtime_threshold = curriculum_config.get('overtime_threshold', 0.8)
                    threshold = target_rate * overtime_threshold  # 从配置读取容忍度
                    
                    if retry_count < max_retries:
                        if best_rate >= threshold:
                            logger.info(f"📈 当前最佳完成率 {best_rate:.1%} 接近目标 {target_rate:.1%}，触发自动加时")
                            retry_count += 1
                            continue
                        else:
                            logger.error(f"❌ 阶段 {stage_idx + 1} 训练失败！最佳完成率 {best_rate:.1%} 远低于目标 {target_rate:.1%}")
                            logger.error("建议：调整课程难度或检查模型参数")
                            if failure_strategy == 'stop':
                                return model  # 终止训练
                            else:
                                logger.warning(f"⚠️ failure_strategy='continue'，跳过阶段 {stage_idx + 1}，继续后续阶段")
                                break  # 跳出加时循环，继续下一阶段
                    else:
                        logger.error(f"❌ 阶段 {stage_idx + 1} 加时赛耗尽仍未达标")
                        if failure_strategy == 'stop':
                            return model
                        else:
                            logger.warning(f"⚠️ failure_strategy='continue'，跳过阶段 {stage_idx + 1}，继续后续阶段")
                            break
                        
                except KeyboardInterrupt:
                    logger.info(f"\n阶段 {stage_idx + 1} 被用户中断")
                    interrupt_path = self.output_dir / f"interrupted_stage_{stage_idx + 1}"
                    model.save(interrupt_path)
                    logger.info(f"中断模型已保存: {interrupt_path}")
                    return model
            
            # 保存阶段模型（只有达标或跳出循环后才保存）
            stage_model_path = self.output_dir / f"stage_{stage_idx + 1}_{stage['name']}"
            model.save(stage_model_path)
            logger.info(f"阶段模型已保存: {stage_model_path}")
            
            total_stages_completed += 1
            
            # 无论如何，关闭评估环境
            eval_env.close()
        
        # 保存最终模型
        final_model_path = self.output_dir / "final_curriculum_model"
        model.save(final_model_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"课程学习完成！")
        logger.info(f"  完成阶段数: {total_stages_completed}/{len(stages)}")
        logger.info(f"  最终模型: {final_model_path}")
        logger.info(f"{'='*60}")
        
        # 关闭最后一个训练环境
        if current_env is not None:
            try:
                current_env.close()
                logger.debug("已关闭最终训练环境")
            except Exception as e:
                logger.warning(f"关闭最终环境时出错: {e}")
        
        return model
    
    def evaluate_model(self, model_path: str = None, n_episodes: int = 10):
        """
        评估训练好的模型
        
        模型路径检测逻辑：
        1. 如果指定了model_path，直接使用
        2. 否则自动检测：final_curriculum_model.zip > final_model.zip > 默认final_model
        3. 这样无论是课程学习还是单场景训练，都能正确找到模型
        
        环境说明：
        - 评估使用非向量环境（单环境 + Monitor包装）
        - 与训练时的VecEnv略有差异，但SB3的predict对1D/2D obs都能处理
        - obs shape保持为(state_dim,)，与训练时一致
        - 如需完全对称，可改用DummyVecEnv包装，但会增加代码复杂度
        
        Args:
            model_path: 模型路径（None则自动检测最新训练的模型）
            n_episodes: 评估Episode数
        
        Returns:
            评估结果字典，包含completion_rate, timeout_rate, mean_reward等
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3未安装")
        
        # 加载模型
        # 注意：train()保存到final_model，train_with_curriculum()保存到final_curriculum_model
        # 自动检测哪个存在
        if model_path is None:
            curriculum_model = self.output_dir / "final_curriculum_model.zip"
            single_model = self.output_dir / "final_model.zip"
            
            if curriculum_model.exists():
                model_path = self.output_dir / "final_curriculum_model"
                logger.info("检测到课程学习模型")
            elif single_model.exists():
                model_path = self.output_dir / "final_model"
                logger.info("检测到单场景模型")
            else:
                # 默认尝试final_model（兼容旧行为）
                model_path = self.output_dir / "final_model"
                logger.warning("未找到训练模型，尝试加载默认路径")
        
        logger.info(f"加载模型: {model_path}")
        
        # 加载PPO模型
        # 注意：这里没有传env给load()，因为评估只需要policy进行predict
        # 如果后续需要继续训练（.learn()）或获取环境（.get_env()），需要先调用model.set_env()
        model = PPO.load(model_path)
        
        # 创建评估环境（独立于模型，仅用于采集轨迹）
        env = self.create_env()
        
        # 运行评估
        results = {
            'episode_rewards': [],
            'completion_rates': [],
            'timeout_rates': [],
            'episode_lengths': []
        }
        
        logger.info(f"运行 {n_episodes} 个评估Episode...")
        
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
            
            logger.info(
                f"  Episode {ep + 1}: "
                f"reward={episode_reward:.2f}, "
                f"完成率={stats.get('completion_rate', 0):.1%}, "
                f"超时率={stats.get('timeout_rate', 0):.1%}"
            )
        
        # 汇总结果
        summary = {
            'mean_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'mean_completion_rate': np.mean(results['completion_rates']),
            'mean_timeout_rate': np.mean(results['timeout_rates']),
            'mean_episode_length': np.mean(results['episode_lengths'])
        }
        
        logger.info("\n评估结果汇总:")
        logger.info(f"  平均奖励: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        logger.info(f"  平均完成率: {summary['mean_completion_rate']:.1%}")
        logger.info(f"  平均超时率: {summary['mean_timeout_rate']:.1%}")
        
        # 保存评估结果
        eval_path = self.output_dir / 'evaluation_results.json'
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump({**results, **summary}, f, indent=2)
        logger.info(f"评估结果已保存: {eval_path}")
        
        env.close()
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Day 13: PPO训练流水线 - 即时配送强化学习调度'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/rl_config.yaml',
        help='RL配置文件路径'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='low_load',
        choices=['low_load', 'medium_load', 'high_load', 'extreme_load', 'low_stress'],
        help='训练场景 (default: low_load)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='覆盖配置文件中的训练步数'
    )
    parser.add_argument(
        '--test-env',
        action='store_true',
        help='仅测试环境兼容性'
    )
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help='使用课程学习（从低负载到高负载）'
    )
    parser.add_argument(
        '--evaluate',
        type=str,
        default=None,
        help='评估已训练模型（指定模型路径）'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式（减少训练步数）'
    )
    
    args = parser.parse_args()
    
    # 调试模式配置
    if args.debug:
        logger.info("调试模式：减少训练步数")
        args.timesteps = args.timesteps or 5000
    
    if args.test_env:
        # 仅测试环境
        print("="*60)
        print("测试RL环境兼容性")
        print("="*60)
        
        config_path = Path(args.config)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 使用低负载场景测试
        sim_config = config.get('simulation', {})
        sim_config.update(config.get('scenarios', {}).get('low_load', {}))
        
        env = DeliveryRLEnvironment(
            simulation_config=sim_config,
            rl_config=config.get('rl', {})
        )
        
        print(f"\n观测空间: {env.observation_space}")
        print(f"  形状: {env.observation_space.shape}")
        print(f"动作空间: {env.action_space}")
        # 根据动作空间类型显示不同信息
        if isinstance(env.action_space, spaces.Discrete):
            print(f"  大小: {env.action_space.n}")
        elif isinstance(env.action_space, spaces.MultiDiscrete):
            print(f"  维度: {env.action_space.nvec}  (每维动作数)")
        elif isinstance(env.action_space, spaces.Box):
            print(f"  形状: {env.action_space.shape}")
        else:
            print(f"  类型: {type(env.action_space)}")
        
        # 检查环境
        if SB3_AVAILABLE:
            print("\n运行Stable-Baselines3环境检查...")
            try:
                check_env(env)
                print("✓ 环境检查通过")
            except Exception as e:
                print(f"⚠ 环境检查警告: {e}")
        
        # 运行几个步骤
        print("\n运行测试步骤...")
        obs, info = env.reset()
        print(f"初始观测形状: {obs.shape}")
        print(f"初始信息: {info}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"\n步骤 {i+1}:")
            print(f"  动作: {action}")
            print(f"  奖励: {reward:.4f}")
            print(f"  终止: {terminated}, 截断: {truncated}")
        
        env.close()
        print("\n✓ 环境测试完成！")
        return
    
    if args.evaluate:
        # 评估模式
        print("="*60)
        print("评估已训练模型")
        print("="*60)
        
        trainer = RLTrainer(args.config, scenario=args.scenario)
        trainer.evaluate_model(model_path=args.evaluate)
        return
    
    # 训练模式
    print("="*60)
    print("Day 13: PPO训练流水线")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"场景: {args.scenario}")
    print(f"课程学习: {args.curriculum}")
    print(f"调试模式: {args.debug}")
    
    trainer = RLTrainer(args.config, scenario=args.scenario)
    
    # 覆盖训练步数
    if args.timesteps:
        trainer.training_config['total_timesteps'] = args.timesteps
        logger.info(f"训练步数已覆盖为: {args.timesteps:,}")
    
    # 课程学习：CLI参数优先，否则从配置文件读取
    training_strategy = trainer.training_config.get('training_strategy', {})
    use_curriculum = args.curriculum or training_strategy.get('use_curriculum_learning', False)
    
    if use_curriculum:
        # 课程学习训练
        trainer.train_with_curriculum()
    else:
        # 标准训练
        trainer.train()
    
    print("\n训练完成！")
    print(f"输出目录: {trainer.output_dir}")
    print(f"TensorBoard命令: tensorboard --logdir={trainer.tensorboard_dir}")


if __name__ == "__main__":
    main()

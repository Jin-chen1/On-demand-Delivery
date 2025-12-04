"""
Day 15: 超参数调优模块 (Hyperparameter Tuning)

功能：
1. Reward权重敏感性分析
2. 超参数网格搜索
3. 训练曲线分析
4. 最优配置推荐
"""

import numpy as np
import pandas as pd
import json
import logging
import time
import yaml
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .rl_environment import DeliveryRLEnvironment


@dataclass 
class TuningResult:
    """调优结果"""
    config_id: str
    config: Dict[str, Any]
    mean_reward: float
    std_reward: float
    mean_completion_rate: float
    mean_timeout_rate: float
    training_time_seconds: float
    converged: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_id': self.config_id,
            **{f'config_{k}': v for k, v in self.config.items()},
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'mean_completion_rate': self.mean_completion_rate,
            'mean_timeout_rate': self.mean_timeout_rate,
            'training_time_seconds': self.training_time_seconds,
            'converged': self.converged
        }


class QuickEvalCallback(BaseCallback):
    """快速评估回调，用于超参数搜索"""
    
    def __init__(self, eval_freq: int = 5000, n_eval: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval = n_eval
        self.rewards = []
        self.completion_rates = []
        self.timeout_rates = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                self.rewards.append(np.mean(rewards))
        return True


class RewardWeightTuner:
    """Reward权重调优器"""
    
    def __init__(self, config_path: str, output_dir: str = None, quick_mode: bool = True):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.quick_mode = quick_mode
        
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(f'./outputs/day15_tuning/{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RewardWeightTuner initialized")
    
    def get_weight_search_space(self) -> Dict[str, List[float]]:
        """获取权重搜索空间"""
        if self.quick_mode:
            return {
                'weight_timeout_penalty': [5.0, 10.0, 20.0],
                'weight_distance_cost': [0.0005, 0.001, 0.002],
                'weight_completion_bonus': [3.0, 5.0, 10.0]
            }
        return {
            'weight_timeout_penalty': [5.0, 10.0, 15.0, 20.0, 30.0],
            'weight_distance_cost': [0.0005, 0.001, 0.002, 0.005],
            'weight_wait_time': [0.005, 0.01, 0.02],
            'weight_completion_bonus': [3.0, 5.0, 10.0, 15.0],
            'weight_balanced_load': [0.5, 1.0, 2.0]
        }
    
    def run_sensitivity_analysis(self, base_weights: Dict = None,
                                 training_steps: int = 20000,
                                 n_eval_episodes: int = 3) -> pd.DataFrame:
        """
        运行Reward权重敏感性分析
        单独变化每个权重参数，观察对性能的影响
        """
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not installed")
        
        if base_weights is None:
            base_weights = {
                'weight_timeout_penalty': 10.0,
                'weight_distance_cost': 0.001,
                'weight_wait_time': 0.01,
                'weight_completion_bonus': 5.0,
                'weight_balanced_load': 1.0
            }
        
        search_space = self.get_weight_search_space()
        
        logger.info("="*60)
        logger.info("Day 15: Reward Weight Sensitivity Analysis")
        logger.info(f"Base weights: {base_weights}")
        logger.info(f"Training steps per config: {training_steps:,}")
        logger.info("="*60)
        
        results = []
        
        # 先评估基准配置
        logger.info("\nEvaluating baseline config...")
        baseline_result = self._train_and_evaluate(
            'baseline', base_weights, training_steps, n_eval_episodes
        )
        results.append(baseline_result)
        
        # 对每个参数进行敏感性分析
        for param_name, param_values in search_space.items():
            logger.info(f"\nAnalyzing: {param_name}")
            
            for value in param_values:
                if value == base_weights.get(param_name):
                    continue  # 跳过基准值
                
                # 创建配置
                weights = base_weights.copy()
                weights[param_name] = value
                config_id = f"{param_name}={value}"
                
                logger.info(f"  Testing {config_id}...")
                result = self._train_and_evaluate(
                    config_id, weights, training_steps, n_eval_episodes
                )
                results.append(result)
        
        df = pd.DataFrame([r.to_dict() for r in results])
        self._save_sensitivity_results(df)
        return df
    
    def _train_and_evaluate(self, config_id: str, weights: Dict,
                            training_steps: int, n_eval: int) -> TuningResult:
        """训练并评估单个配置"""
        start_time = time.time()
        
        # 更新RL配置中的奖励权重
        rl_config = self.rl_config.copy()
        if 'reward_calculator' not in rl_config:
            rl_config['reward_calculator'] = {}
        rl_config['reward_calculator'].update(weights)
        
        # 使用低负载场景进行快速训练
        sim_config = self.sim_config.copy()
        sim_config['total_orders'] = 300 if self.quick_mode else 500
        sim_config['num_couriers'] = 20
        sim_config['simulation_duration'] = 3600 if self.quick_mode else 7200
        
        # 创建环境
        def make_env():
            return DeliveryRLEnvironment(sim_config, rl_config)
        
        env = DummyVecEnv([make_env])
        
        # 创建模型
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=512 if self.quick_mode else 2048,
            batch_size=32 if self.quick_mode else 64,
            verbose=0
        )
        
        # 训练
        callback = QuickEvalCallback(eval_freq=2000)
        try:
            model.learn(total_timesteps=training_steps, callback=callback)
        except Exception as e:
            logger.error(f"Training failed for {config_id}: {e}")
            return TuningResult(
                config_id=config_id, config=weights,
                mean_reward=-1000, std_reward=0,
                mean_completion_rate=0, mean_timeout_rate=1,
                training_time_seconds=time.time()-start_time, converged=False
            )
        
        # 评估
        rewards, completion_rates, timeout_rates = [], [], []
        eval_env = make_env()
        
        for _ in range(n_eval):
            obs, _ = eval_env.reset()
            done, ep_reward = False, 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
            
            # 获取统计信息
            sim_env = eval_env.sim_env
            if sim_env:
                total = len(sim_env.orders)
                completed = len([o for o in sim_env.orders.values() 
                               if hasattr(o, 'status') and 'DELIVERED' in str(o.status)])
                timeout = len([o for o in sim_env.orders.values() 
                              if hasattr(o, 'status') and 'TIMEOUT' in str(o.status)])
                completion_rates.append(completed / max(total, 1))
                timeout_rates.append(timeout / max(total, 1))
        
        eval_env.close()
        env.close()
        
        # 判断是否收敛
        converged = len(callback.rewards) > 2 and callback.rewards[-1] > callback.rewards[0]
        
        training_time = time.time() - start_time
        
        result = TuningResult(
            config_id=config_id, config=weights,
            mean_reward=np.mean(rewards), std_reward=np.std(rewards),
            mean_completion_rate=np.mean(completion_rates) if completion_rates else 0,
            mean_timeout_rate=np.mean(timeout_rates) if timeout_rates else 1,
            training_time_seconds=training_time, converged=converged
        )
        
        logger.info(f"    reward={result.mean_reward:.2f}, "
                   f"completion={result.mean_completion_rate:.1%}, "
                   f"time={training_time:.1f}s")
        
        return result
    
    def _save_sensitivity_results(self, df: pd.DataFrame):
        """保存敏感性分析结果"""
        df.to_csv(self.output_dir / 'sensitivity_analysis.csv', index=False)
        
        # 生成报告
        report_path = self.output_dir / 'sensitivity_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Day 15: Reward Weight Sensitivity Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write(f"```\n{df.to_string()}\n```\n\n")
            
            f.write("## Recommendations\n\n")
            best_idx = df['mean_reward'].idxmax()
            best = df.loc[best_idx]
            f.write(f"- **Best config**: {best['config_id']}\n")
            f.write(f"- **Mean reward**: {best['mean_reward']:.2f}\n")
            f.write(f"- **Completion rate**: {best['mean_completion_rate']:.1%}\n")
        
        logger.info(f"Sensitivity report saved: {report_path}")


class HyperparameterTuner:
    """超参数网格搜索调优器"""
    
    def __init__(self, config_path: str, output_dir: str = None, quick_mode: bool = True):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.quick_mode = quick_mode
        
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(f'./outputs/day15_hp_tuning/{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_hyperparameter_grid(self) -> Dict[str, List[Any]]:
        """获取超参数搜索网格"""
        if self.quick_mode:
            return {
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'gamma': [0.95, 0.99],
                'ent_coef': [0.01, 0.02]
            }
        return {
            'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],
            'gamma': [0.95, 0.99, 0.995],
            'n_steps': [1024, 2048, 4096],
            'batch_size': [32, 64, 128],
            'ent_coef': [0.005, 0.01, 0.02],
            'clip_range': [0.1, 0.2, 0.3]
        }
    
    def run_grid_search(self, training_steps: int = 30000,
                        n_eval_episodes: int = 3,
                        max_configs: int = 20) -> pd.DataFrame:
        """运行超参数网格搜索"""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not installed")
        
        grid = self.get_hyperparameter_grid()
        
        # 生成所有配置组合
        keys = list(grid.keys())
        values = list(grid.values())
        all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # 限制配置数量
        if len(all_configs) > max_configs:
            np.random.shuffle(all_configs)
            all_configs = all_configs[:max_configs]
        
        logger.info("="*60)
        logger.info("Day 15: Hyperparameter Grid Search")
        logger.info(f"Total configs to test: {len(all_configs)}")
        logger.info(f"Training steps per config: {training_steps:,}")
        logger.info("="*60)
        
        results = []
        
        for i, hp_config in enumerate(all_configs):
            config_id = f"config_{i+1}"
            logger.info(f"\n[{i+1}/{len(all_configs)}] Testing {hp_config}")
            
            result = self._train_with_hyperparams(
                config_id, hp_config, training_steps, n_eval_episodes
            )
            results.append(result)
        
        df = pd.DataFrame([r.to_dict() for r in results])
        self._save_grid_search_results(df)
        return df
    
    def _train_with_hyperparams(self, config_id: str, hp_config: Dict,
                                training_steps: int, n_eval: int) -> TuningResult:
        """使用指定超参数训练"""
        start_time = time.time()
        
        sim_config = self.sim_config.copy()
        sim_config['total_orders'] = 300 if self.quick_mode else 500
        sim_config['num_couriers'] = 20
        
        def make_env():
            return DeliveryRLEnvironment(sim_config, self.rl_config)
        
        env = DummyVecEnv([make_env])
        
        # 使用配置创建模型
        model = PPO(
            "MlpPolicy", env,
            learning_rate=hp_config.get('learning_rate', 3e-4),
            gamma=hp_config.get('gamma', 0.99),
            n_steps=hp_config.get('n_steps', 2048),
            batch_size=hp_config.get('batch_size', 64),
            ent_coef=hp_config.get('ent_coef', 0.01),
            clip_range=hp_config.get('clip_range', 0.2),
            verbose=0
        )
        
        try:
            model.learn(total_timesteps=training_steps)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TuningResult(
                config_id=config_id, config=hp_config,
                mean_reward=-1000, std_reward=0,
                mean_completion_rate=0, mean_timeout_rate=1,
                training_time_seconds=time.time()-start_time, converged=False
            )
        
        # 评估
        rewards, completion_rates, timeout_rates = [], [], []
        eval_env = make_env()
        
        for _ in range(n_eval):
            obs, _ = eval_env.reset()
            done, ep_reward = False, 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            rewards.append(ep_reward)
            sim_env = eval_env.sim_env
            if sim_env:
                total = len(sim_env.orders)
                completed = len([o for o in sim_env.orders.values() 
                               if hasattr(o, 'status') and 'DELIVERED' in str(o.status)])
                timeout = len([o for o in sim_env.orders.values() 
                              if hasattr(o, 'status') and 'TIMEOUT' in str(o.status)])
                completion_rates.append(completed / max(total, 1))
                timeout_rates.append(timeout / max(total, 1))
        
        eval_env.close()
        env.close()
        
        training_time = time.time() - start_time
        
        result = TuningResult(
            config_id=config_id, config=hp_config,
            mean_reward=np.mean(rewards), std_reward=np.std(rewards),
            mean_completion_rate=np.mean(completion_rates) if completion_rates else 0,
            mean_timeout_rate=np.mean(timeout_rates) if timeout_rates else 1,
            training_time_seconds=training_time, converged=True
        )
        
        logger.info(f"  reward={result.mean_reward:.2f}, completion={result.mean_completion_rate:.1%}")
        return result
    
    def _save_grid_search_results(self, df: pd.DataFrame):
        """保存网格搜索结果"""
        df.to_csv(self.output_dir / 'grid_search_results.csv', index=False)
        
        # 找出最佳配置
        best_idx = df['mean_reward'].idxmax()
        best = df.loc[best_idx]
        
        # 保存最佳配置
        best_config = {col.replace('config_', ''): best[col] 
                      for col in df.columns if col.startswith('config_')}
        
        with open(self.output_dir / 'best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # 生成报告
        with open(self.output_dir / 'grid_search_report.md', 'w', encoding='utf-8') as f:
            f.write("# Day 15: Hyperparameter Grid Search Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Best Configuration\n\n")
            f.write(f"```json\n{json.dumps(best_config, indent=2)}\n```\n\n")
            f.write(f"- Mean reward: {best['mean_reward']:.2f}\n")
            f.write(f"- Completion rate: {best['mean_completion_rate']:.1%}\n\n")
            
            f.write("## All Results\n\n")
            f.write(f"```\n{df.sort_values('mean_reward', ascending=False).to_string()}\n```\n")
        
        logger.info(f"Grid search results saved: {self.output_dir}")

"""
Day 15: 模型评估与超参数调优模块 (Evaluation & Tuning)

功能：
1. 在测试集上评估模型性能（多场景：低/中/高负载、暴雨等）
2. 调整Reward权重（超时惩罚系数、距离成本等）敏感性分析
3. 超参数网格搜索优化
4. 生成评估报告和可视化图表
"""

import numpy as np
import pandas as pd
import json
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# RL训练库
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .rl_environment import DeliveryRLEnvironment
from .baseline_agents import create_baseline_agent, run_baseline_episode
from .utils import extract_simulation_metrics


@dataclass
class EvaluationResult:
    """单次评估结果"""
    scenario_name: str
    agent_type: str
    episode: int
    total_reward: float
    completion_rate: float
    timeout_rate: float
    total_orders: int
    completed_orders: int
    timeout_orders: int
    avg_service_time: float = 0.0
    total_distance: float = 0.0  # 新增：总配送距离
    episode_length: int = 0
    wall_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class TestScenarios:
    """测试场景定义"""
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, Dict[str, Any]]:
        return {
            'low_load': {
                'name': 'Low Load', 'description': 'order/courier=25',
                'total_orders': 500, 'num_couriers': 20,
                'simulation_duration': 7200, 'time_window_multiplier': 1.2
            },
            'medium_load': {
                'name': 'Medium Load (MVP)', 'description': 'order/courier=50',
                'total_orders': 1000, 'num_couriers': 20,
                'simulation_duration': 7200, 'time_window_multiplier': 1.0
            },
            'high_load': {
                'name': 'High Load', 'description': 'order/courier=75',
                'total_orders': 1500, 'num_couriers': 20,
                'simulation_duration': 7200, 'time_window_multiplier': 0.9
            },
            'extreme_load': {
                'name': 'Extreme Load', 'description': 'order/courier=100',
                'total_orders': 2000, 'num_couriers': 20,
                'simulation_duration': 7200, 'time_window_multiplier': 0.8
            },
            'high_capacity': {
                'name': 'High Capacity', 'description': 'order/courier=25 with more couriers',
                'total_orders': 1000, 'num_couriers': 40,
                'simulation_duration': 7200, 'time_window_multiplier': 1.0
            },
            'rainy_weather': {
                'name': 'Rainy Weather', 'description': 'speed reduced 30%',
                'total_orders': 1000, 'num_couriers': 20,
                'simulation_duration': 7200, 'speed_multiplier': 0.7
            }
        }
    
    @staticmethod
    def get_quick_scenarios() -> Dict[str, Dict[str, Any]]:
        scenarios = TestScenarios.get_all_scenarios()
        quick = {}
        for name, cfg in scenarios.items():
            c = cfg.copy()
            c['total_orders'] = min(cfg['total_orders'], 200)
            c['simulation_duration'] = min(cfg['simulation_duration'], 3600)
            quick[name] = c
        return quick


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path: str, model_path: str = None,
                 output_dir: str = None, quick_mode: bool = False):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.model_path = Path(model_path) if model_path else None
        self.quick_mode = quick_mode
        
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(f'./outputs/day15_eval/{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenarios = TestScenarios.get_quick_scenarios() if quick_mode else TestScenarios.get_all_scenarios()
        self.results = []
        
        logger.info(f"ModelEvaluator initialized: {len(self.scenarios)} scenarios")
    
    def create_env(self, scenario_config: Dict) -> DeliveryRLEnvironment:
        sim_config = {**self.sim_config, **scenario_config}
        return DeliveryRLEnvironment(sim_config, self.rl_config)
    
    def evaluate_rl_model(self, scenario_name: str, scenario_config: Dict,
                          n_episodes: int = 5) -> List[EvaluationResult]:
        if not self.model_path or not SB3_AVAILABLE:
            return []
        
        model = PPO.load(self.model_path)
        results = []
        
        for ep in range(n_episodes):
            start = time.time()
            env = self.create_env(scenario_config)
            obs, _ = env.reset()
            done, total_reward, steps = False, 0.0, 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            # 使用公共工具函数提取指标
            metrics = extract_simulation_metrics(env.sim_env)

            results.append(EvaluationResult(
                scenario_name=scenario_name, agent_type='rl_ppo', episode=ep+1,
                total_reward=total_reward, 
                completion_rate=metrics['completion_rate'],
                timeout_rate=metrics['timeout_rate'], 
                total_orders=metrics['total_orders'],
                completed_orders=metrics['completed_orders'], 
                timeout_orders=metrics['timeout_orders'],
                avg_service_time=metrics['avg_service_time'],
                total_distance=metrics['total_distance'],
                episode_length=steps, wall_time_seconds=time.time()-start
            ))
            env.close()
        
        return results
    
    def evaluate_baseline(self, agent_type: str, scenario_name: str,
                          scenario_config: Dict, n_episodes: int = 5) -> List[EvaluationResult]:
        results = []
        for ep in range(n_episodes):
            start = time.time()
            try:
                env = self.create_env(scenario_config)
                agent = create_baseline_agent(agent_type, env)
                ep_result = run_baseline_episode(agent, env, max_steps=500, verbose=False)
                
                results.append(EvaluationResult(
                    scenario_name=scenario_name, agent_type=agent_type, episode=ep+1,
                    total_reward=ep_result.get('total_reward', 0),
                    completion_rate=ep_result.get('completion_rate', 0),
                    timeout_rate=ep_result.get('timeout_rate', 0),
                    total_orders=ep_result.get('total_orders', 0),
                    completed_orders=ep_result.get('completed_orders', 0),
                    timeout_orders=ep_result.get('timeout_orders', 0),
                    avg_service_time=ep_result.get('avg_service_time', 0.0),
                    total_distance=ep_result.get('total_distance', 0.0),
                    episode_length=ep_result.get('step_count', 0),
                    wall_time_seconds=time.time()-start
                ))
                env.close()
            except Exception as e:
                logger.error(f"Baseline {agent_type} failed: {e}")
        return results
    
    def run_evaluation(self, agent_types: List[str] = None,
                       scenario_names: List[str] = None,
                       n_episodes: int = 3) -> pd.DataFrame:
        """运行完整评估"""
        if agent_types is None:
            agent_types = ['greedy', 'ortools', 'alns']
            if self.model_path and SB3_AVAILABLE:
                agent_types.insert(0, 'rl_ppo')
        
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        logger.info("="*60)
        logger.info("Day 15: Model Evaluation Started")
        logger.info(f"Agents: {agent_types}, Scenarios: {scenario_names}")
        logger.info("="*60)
        
        all_results = []
        
        for scenario_name in scenario_names:
            if scenario_name not in self.scenarios:
                continue
            scenario_config = self.scenarios[scenario_name]
            logger.info(f"\nScenario: {scenario_config.get('name', scenario_name)}")
            
            for agent_type in agent_types:
                logger.info(f"  Evaluating {agent_type}...")
                if agent_type == 'rl_ppo':
                    results = self.evaluate_rl_model(scenario_name, scenario_config, n_episodes)
                else:
                    results = self.evaluate_baseline(agent_type, scenario_name, scenario_config, n_episodes)
                all_results.extend(results)
                
                if results:
                    avg_completion = np.mean([r.completion_rate for r in results])
                    avg_timeout = np.mean([r.timeout_rate for r in results])
                    logger.info(f"    completion={avg_completion:.1%}, timeout={avg_timeout:.1%}")
        
        df = pd.DataFrame([r.to_dict() for r in all_results])
        self._save_results(df)
        return df
    
    def _save_results(self, df: pd.DataFrame):
        df.to_csv(self.output_dir / 'evaluation_results.csv', index=False)
        df.to_json(self.output_dir / 'evaluation_results.json', orient='records', indent=2)
        self._generate_report(df)
    
    def _generate_report(self, df: pd.DataFrame):
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Day 15: Model Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary by Scenario and Agent\n\n")
            summary = df.groupby(['scenario_name', 'agent_type']).agg({
                'completion_rate': ['mean', 'std'],
                'timeout_rate': ['mean', 'std'],
                'total_reward': 'mean'
            }).round(4)
            f.write(f"```\n{summary.to_string()}\n```\n\n")
            
            f.write("## Best Agent per Scenario (by completion rate)\n\n")
            for scenario in df['scenario_name'].unique():
                sdata = df[df['scenario_name'] == scenario].groupby('agent_type')['completion_rate'].mean()
                best = sdata.idxmax()
                f.write(f"- **{scenario}**: {best} ({sdata.max():.1%})\n")
        
        logger.info(f"Report saved: {report_path}")

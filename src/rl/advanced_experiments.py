"""
Day 16: 高级实验模块 (Advanced Experiments)

功能：
1. 最终对比实验（RL vs ALNS vs OR-Tools vs Greedy）
2. 多场景鲁棒性测试（暴雨、爆单、正常、极端负载）
3. 统计分析和置信区间计算
4. 结果汇总和报告生成
"""

import numpy as np
import pandas as pd
import json
import logging
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
import traceback

logger = logging.getLogger(__name__)

# RL训练库
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from .rl_environment import DeliveryRLEnvironment
from .baseline_agents import create_baseline_agent, run_baseline_episode
from .utils import extract_simulation_metrics


@dataclass
class ExperimentResult:
    """单次实验结果"""
    scenario_name: str
    scenario_type: str  # normal, rain, surge, extreme
    agent_type: str
    run_id: int
    seed: int
    
    # 核心指标
    total_orders: int = 0
    completed_orders: int = 0
    timeout_orders: int = 0
    completion_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # 效率指标
    avg_service_time: float = 0.0
    total_distance: float = 0.0
    avg_courier_utilization: float = 0.0
    
    # RL特定指标
    total_reward: float = 0.0
    episode_length: int = 0
    
    # 运行时间
    wall_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RobustnessScenarios:
    """鲁棒性测试场景定义"""
    
    @staticmethod
    def get_all_scenarios() -> Dict[str, Dict[str, Any]]:
        """获取所有测试场景配置"""
        return {
            # 正常场景
            'normal_low': {
                'name': 'Normal - Low Load', 'type': 'normal',
                'description': 'Normal conditions, order/courier=25',
                'total_orders': 500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 1.0,
            },
            'normal_medium': {
                'name': 'Normal - Medium Load', 'type': 'normal',
                'description': 'Normal conditions, order/courier=50',
                'total_orders': 1000, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 1.0,
            },
            'normal_high': {
                'name': 'Normal - High Load', 'type': 'normal',
                'description': 'Normal conditions, order/courier=75',
                'total_orders': 1500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 1.0,
            },
            # 暴雨场景
            'rain_low': {
                'name': 'Rain - Low Load', 'type': 'rain',
                'description': 'Rainy weather (speed -30%)',
                'total_orders': 500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 0.7,
            },
            'rain_medium': {
                'name': 'Rain - Medium Load', 'type': 'rain',
                'description': 'Rainy weather (speed -30%)',
                'total_orders': 1000, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 0.7,
            },
            'rain_high': {
                'name': 'Rain - High Load', 'type': 'rain',
                'description': 'Rainy weather (speed -30%)',
                'total_orders': 1500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 1.0, 'speed_multiplier': 0.7,
            },
            # 爆单场景
            'surge_50': {
                'name': 'Order Surge +50%', 'type': 'surge',
                'description': 'Order surge (+50%)',
                'total_orders': 1500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 0.9, 'speed_multiplier': 1.0,
                'peak_rate_multiplier': 2.0,
            },
            'surge_100': {
                'name': 'Order Surge +100%', 'type': 'surge',
                'description': 'Order surge (+100%)',
                'total_orders': 2000, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 0.8, 'speed_multiplier': 1.0,
                'peak_rate_multiplier': 2.5,
            },
            # 极端场景
            'extreme_rain_surge': {
                'name': 'Extreme: Rain + Surge', 'type': 'extreme',
                'description': 'Rain + 50% order surge',
                'total_orders': 1500, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 0.9, 'speed_multiplier': 0.7,
                'peak_rate_multiplier': 2.0,
            },
            'extreme_max_stress': {
                'name': 'Extreme: Max Stress', 'type': 'extreme',
                'description': 'Heavy rain + double orders',
                'total_orders': 2000, 'num_couriers': 20,
                'simulation_duration': 7200,
                'time_window_multiplier': 0.7, 'speed_multiplier': 0.6,
                'peak_rate_multiplier': 3.0,
            },
        }
    
    @staticmethod
    def get_quick_scenarios() -> Dict[str, Dict[str, Any]]:
        """快速测试场景"""
        all_scenarios = RobustnessScenarios.get_all_scenarios()
        quick = {}
        for name, cfg in all_scenarios.items():
            c = cfg.copy()
            c['total_orders'] = min(cfg['total_orders'], 100)
            c['simulation_duration'] = min(cfg['simulation_duration'], 1800)
            quick[name] = c
        return quick


class AdvancedExperimenter:
    """高级实验管理器"""
    
    def __init__(self, config_path: str, rl_model_path: str = None,
                 output_dir: str = None, quick_mode: bool = False):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.rl_model_path = Path(rl_model_path) if rl_model_path else None
        self.quick_mode = quick_mode
        
        self.sim_config = self.config.get('simulation', {})
        self.rl_config = self.config.get('rl', {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) if output_dir else Path(f'./outputs/day16_advanced/{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scenarios = RobustnessScenarios.get_quick_scenarios() if quick_mode else RobustnessScenarios.get_all_scenarios()
        self.results: List[ExperimentResult] = []
        
        self.available_agents = ['greedy', 'ortools', 'alns']
        if self.rl_model_path and self.rl_model_path.exists() and SB3_AVAILABLE:
            self.available_agents.insert(0, 'rl_ppo')
        
        logger.info(f"AdvancedExperimenter: {len(self.scenarios)} scenarios, agents: {self.available_agents}")
    
    def _create_env(self, scenario_config: Dict[str, Any]) -> DeliveryRLEnvironment:
        sim_config = {**self.sim_config, **scenario_config}
        return DeliveryRLEnvironment(sim_config, self.rl_config)
    
    def _run_single_experiment(self, scenario_name: str, scenario_config: Dict[str, Any],
                               agent_type: str, run_id: int, seed: int) -> ExperimentResult:
        """运行单次实验"""
        start_time = time.time()
        result = ExperimentResult(
            scenario_name=scenario_name,
            scenario_type=scenario_config.get('type', 'unknown'),
            agent_type=agent_type, run_id=run_id, seed=seed
        )
        
        try:
            env = self._create_env(scenario_config)
            
            if agent_type == 'rl_ppo':
                if not self.rl_model_path or not SB3_AVAILABLE:
                    raise RuntimeError("RL model not available")
                model = PPO.load(self.rl_model_path)
                obs, _ = env.reset(seed=seed)
                done, total_reward, steps = False, 0.0, 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated
                result.total_reward = total_reward
                result.episode_length = steps
            else:
                agent = create_baseline_agent(agent_type, env)
                ep_result = run_baseline_episode(agent, env, max_steps=500, verbose=False)
                result.total_reward = ep_result.get('total_reward', 0)
                result.episode_length = ep_result.get('step_count', 0)
            
            metrics = extract_simulation_metrics(env.sim_env)
            result.total_orders = metrics['total_orders']
            result.completed_orders = metrics['completed_orders']
            result.timeout_orders = metrics['timeout_orders']
            result.completion_rate = metrics['completion_rate']
            result.timeout_rate = metrics['timeout_rate']
            result.avg_service_time = metrics['avg_service_time']
            result.total_distance = metrics['total_distance']
            env.close()
        except Exception as e:
            logger.error(f"Experiment failed [{scenario_name}][{agent_type}][{run_id}]: {e}")
        
        result.wall_time_seconds = time.time() - start_time
        return result
    
    def run_comparison_experiment(self, agent_types: List[str] = None,
                                   scenario_names: List[str] = None,
                                   n_runs: int = 5, base_seed: int = 42) -> pd.DataFrame:
        """运行算法对比实验"""
        if agent_types is None:
            agent_types = self.available_agents
        if scenario_names is None:
            scenario_names = list(self.scenarios.keys())
        
        logger.info("="*70)
        logger.info("Day 16: Advanced Comparison Experiment")
        logger.info(f"  Agents: {agent_types}, Scenarios: {len(scenario_names)}, Runs: {n_runs}")
        logger.info("="*70)
        
        all_results = []
        for scenario_name in scenario_names:
            if scenario_name not in self.scenarios:
                continue
            scenario_config = self.scenarios[scenario_name]
            logger.info(f"\n[Scenario] {scenario_config.get('name', scenario_name)}")
            
            for agent_type in agent_types:
                if agent_type not in self.available_agents:
                    continue
                logger.info(f"  [Agent] {agent_type}...")
                
                for run_id in range(n_runs):
                    seed = base_seed + run_id
                    result = self._run_single_experiment(
                        scenario_name, scenario_config, agent_type, run_id + 1, seed
                    )
                    all_results.append(result)
                
                agent_results = [r for r in all_results 
                                if r.scenario_name == scenario_name and r.agent_type == agent_type]
                avg_comp = np.mean([r.completion_rate for r in agent_results])
                avg_timeout = np.mean([r.timeout_rate for r in agent_results])
                logger.info(f"    Avg: completion={avg_comp:.1%}, timeout={avg_timeout:.1%}")
        
        self.results = all_results
        df = pd.DataFrame([r.to_dict() for r in all_results])
        self._save_results(df)
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """保存实验结果"""
        df.to_csv(self.output_dir / 'experiment_results.csv', index=False)
        df.to_json(self.output_dir / 'experiment_results.json', orient='records', indent=2)
        self._generate_summary_report(df)
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """生成汇总报告"""
        report_path = self.output_dir / 'summary_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Day 16: Advanced Experiments Summary Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Overview\n\n")
            f.write(f"- Total experiments: {len(df)}\n")
            f.write(f"- Scenarios: {df['scenario_name'].nunique()}\n")
            f.write(f"- Agents: {df['agent_type'].nunique()}\n\n")
            
            f.write("## Performance by Agent\n\n")
            agent_summary = df.groupby('agent_type').agg({
                'completion_rate': ['mean', 'std'],
                'timeout_rate': ['mean', 'std'],
                'avg_service_time': 'mean'
            }).round(4)
            f.write(f"```\n{agent_summary.to_string()}\n```\n\n")
            
            f.write("## Performance by Scenario Type\n\n")
            for scenario_type in ['normal', 'rain', 'surge', 'extreme']:
                type_df = df[df['scenario_type'] == scenario_type]
                if type_df.empty:
                    continue
                f.write(f"### {scenario_type.upper()}\n\n")
                type_summary = type_df.groupby('agent_type').agg({
                    'completion_rate': ['mean', 'std'],
                    'timeout_rate': ['mean', 'std']
                }).round(4)
                f.write(f"```\n{type_summary.to_string()}\n```\n\n")
            
            f.write("## Best Agent per Scenario\n\n")
            for scenario in df['scenario_name'].unique():
                sdata = df[df['scenario_name'] == scenario].groupby('agent_type')['completion_rate'].mean()
                if not sdata.empty:
                    best = sdata.idxmax()
                    f.write(f"- **{scenario}**: {best} ({sdata.max():.1%})\n")
            
            f.write("\n## Robustness Analysis\n\n")
            for agent_type in df['agent_type'].unique():
                agent_df = df[df['agent_type'] == agent_type]
                normal_perf = agent_df[agent_df['scenario_type'] == 'normal']['completion_rate'].mean()
                stress_perf = agent_df[agent_df['scenario_type'].isin(['rain', 'surge', 'extreme'])]['completion_rate'].mean()
                if normal_perf > 0:
                    degradation = (normal_perf - stress_perf) / normal_perf * 100
                    f.write(f"- **{agent_type}**: {degradation:.1f}% degradation under stress\n")
        
        logger.info(f"Report saved: {report_path}")


def run_day16_experiments(config_path: str, rl_model_path: str = None,
                          output_dir: str = None, quick_mode: bool = False,
                          agent_types: List[str] = None, n_runs: int = 5) -> pd.DataFrame:
    """Day 16主实验入口函数"""
    experimenter = AdvancedExperimenter(
        config_path=config_path,
        rl_model_path=rl_model_path,
        output_dir=output_dir,
        quick_mode=quick_mode
    )
    return experimenter.run_comparison_experiment(
        agent_types=agent_types,
        n_runs=n_runs
    )


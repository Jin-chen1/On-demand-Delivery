"""
Baseline Agents - Day 12 实现
将Greedy、OR-Tools、ALNS调度器封装为统一的Agent接口，
使其能在相同的Gym接口下运行，收集Benchmark数据

遵循Stable-Baselines3的Agent接口规范：
- predict(observation, state=None, deterministic=True) -> (action, state)
"""

import logging
import time
import numpy as np
from typing import Tuple, Optional, Any, Dict, List
from gymnasium import spaces

from .utils import extract_simulation_metrics

logger = logging.getLogger(__name__)


class RandomAgent:
    """
    随机Agent - 最低基准
    随机选择动作，用于对比其他策略的基础性能
    """
    
    def __init__(self, action_space: spaces.Space):
        """
        初始化随机Agent
        
        Args:
            action_space: 动作空间（从RL环境获取）
        """
        self.action_space = action_space
        self.agent_type = 'random'
        logger.info(f"RandomAgent初始化完成，动作空间: {action_space}")
    
    def predict(self, 
                observation: np.ndarray,
                state: Optional[Any] = None,
                deterministic: bool = False) -> Tuple[Any, Optional[Any]]:
        """
        预测动作（随机采样）
        
        Args:
            observation: 当前观测（不使用）
            state: 隐藏状态（不使用）
            deterministic: 是否确定性（随机Agent忽略此参数）
        
        Returns:
            (action, state): 采样的动作和None状态
        """
        action = self.action_space.sample()
        return action, state
    
    def reset(self):
        """重置Agent状态（随机Agent无需重置）"""
        pass


class BaselineAgent:
    """
    基线Agent包装器
    将现有调度器(Greedy/OR-Tools/ALNS)封装为统一的Agent接口
    
    工作模式：
    1. 接收observation（但实际通过访问仿真环境获取状态）
    2. 调用内部调度器计算最佳分配
    3. 将分配决策转换为等效的离散动作
    4. 返回action供环境执行
    
    注意：BaselineAgent需要访问RL环境的仿真环境实例
    """
    
    def __init__(self, 
                 agent_type: str,
                 rl_env: Any = None,
                 dispatcher_config: Optional[Dict[str, Any]] = None):
        """
        初始化基线Agent
        
        Args:
            agent_type: Agent类型 ('greedy', 'ortools', 'alns')
            rl_env: DeliveryRLEnvironment实例
            dispatcher_config: 调度器配置（可选）
        """
        self.agent_type = agent_type.lower()
        self.rl_env = rl_env
        self.dispatcher_config = dispatcher_config or {}
        self.dispatcher = None
        
        # 统计信息
        self.dispatch_count = 0
        self.total_dispatch_time = 0.0
        
        # 验证agent类型
        valid_types = ['greedy', 'ortools', 'alns']
        if self.agent_type not in valid_types:
            raise ValueError(f"无效的agent_type: {agent_type}，支持: {valid_types}")
        
        logger.info(f"BaselineAgent初始化: {self.agent_type}")
    
    def set_env(self, rl_env: Any):
        """
        设置RL环境引用
        
        Args:
            rl_env: DeliveryRLEnvironment实例
        """
        self.rl_env = rl_env
        self.dispatcher = None  # 重置调度器，下次predict时重新创建
        logger.info(f"BaselineAgent [{self.agent_type}] 已绑定到RL环境")
    
    def _ensure_dispatcher(self):
        """确保调度器已创建"""
        if self.dispatcher is not None:
            return
        
        if self.rl_env is None or self.rl_env.sim_env is None:
            raise RuntimeError("RL环境未初始化，请先调用env.reset()")
        
        sim_env = self.rl_env.sim_env
        
        if self.agent_type == 'greedy':
            from ..simulation.dispatchers.greedy_dispatcher import GreedyDispatcher
            self.dispatcher = GreedyDispatcher(sim_env)
            
        elif self.agent_type == 'ortools':
            from ..simulation.dispatchers.ortools_dispatcher import ORToolsDispatcher
            config = self.dispatcher_config.get('ortools', {
                'time_limit_seconds': 3,
                'enable_batching': True,
                'max_batch_size': 10
            })
            self.dispatcher = ORToolsDispatcher(sim_env, config)
            
        elif self.agent_type == 'alns':
            from ..simulation.dispatchers.alns_dispatcher import ALNSDispatcher
            config = self.dispatcher_config.get('alns', {
                'max_iterations': 50,
                'time_limit_seconds': 2
            })
            self.dispatcher = ALNSDispatcher(sim_env, config)
        
        logger.debug(f"调度器 [{self.agent_type}] 创建完成")
    
    def predict(self,
                observation: np.ndarray,
                state: Optional[Any] = None,
                deterministic: bool = True) -> Tuple[Any, Optional[Any]]:
        """
        预测动作
        
        调用内部调度器处理待分配订单，然后返回一个dummy action。
        实际的派单工作由调度器直接在仿真环境中完成。
        
        Args:
            observation: 当前观测
            state: 隐藏状态（不使用）
            deterministic: 是否确定性（基线策略总是确定性的）
        
        Returns:
            (action, state): 动作（0表示延迟/无操作）和None状态
        """
        start_time = time.time()
        
        try:
            # 确保调度器已创建
            self._ensure_dispatcher()
            
            # 获取仿真环境
            sim_env = self.rl_env.sim_env
            
            # 执行调度（调度器直接修改仿真环境状态）
            dispatched_count = self.dispatcher.dispatch_pending_orders()
            
            # 更新统计
            self.dispatch_count += 1
            elapsed = time.time() - start_time
            self.total_dispatch_time += elapsed
            
            logger.debug(
                f"[{self.agent_type}] 调度完成: "
                f"分配 {dispatched_count} 个订单, 耗时 {elapsed*1000:.1f}ms"
            )
            
            # 返回延迟动作（0），表示不通过RL环境执行额外分配
            # 实际分配已由调度器完成
            action = 0
            
            return action, state
            
        except Exception as e:
            logger.error(f"[{self.agent_type}] 调度失败: {e}")
            # 返回延迟动作，让环境自然推进
            return 0, state
    
    def reset(self):
        """
        重置Agent状态
        在新Episode开始时调用
        """
        self.dispatcher = None  # 下次predict时重新创建
        self.dispatch_count = 0
        self.total_dispatch_time = 0.0
        logger.debug(f"BaselineAgent [{self.agent_type}] 已重置")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取Agent统计信息
        
        Returns:
            统计信息字典
        """
        avg_time = (self.total_dispatch_time / self.dispatch_count * 1000 
                   if self.dispatch_count > 0 else 0)
        return {
            'agent_type': self.agent_type,
            'dispatch_count': self.dispatch_count,
            'total_dispatch_time_ms': self.total_dispatch_time * 1000,
            'avg_dispatch_time_ms': avg_time
        }


def create_baseline_agent(agent_type: str, 
                          rl_env: Any = None,
                          dispatcher_config: Optional[Dict[str, Any]] = None) -> Any:
    """
    工厂函数：创建基线Agent
    
    Args:
        agent_type: Agent类型 ('random', 'greedy', 'ortools', 'alns')
        rl_env: DeliveryRLEnvironment实例（random类型可不传）
        dispatcher_config: 调度器配置
    
    Returns:
        Agent实例
    
    Example:
        >>> env = DeliveryRLEnvironment(sim_config, rl_config)
        >>> agent = create_baseline_agent('greedy', env)
        >>> obs, info = env.reset()
        >>> action, _ = agent.predict(obs)
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'random':
        if rl_env is None:
            raise ValueError("创建RandomAgent需要传入rl_env以获取动作空间")
        return RandomAgent(rl_env.action_space)
    
    elif agent_type in ['greedy', 'ortools', 'alns']:
        agent = BaselineAgent(agent_type, rl_env, dispatcher_config)
        return agent
    
    else:
        raise ValueError(
            f"未知的agent_type: {agent_type}，"
            f"支持: random, greedy, ortools, alns"
        )


def run_baseline_episode(agent: Any,
                         env: Any,
                         max_steps: int = 1000,
                         verbose: bool = False) -> Dict[str, Any]:
    """
    运行单个Episode并收集指标
    
    Args:
        agent: Agent实例（RandomAgent或BaselineAgent）
        env: DeliveryRLEnvironment实例
        max_steps: 最大步数
        verbose: 是否输出详细日志
    
    Returns:
        Episode统计信息字典
    """
    # 重置环境和Agent
    obs, info = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()
    if hasattr(agent, 'set_env'):
        agent.set_env(env)
    
    # 统计变量
    total_reward = 0.0
    step_count = 0
    step_times = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and step_count < max_steps:
        step_start = time.time()
        
        # Agent预测动作
        action, _ = agent.predict(obs, deterministic=True)
        
        # 环境执行
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        total_reward += reward
        step_count += 1
        
        if verbose and step_count % 50 == 0:
            logger.info(f"  Step {step_count}: reward={reward:.2f}, cumulative={total_reward:.2f}")
    
    # 收集最终统计
    episode_stats = info.get('episode_stats', {})
    
    # 使用公共工具函数提取仿真环境指标
    metrics = extract_simulation_metrics(env.sim_env)
    
    # Agent统计
    agent_stats = {}
    if hasattr(agent, 'get_statistics'):
        agent_stats = agent.get_statistics()
    
    return {
        'agent_type': getattr(agent, 'agent_type', 'unknown'),
        'total_reward': total_reward,
        'step_count': step_count,
        'avg_step_time_ms': np.mean(step_times) * 1000 if step_times else 0,
        'total_orders': metrics['total_orders'],
        'completed_orders': metrics['completed_orders'],
        'timeout_orders': metrics['timeout_orders'],
        'completion_rate': metrics['completion_rate'],
        'timeout_rate': metrics['timeout_rate'],
        'avg_service_time': metrics['avg_service_time'],
        'total_distance': metrics['total_distance'],
        'terminated': terminated,
        'truncated': truncated,
        **agent_stats,
        **episode_stats
    }


def benchmark_agents(env_config: Dict[str, Any],
                     rl_config: Dict[str, Any],
                     agent_types: List[str] = None,
                     num_episodes: int = 3,
                     max_steps: int = 500,
                     verbose: bool = True) -> List[Dict[str, Any]]:
    """
    对多个Agent进行Benchmark测试
    
    Args:
        env_config: 仿真环境配置
        rl_config: RL配置
        agent_types: 要测试的Agent类型列表
        num_episodes: 每个Agent运行的Episode数
        max_steps: 每个Episode的最大步数
        verbose: 是否输出详细日志
    
    Returns:
        所有运行结果的列表
    """
    from .rl_environment import DeliveryRLEnvironment
    
    if agent_types is None:
        agent_types = ['random', 'greedy', 'ortools', 'alns']
    
    all_results = []
    
    for agent_type in agent_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {agent_type.upper()}")
        logger.info(f"{'='*60}")
        
        for episode in range(num_episodes):
            logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            try:
                # 每个Episode创建新环境（确保独立性）
                env = DeliveryRLEnvironment(env_config, rl_config)
                
                # 创建Agent
                agent = create_baseline_agent(agent_type, env)
                
                # 运行Episode
                result = run_baseline_episode(
                    agent, env, 
                    max_steps=max_steps, 
                    verbose=verbose
                )
                result['episode'] = episode + 1
                
                all_results.append(result)
                
                if verbose:
                    logger.info(
                        f"结果: 完成率={result['completion_rate']:.1%}, "
                        f"超时率={result['timeout_rate']:.1%}, "
                        f"总奖励={result['total_reward']:.2f}"
                    )
                
                # 清理
                env.close()
                
            except Exception as e:
                logger.error(f"[{agent_type}] Episode {episode + 1} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    return all_results

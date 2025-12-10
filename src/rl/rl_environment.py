"""
强化学习环境接口
实现Gym-like接口，将仿真环境包装为标准RL训练环境

符合Gymnasium (Gym) API标准，可与Stable-Baselines3等库无缝集成
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import networkx as nx
import json

from .state_representation import StateEncoder
from .reward_function import RewardCalculator
from ..simulation.environment import SimulationEnvironment

logger = logging.getLogger(__name__)


class DeliveryRLEnvironment(gym.Env):
    """
    即时配送RL环境
    
    遵循Gymnasium API:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    - render() (可选)
    - close()
    
    动作空间设计（分层决策）：
    Level 1 (RL控制): 派单决策
      - 为每个待分配订单选择：分配给哪个骑手 or 延迟派单
    
    Level 2 (启发式): 路径优化
      - 使用ALNS/贪婪插入优化骑手路线（不需要RL）
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 simulation_config: Dict[str, Any],
                 rl_config: Optional[Dict[str, Any]] = None):
        """
        初始化RL环境
        
        Args:
            simulation_config: 仿真环境配置（路网、订单、骑手等）
            rl_config: RL特定配置（状态编码、奖励函数等）
        """
        super().__init__()
        
        self.simulation_config = simulation_config
        self.rl_config = rl_config or {}
        
        # 初始化状态编码器
        state_config = self.rl_config.get('state_encoder', {})
        self.state_encoder = StateEncoder(state_config)
        
        # 初始化奖励计算器
        # 根据use_shaping配置选择是否使用Potential-based Reward Shaping
        reward_config = self.rl_config.get('reward_calculator', {})
        use_shaping = reward_config.get('use_shaping', False)
        
        if use_shaping:
            from .reward_function import ShapingRewardCalculator
            self.reward_calculator = ShapingRewardCalculator(reward_config)
            logger.info("使用ShapingRewardCalculator (Potential-based Reward Shaping)")
        else:
            self.reward_calculator = RewardCalculator(reward_config)
            logger.info("使用标准RewardCalculator")
        
        # 动作空间设计
        # 优先从 action_space.mode 读取，兼容旧配置的 action_mode
        action_space_config = self.rl_config.get('action_space', {})
        self.action_mode = action_space_config.get('mode', self.rl_config.get('action_mode', 'discrete'))
        self.max_pending_orders = state_config.get('max_pending_orders', 50)
        # 关键修复：动作空间使用固定的最大骑手数，确保课程学习不同阶段动作空间一致
        # 使用state_encoder中的max_couriers（默认50），而非当前阶段的实际骑手数
        # 当选择不存在的骑手时，_try_assign_with_fallback会自动回退到其他可用骑手
        self.max_couriers = state_config.get('max_couriers', 50)
        
        # 路径优化策略配置
        # 选项: 'greedy_insert' (贪婪最优插入), 'alns' (ALNS 2-opt优化), 'fifo' (简单追加)
        self.routing_strategy = self.rl_config.get('routing_strategy', 'greedy_insert')
        # ALNS路径优化参数
        self.alns_max_iterations = self.rl_config.get('alns_max_iterations', 10)
        
        self._define_spaces()
        
        # 仿真环境实例（延迟初始化）
        self.sim_env: Optional[SimulationEnvironment] = None
        
        # Episode统计
        self.current_episode = 0
        self.total_steps = 0
        self.episode_rewards = []
        
        # 状态缓存
        self._current_state = None
        self._previous_state = None
        
        # === 性能优化：预加载静态数据，避免reset时重复IO ===
        # 路网、距离矩阵等静态数据在__init__时加载一次，reset时重用
        self._cached_static_data = {}
        self._preload_static_data()
        
        logger.info("DeliveryRLEnvironment初始化完成")
        logger.info(f"  观测空间维度: {self.observation_space.shape}")
        logger.info(f"  动作空间: {self.action_space}")
        logger.info(f"  路径优化策略: {self.routing_strategy}")
    
    def _define_spaces(self):
        """定义观测空间和动作空间"""
        # 观测空间：连续向量
        state_dim = self.state_encoder.get_state_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # 动作空间设计
        if self.action_mode == 'discrete':
            # 离散动作：为单个订单选择骑手
            # 动作 = [0, num_couriers]: 0表示延迟派单，1-num_couriers表示分配给对应骑手
            num_actions = self.max_couriers + 1  # +1 for "delay" action
            self.action_space = spaces.Discrete(num_actions)
            
        elif self.action_mode == 'multi_discrete':
            # 多离散动作：同时为多个订单决策
            # 适用于批量派单场景
            action_dims = [self.max_couriers + 1] * self.max_pending_orders
            self.action_space = spaces.MultiDiscrete(action_dims)
            
        elif self.action_mode == 'continuous':
            # 连续动作模式尚未实现，禁止使用
            raise NotImplementedError(
                "continuous 动作模式尚未实现，请改用 'discrete' 或 'multi_discrete'。"
                "如需使用连续动作空间，请先实现 _execute_continuous_action 方法。"
            )
            # 预留代码（实现后取消注释）：
            # 连续动作：输出派单概率分布
            # 维度：max_pending_orders × (max_couriers + 1)
            action_dim = self.max_pending_orders * (self.max_couriers + 1)
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32
            )
        
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境，开始新的Episode
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        logger.info(f"重置环境，开始 Episode {self.current_episode + 1}")
        
        # 重新创建仿真环境（确保每个Episode独立）
        self._create_simulation_environment()
        
        # 重置奖励计算器
        self.reward_calculator.reset()
        
        # 获取初始观测
        observation = self._get_observation()
        
        # 初始化状态缓存
        self._current_state = self._get_full_state()
        self._previous_state = None
        
        # 初始化跟踪变量
        self._last_completed_count = 0
        self.episode_start_time = self.sim_env.env.now if self.sim_env else 0.0
        self.episode_steps = 0
        self.episode_rewards = []
        
        # Day 27: 延迟派单追踪（用于delay_justified奖励）
        # 记录每个订单在延迟时的最佳可用骑手距离
        # 格式: {order_id: {'delay_time': float, 'best_distance_at_delay': float}}
        self._delay_tracking: Dict[int, Dict[str, float]] = {}
        
        # 关键检查：实际骑手数不能超过max_couriers，否则动作空间无法覆盖所有骑手
        actual_couriers = len(self.sim_env.couriers) if self.sim_env else 0
        if actual_couriers > self.max_couriers:
            raise ValueError(
                f"实际骑手数 {actual_couriers} 超过 state_encoder.max_couriers={self.max_couriers}，"
                f"请增大 max_couriers 或减少场景中的 num_couriers。"
            )
        
        # 关键检查：骑手ID必须是从1到num_couriers的连续整数
        # 动作空间设计依赖这个假设：action=1对应courier_id=1，action=2对应courier_id=2...
        if self.sim_env and actual_couriers > 0:
            expected_ids = set(range(1, actual_couriers + 1))
            actual_ids = set(self.sim_env.couriers.keys())
            if actual_ids != expected_ids:
                raise ValueError(
                    f"骑手ID必须是从1到{actual_couriers}的连续整数，"
                    f"当前keys={sorted(actual_ids)}，期望keys={sorted(expected_ids)}。"
                    f"请检查SimulationEnvironment的骑手初始化逻辑。"
                )
        
        info = {
            'episode': self.current_episode,
            # 使用实际加载的订单数和骑手数，而非配置值（可能不一致）
            'total_orders': len(self.sim_env.orders) if self.sim_env else 0,
            'num_couriers': actual_couriers
        }
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作，推进环境一步
        
        Args:
            action: RL Agent的动作
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.sim_env is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        # 保存前一状态
        self._previous_state = self._current_state
        
        # 执行动作（派单决策）
        action_info = self._execute_action(action)
        
        # 推进仿真（到下一个调度时刻）
        self._advance_simulation()
        
        # 获取新状态
        self._current_state = self._get_full_state()
        observation = self._get_observation()
        
        # 获取步骤信息（只调用一次，避免completed_orders被重复消费）
        step_info = self._get_step_info()
        
        # 计算奖励
        reward = self.reward_calculator.calculate_step_reward(
            self._previous_state,
            action_info,
            self._current_state,
            step_info
        )
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self._check_truncation()
        
        # 更新统计
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_rewards.append(reward)
        
        # 构建info（复用step_info，避免二次调用）
        info = {**step_info, 'action_info': action_info, 'step': self.total_steps}
        
        # 如果episode结束，添加统计信息
        if terminated or truncated:
            info['episode_stats'] = self.get_episode_statistics()
            self.current_episode += 1
        
        return observation, reward, terminated, truncated, info
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        获取Episode统计信息
        
        包含RL训练指标和业务指标：
        - RL指标：total_reward, episode_steps, average_reward
        - 业务指标：completion_rate, timeout_rate, avg_service_time等
        
        Returns:
            Episode统计信息
        """
        # 基础RL统计
        episode_stats = {
            'episode': self.current_episode,
            'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0.0,
            'episode_steps': self.episode_steps,
            'average_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            # 注意：移除了无意义的average_steps（永远等于1，因为episode_steps == len(episode_rewards)）
        }
        
        # 从仿真环境获取业务指标
        if self.sim_env:
            # 订单统计
            total_orders = len(self.sim_env.orders)
            completed_count = len(self.sim_env.completed_orders)
            timeout_count = self.sim_env.stats.get('timeout_orders', 0)
            
            # 计算平均服务时间（送达时间 - 到达时间）
            service_times = []
            for order_id in self.sim_env.completed_orders:
                order = self.sim_env.orders.get(order_id)
                if order and order.delivery_complete_time is not None:
                    service_time = order.delivery_complete_time - order.arrival_time
                    service_times.append(service_time)
            
            avg_service_time = float(np.mean(service_times)) if service_times else 0.0
            
            # 计算总配送距离
            total_distance = 0.0
            for courier in self.sim_env.couriers.values():
                total_distance += getattr(courier, 'total_distance', 0.0)
            
            avg_distance = total_distance / completed_count if completed_count > 0 else 0.0
            
            # 更新统计信息
            episode_stats.update({
                'total_orders': total_orders,
                'completed_orders': completed_count,
                'timeout_orders': timeout_count,
                'pending_orders': len(self.sim_env.pending_orders),
                'assigned_orders': len(self.sim_env.assigned_orders),
                'completion_rate': completed_count / total_orders if total_orders > 0 else 0.0,
                'timeout_rate': timeout_count / total_orders if total_orders > 0 else 0.0,
                'avg_service_time': avg_service_time,
                'total_distance': total_distance,
                'avg_distance_per_order': avg_distance,
                'simulation_time': self.sim_env.env.now if self.sim_env.env else 0.0
            })
        
        return episode_stats
    
    def _preload_static_data(self):
        """
        预加载静态数据（路网、距离矩阵等），避免reset时重复IO
        
        性能优化：
        - 路网图、距离矩阵、时间矩阵、节点映射在__init__时加载一次
        - reset时直接使用缓存数据，避免每次都从磁盘读取
        - 训练速度可提升10-100倍
        """
        data_dir = Path(self.simulation_config.get('data_dir', 'data/processed'))
        allow_fallback = self.simulation_config.get('allow_random_network_fallback', False)
        
        logger.info("预加载静态数据...")
        
        try:
            # 加载路网图
            graph_file = data_dir / 'road_network.graphml'
            if graph_file.exists():
                self._cached_static_data['graph'] = nx.read_graphml(graph_file)
                logger.info(f"  路网图: {len(self._cached_static_data['graph'].nodes())} 节点")
            elif allow_fallback:
                logger.warning("路网文件不存在，生成随机图（allow_random_network_fallback=True）")
                self._cached_static_data['graph'] = self._generate_fallback_graph()
            else:
                raise FileNotFoundError(f"路网文件不存在: {graph_file}")
            
            # 加载距离和时间矩阵
            distance_matrix_file = data_dir / 'distance_matrix.npy'
            time_matrix_file = data_dir / 'time_matrix.npy'
            
            if distance_matrix_file.exists() and time_matrix_file.exists():
                self._cached_static_data['distance_matrix'] = np.load(distance_matrix_file)
                self._cached_static_data['time_matrix'] = np.load(time_matrix_file)
                logger.info(f"  距离矩阵: {self._cached_static_data['distance_matrix'].shape}")
            elif allow_fallback:
                logger.warning("矩阵文件不存在，生成随机矩阵（allow_random_network_fallback=True）")
                n_nodes = len(self._cached_static_data['graph'].nodes())
                self._cached_static_data['distance_matrix'] = np.random.rand(n_nodes, n_nodes) * 5000 + 500
                np.fill_diagonal(self._cached_static_data['distance_matrix'], 0)
                self._cached_static_data['time_matrix'] = self._cached_static_data['distance_matrix'] / 15 * 3.6
            else:
                raise FileNotFoundError(f"距离/时间矩阵文件不存在: {distance_matrix_file}")
            
            # 加载节点映射
            node_mapping_file = data_dir / 'node_id_mapping.json'
            if node_mapping_file.exists():
                with open(node_mapping_file, 'r') as f:
                    self._cached_static_data['node_mapping'] = json.load(f)
                if 'node_list' not in self._cached_static_data['node_mapping']:
                    self._cached_static_data['node_mapping']['node_list'] = list(
                        self._cached_static_data['node_mapping']['node_to_idx'].keys()
                    )
                logger.info(f"  节点映射: {len(self._cached_static_data['node_mapping']['node_list'])} 个节点")
            else:
                # 生成默认映射
                node_list = list(self._cached_static_data['graph'].nodes())
                self._cached_static_data['node_mapping'] = {
                    'node_to_idx': {str(node): i for i, node in enumerate(node_list)},
                    'idx_to_node': {i: str(node) for i, node in enumerate(node_list)},
                    'node_list': [str(node) for node in node_list]
                }
                logger.info(f"  节点映射: 自动生成 {len(node_list)} 个节点")
            
            # 处理子图提取（与node_mapping匹配）
            graph = self._cached_static_data['graph']
            node_mapping = self._cached_static_data['node_mapping']
            graph_node_sample = list(graph.nodes())[0] if len(graph.nodes()) > 0 else None
            if graph_node_sample is not None:
                if isinstance(graph_node_sample, str):
                    valid_nodes = node_mapping['node_list']
                else:
                    valid_nodes = [int(n) for n in node_mapping['node_list']]
                
                actual_nodes = [n for n in valid_nodes if n in graph.nodes()]
                if len(actual_nodes) < len(valid_nodes):
                    logger.warning(f"部分节点不在GraphML中: {len(valid_nodes)} -> {len(actual_nodes)}")
                self._cached_static_data['graph'] = graph.subgraph(actual_nodes).copy()
            
            # === 优化1: 预加载订单数据 ===
            # 订单文件在__init__时加载一次，reset时使用深拷贝
            orders_file = Path(self.simulation_config.get('orders_file', 'data/orders/orders.csv'))
            if orders_file.exists():
                self._cached_static_data['orders_raw'] = self._load_orders_from_csv(orders_file)
                logger.info(f"  订单数据: {len(self._cached_static_data['orders_raw'])} 条")
            else:
                self._cached_static_data['orders_raw'] = None
                logger.warning(f"  订单文件不存在: {orders_file}")
            
            logger.info("静态数据预加载完成")
            
        except Exception as e:
            if allow_fallback:
                logger.error(f"预加载数据失败: {e}，使用回退方案")
                self._cached_static_data['graph'] = self._generate_fallback_graph()
                n_nodes = len(self._cached_static_data['graph'].nodes())
                self._cached_static_data['distance_matrix'] = np.random.rand(n_nodes, n_nodes) * 5000 + 500
                np.fill_diagonal(self._cached_static_data['distance_matrix'], 0)
                self._cached_static_data['time_matrix'] = self._cached_static_data['distance_matrix'] / 15 * 3.6
                node_list = list(self._cached_static_data['graph'].nodes())
                self._cached_static_data['node_mapping'] = {
                    'node_to_idx': {str(node): i for i, node in enumerate(node_list)},
                    'idx_to_node': {i: str(node) for i, node in enumerate(node_list)},
                    'node_list': [str(node) for node in node_list]
                }
            else:
                raise RuntimeError(f"预加载静态数据失败: {e}")
    
    def _create_simulation_environment(self):
        """
        创建或重置仿真环境
        
        性能优化：
        - 使用_preload_static_data()预加载的缓存数据
        - 避免每次reset都从磁盘读取路网和矩阵
        - 训练速度可提升10-100倍
        
        完整流程：
        1. 使用缓存的路网、距离矩阵、节点映射
        2. 加载订单（订单文件每次都需要重新加载，因为可能被截断）
        3. 初始化骑手
        4. 初始化仿真进程
        """
        logger.debug("创建仿真环境实例（使用缓存数据）")
        
        # === 性能优化：使用预加载的缓存数据 ===
        # 优化2: 减少图复制开销
        # 仿真环境不会修改图结构，直接传引用（节省~10ms/reset）
        # 如果未来仿真需要修改图，再改回copy()
        graph = self._cached_static_data['graph']
        distance_matrix = self._cached_static_data['distance_matrix']
        time_matrix = self._cached_static_data['time_matrix']
        # node_mapping需要浅拷贝（仿真可能添加新节点）
        node_mapping = {
            k: (v.copy() if isinstance(v, (dict, list)) else v)
            for k, v in self._cached_static_data['node_mapping'].items()
        }
        
        # 创建仿真环境实例
        # 关键：禁用自动调度，避免后台GreedyDispatcher与RL Agent冲突
        rl_sim_config = {**self.simulation_config, 'enable_auto_dispatch': False}
        self.sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            node_mapping=node_mapping,
            config=rl_sim_config
        )
        
        # === 优化1: 使用预加载的订单数据 ===
        # 如果有预加载的订单数据，直接使用（避免每次reset都读取CSV）
        if self._cached_static_data.get('orders_raw') is not None:
            self.sim_env.load_orders_from_raw_data(self._cached_static_data['orders_raw'])
        else:
            # 回退：从文件加载
            orders_file = Path(self.simulation_config.get('orders_file', 'data/orders/orders.csv'))
            if orders_file.exists():
                self.sim_env.load_orders_from_csv(orders_file)
            else:
                logger.warning(f"订单文件不存在: {orders_file}")
                raise FileNotFoundError(f"订单文件不存在: {orders_file}")
        
        # 注意：SimulationEnvironment.load_orders_from_csv 已自动调整订单到达时间
        
        # 如果配置了total_orders，截断订单数量以匹配课程学习阶段
        # 这确保了配置的total_orders与实际订单数一致
        total_orders_config = self.simulation_config.get('total_orders')
        if total_orders_config is not None and len(self.sim_env.orders) > total_orders_config:
            # 按到达时间排序，保留前N个订单
            sorted_order_ids = sorted(
                self.sim_env.orders.keys(),
                key=lambda oid: self.sim_env.orders[oid].arrival_time
            )
            orders_to_remove = sorted_order_ids[total_orders_config:]
            for order_id in orders_to_remove:
                del self.sim_env.orders[order_id]
            logger.info(f"订单数量截断: {len(sorted_order_ids)} -> {total_orders_config}")
        
        # 初始化骑手
        num_couriers = self.simulation_config.get('num_couriers', 20)
        courier_config = self.simulation_config.get('courier_config', {
            'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
            'capacity': {'max_orders': 5}
        })
        self.sim_env.initialize_couriers(num_couriers, courier_config)
        
        # 初始化仿真进程（但不运行）
        self.sim_env.initialize_processes()
        
        logger.info("仿真环境创建完成")
    
    def _generate_fallback_graph(self) -> nx.MultiDiGraph:
        """
        生成回退用的随机路网图
        
        Returns:
            随机生成的路网图
        """
        logger.info("生成回退路网图")
        graph = nx.MultiDiGraph()
        
        # 生成100个节点
        n_nodes = 100
        for i in range(n_nodes):
            graph.add_node(i, x=116.38 + np.random.rand() * 0.06, 
                          y=39.88 + np.random.rand() * 0.06)
        
        # 添加边（连接每个节点与最近的5个节点）
        for i in range(n_nodes):
            for j in range(i+1, min(i+6, n_nodes)):
                dist = np.random.rand() * 1000 + 200
                graph.add_edge(i, j, length=dist)
                graph.add_edge(j, i, length=dist)
        
        return graph
    
    def _load_orders_from_csv(self, orders_file: Path) -> list:
        """
        从CSV文件加载订单原始数据（用于预加载缓存）
        
        Args:
            orders_file: 订单CSV文件路径
        
        Returns:
            订单原始数据列表（字典格式）
        """
        import pandas as pd
        
        try:
            df = pd.read_csv(orders_file)
            orders_raw = df.to_dict('records')
            return orders_raw
        except Exception as e:
            logger.error(f"加载订单文件失败: {e}")
            return []
    
    def action_masks(self) -> np.ndarray:
        """
        返回当前状态下的动作掩码（Action Mask）
        
        用于MaskablePPO，屏蔽无效动作（如满载骑手）
        
        Returns:
            对于discrete模式: shape=(max_couriers+1,) 的布尔数组
            对于multi_discrete模式: shape=(max_pending_orders, max_couriers+1) 的布尔数组
        """
        if self.action_mode == 'discrete':
            return self._get_discrete_action_mask()
        elif self.action_mode == 'multi_discrete':
            return self._get_multi_discrete_action_mask()
        else:
            # continuous模式不支持动作屏蔽
            raise NotImplementedError(f"动作屏蔽不支持{self.action_mode}模式")
    
    def _get_discrete_action_mask(self) -> np.ndarray:
        """
        获取discrete模式的动作掩码
        
        检查条件：
        1. 骑手有空余容量 (assigned < capacity)
        2. 骑手处于可接单状态 (is_available)
        
        Returns:
            shape=(max_couriers+1,) 的布尔数组
            mask[0] = True (延迟动作始终有效)
            mask[i] = True 如果骑手i可以接单
        """
        mask = np.zeros(self.max_couriers + 1, dtype=bool)
        
        # 动作0（延迟派单）始终有效
        mask[0] = True
        
        if not self.sim_env:
            return mask
        
        # 检查每个骑手是否可以接单
        for courier_id, courier in self.sim_env.couriers.items():
            if courier_id <= self.max_couriers:
                # 检查容量
                has_capacity = len(courier.assigned_orders) < courier.max_capacity
                # 检查是否可用（如果有is_available方法）
                is_available = courier.is_available() if hasattr(courier, 'is_available') else True
                # 检查是否可以接受新订单（如果有can_accept_new_order方法）
                can_accept = courier.can_accept_new_order() if hasattr(courier, 'can_accept_new_order') else has_capacity
                
                if can_accept:
                    mask[courier_id] = True
        
        return mask
    
    def _get_multi_discrete_action_mask(self) -> np.ndarray:
        """
        获取multi_discrete模式的动作掩码
        
        检查条件：
        1. 骑手有空余容量 (assigned < capacity)
        2. 骑手处于可接单状态 (can_accept_new_order)
        3. 订单槽位有对应的待分配订单
        
        Returns:
            shape=(max_pending_orders, max_couriers+1) 的布尔数组
            每行对应一个订单槽位的有效动作
        """
        mask = np.zeros((self.max_pending_orders, self.max_couriers + 1), dtype=bool)
        
        # 动作0（延迟/无操作）始终有效
        mask[:, 0] = True
        
        if not self.sim_env:
            return mask
        
        # 获取当前待分配订单数
        num_pending = len(self.sim_env.pending_orders)
        
        # 计算每个骑手的可接单状态和剩余容量
        courier_can_accept = {}
        for courier_id, courier in self.sim_env.couriers.items():
            if courier_id <= self.max_couriers:
                # 使用can_accept_new_order方法（如果存在）
                if hasattr(courier, 'can_accept_new_order'):
                    can_accept = courier.can_accept_new_order()
                else:
                    # 回退：检查容量
                    can_accept = len(courier.assigned_orders) < courier.max_capacity
                courier_can_accept[courier_id] = can_accept
        
        # 为每个订单槽位设置掩码
        # 注意：需要考虑前面订单分配后对容量的影响
        # 这里使用保守估计：假设每个订单都可能分配给任何可接单的骑手
        for order_idx in range(self.max_pending_orders):
            if order_idx >= num_pending:
                # 没有订单的槽位，只有延迟动作有效
                continue
            
            # 检查每个骑手是否可用
            for courier_id, can_accept in courier_can_accept.items():
                if can_accept:
                    mask[order_idx, courier_id] = True
        
        return mask
    
    def _execute_action(self, action: Any) -> Dict[str, Any]:
        """
        执行RL动作（派单决策）
        
        Args:
            action: RL Agent输出的动作
        
        Returns:
            动作执行信息
        """
        action_info = {
            'raw_action': action,
            'action_type': None,
            'assignments': [],
            'distance_increase': 0.0
        }
        
        if self.action_mode == 'discrete':
            # 单订单派单
            action_info.update(self._execute_discrete_action(action))
        
        elif self.action_mode == 'multi_discrete':
            # 多订单批量派单
            action_info.update(self._execute_multi_discrete_action(action))
        
        elif self.action_mode == 'continuous':
            # 基于概率分布采样
            action_info.update(self._execute_continuous_action(action))
        
        return action_info
    
    def _execute_discrete_action(self, action: int) -> Dict[str, Any]:
        """
        执行离散动作
        
        Args:
            action: 骑手ID（0=延迟，1-N=分配给骑手）
        
        Returns:
            执行详情
        """
        # 确保action是Python整数（处理numpy数组情况）
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)
        
        # 获取当前第一个待分配订单
        if not self.sim_env or not self.sim_env.pending_orders:
            return {
                'action_type': 'no_op',
                'assignments': [],
                'distance_increase': 0.0
            }
        
        order_id = self.sim_env.pending_orders[0]
        order = self.sim_env.orders[order_id]
        
        if action == 0:
            # 延迟派单
            # Day 27: 记录延迟时的最佳可用骑手距离，用于后续判断delay_justified
            best_distance = self._get_best_available_courier_distance(order)
            if order_id not in self._delay_tracking:
                self._delay_tracking[order_id] = {
                    'delay_time': self.sim_env.env.now,
                    'best_distance_at_delay': best_distance,
                    'delay_count': 1
                }
            else:
                # 多次延迟，更新最佳距离（取最小值）
                self._delay_tracking[order_id]['delay_count'] += 1
                self._delay_tracking[order_id]['best_distance_at_delay'] = min(
                    self._delay_tracking[order_id]['best_distance_at_delay'],
                    best_distance
                )
            
            return {
                'action_type': 'delay',
                'order_id': order_id,
                'assignments': [],
                'distance_increase': 0.0,
                'best_distance_at_delay': best_distance
            }
        else:
            # 分配给骑手
            courier_id = action  # 1-indexed to 1-indexed
            
            # 尝试分配给指定骑手，如果满载则回退到其他可用骑手
            assigned_courier_id, distance_increase = self._try_assign_with_fallback(
                order, courier_id
            )
            
            if assigned_courier_id is None:
                # 所有骑手都满载，订单延迟
                logger.warning(f"所有骑手容量已满，订单{order_id}延迟")
                return {
                    'action_type': 'all_couriers_full',
                    'order_id': order_id,
                    'assignments': [],
                    'distance_increase': 0.0
                }
            
            # 从待分配队列中移除
            self.sim_env.pending_orders.remove(order_id)
            # 添加到已分配队列
            self.sim_env.assigned_orders.append(order_id)
            # 注意：订单状态已在_greedy_insert_order内部更新，无需重复调用
            
            action_type = 'assign' if assigned_courier_id == courier_id else 'assign_fallback'
            logger.debug(f"订单{order_id}分配给骑手{assigned_courier_id}, 距离增加{distance_increase:.0f}m")
            
            # Day 27: 判断延迟是否合理（如果该订单之前被延迟过）
            delay_justified = False
            if order_id in self._delay_tracking:
                tracking = self._delay_tracking[order_id]
                # 计算当前分配的骑手到商家距离
                courier = self.sim_env.couriers[assigned_courier_id]
                try:
                    current_distance = self.sim_env.get_distance(
                        courier.current_node, order.merchant_node
                    )
                    # 如果当前距离比延迟时的最佳距离更短，说明延迟是合理的
                    # 使用10%的阈值，避免微小差异触发
                    if current_distance < tracking['best_distance_at_delay'] * 0.9:
                        delay_justified = True
                        logger.debug(
                            f"订单{order_id}延迟合理: 延迟时最佳距离={tracking['best_distance_at_delay']:.0f}m, "
                            f"当前距离={current_distance:.0f}m"
                        )
                except Exception:
                    pass
                # 清理追踪记录
                del self._delay_tracking[order_id]
            
            return {
                'action_type': action_type,
                'order_id': order_id,
                'courier_id': assigned_courier_id,
                'original_courier_id': courier_id,
                'assignments': [(order_id, assigned_courier_id)],
                'distance_increase': distance_increase,
                'delay_justified': delay_justified
            }
    
    def _try_assign_with_fallback(self, order: Any, preferred_courier_id: int) -> Tuple[Optional[int], float]:
        """
        尝试分配订单给骑手，如果首选骑手满载则回退到其他骑手
        
        Args:
            order: 订单对象
            preferred_courier_id: 首选骑手ID
        
        Returns:
            (成功分配的骑手ID, 距离增加量)，如果无法分配返回(None, 0)
        """
        # 首先尝试首选骑手
        if preferred_courier_id in self.sim_env.couriers:
            courier = self.sim_env.couriers[preferred_courier_id]
            distance_increase = self._insert_order_with_strategy(order, courier)
            if distance_increase >= 0:
                return preferred_courier_id, distance_increase
        
        # 首选骑手不可用，按距离排序尝试其他骑手
        candidates = []
        for cid, courier in self.sim_env.couriers.items():
            if cid == preferred_courier_id:
                continue
            if len(courier.assigned_orders) < courier.max_capacity:
                # 计算骑手到商家的距离
                try:
                    dist = self.sim_env.get_distance(courier.current_node, order.merchant_node)
                    candidates.append((cid, dist, courier))
                except:
                    candidates.append((cid, float('inf'), courier))
        
        # 按距离排序
        candidates.sort(key=lambda x: x[1])
        
        # 尝试分配给最近的可用骑手
        for cid, _, courier in candidates:
            distance_increase = self._insert_order_with_strategy(order, courier)
            if distance_increase >= 0:
                return cid, distance_increase
        
        # 所有骑手都满载
        return None, 0.0
    
    def _get_best_available_courier_distance(self, order: Any) -> float:
        """
        获取当前可用骑手中到商家最近的距离
        
        用于延迟派单追踪，判断延迟是否合理
        
        Args:
            order: 订单对象
            
        Returns:
            最近可用骑手到商家的距离（米），如果没有可用骑手返回无穷大
        """
        best_distance = float('inf')
        
        for courier in self.sim_env.couriers.values():
            # 只考虑有容量的骑手
            if len(courier.assigned_orders) >= courier.max_capacity:
                continue
            
            try:
                dist = self.sim_env.get_distance(courier.current_node, order.merchant_node)
                if dist < best_distance:
                    best_distance = dist
            except Exception:
                continue
        
        return best_distance
    
    def _greedy_insert_order(self, order: Any, courier: Any) -> float:
        """
        使用贪婪插入算法将订单插入到骑手路线中
        
        职责划分说明：
        - 本方法负责：订单状态更新(order.status)、骑手路线更新(courier.current_route)
        - 调用方负责：队列更新(pending_orders.remove, assigned_orders.append)
        - 这样分层确保队列更新只在插入成功时执行
        
        Args:
            order: 订单对象
            courier: 骑手对象
        
        Returns:
            距离增加量（米），如果无法分配则返回-1
        """
        from ..simulation.entities import OrderStatus
        
        # 【关键检查】订单状态全局检查：只有PENDING状态的订单可以被分配
        # 如果订单已被分配给其他骑手（状态不是PENDING），拒绝再次分配
        if order.status != OrderStatus.PENDING:
            logger.debug(f"订单{order.order_id}状态为{order.status}，非PENDING，跳过分配")
            return -1.0
        
        # 【新增】全局检查：确保订单没有在任何骑手的assigned_orders或current_route中
        # 这是防止跨骑手重复分配的最后一道防线
        for cid, c in self.sim_env.couriers.items():
            if order.order_id in c.assigned_orders:
                logger.warning(f"订单{order.order_id}已在骑手{cid}的assigned_orders中，拒绝分配给骑手{courier.courier_id}")
                return -1.0
            for task in c.current_route:
                if task[1] == order.order_id:
                    logger.warning(f"订单{order.order_id}已在骑手{cid}的路线中，拒绝分配给骑手{courier.courier_id}")
                    return -1.0
        
        # 检查骑手容量（本骑手特定检查）
        if order.order_id in courier.assigned_orders:
            logger.debug(f"订单{order.order_id}已在骑手{courier.courier_id}的任务列表中，跳过重复分配")
            return -1.0
        
        # 检查骑手容量
        if len(courier.assigned_orders) >= courier.max_capacity:
            logger.debug(f"骑手{courier.courier_id}容量已满，无法接单")
            return -1.0  # 返回负值表示无法分配
        
        # 构建取货和送货任务
        pickup_task = ('pickup', order.order_id, order.merchant_node)
        delivery_task = ('delivery', order.order_id, order.customer_node)
        
        # 【关键】在添加任务前先更新订单状态，防止并发分配
        # 这必须在添加任务到路线之前完成，确保其他分配尝试会被拒绝
        # 保存原始状态以便失败时回滚
        original_status = order.status
        original_courier_id = order.assigned_courier_id
        original_assigned_time = order.assigned_time
        
        try:
            order.status = OrderStatus.ASSIGNED
            order.assigned_courier_id = courier.courier_id
            order.assigned_time = self.sim_env.env.now
            
            # 如果骑手当前无任务，直接添加
            if len(courier.current_route) == 0:
                courier.current_route.append(pickup_task)
                courier.current_route.append(delivery_task)
                courier.assign_order(order.order_id)
                
                # 计算距离（添加错误处理）
                try:
                    pickup_dist = self.sim_env.get_distance(courier.current_node, order.merchant_node)
                    delivery_dist = self.sim_env.get_distance(order.merchant_node, order.customer_node)
                    return pickup_dist + delivery_dist
                except (ValueError, IndexError) as e:
                    logger.warning(f"无法计算订单{order.order_id}的距离: {e}，使用估计值")
                    return 5000.0  # 默认5km
            
            # 找到最优插入位置（最小化距离增加）
            best_pickup_pos = 0
            best_delivery_pos = 1
            min_cost = float('inf')
            
            current_route = courier.current_route.copy()
            current_route_nodes = [courier.current_node] + [task[2] for task in current_route]
            
            # 尝试所有可能的插入位置
            for pickup_pos in range(len(current_route) + 1):
                for delivery_pos in range(pickup_pos + 1, len(current_route) + 2):
                    # 计算插入成本
                    route_with_insert = current_route[:pickup_pos] + [pickup_task] + \
                                      current_route[pickup_pos:delivery_pos-1] + [delivery_task] + \
                                      current_route[delivery_pos-1:]
                    
                    nodes_with_insert = [courier.current_node] + [task[2] for task in route_with_insert]
                    
                    # 计算总距离
                    total_dist = 0
                    for i in range(len(nodes_with_insert) - 1):
                        try:
                            total_dist += self.sim_env.get_distance(nodes_with_insert[i], nodes_with_insert[i+1])
                        except:
                            total_dist += 1e9  # 惩罚不可达路径
                    
                    if total_dist < min_cost:
                        min_cost = total_dist
                        best_pickup_pos = pickup_pos
                        best_delivery_pos = delivery_pos
            
            # 计算原路线距离
            original_dist = 0
            for i in range(len(current_route_nodes) - 1):
                try:
                    original_dist += self.sim_env.get_distance(current_route_nodes[i], current_route_nodes[i+1])
                except:
                    original_dist += 0
            
            # 【关键】插入前再次验证，确保订单没有被其他并发操作添加
            route_order_ids_final = [task[1] for task in courier.current_route]
            if order.order_id in route_order_ids_final:
                logger.warning(f"订单{order.order_id}在插入前已存在于骑手{courier.courier_id}的路线中，放弃插入")
                # 回滚订单状态
                order.status = original_status
                order.assigned_courier_id = original_courier_id
                order.assigned_time = original_assigned_time
                return -1.0
            
            # 插入任务
            courier.current_route.insert(best_pickup_pos, pickup_task)
            courier.current_route.insert(best_delivery_pos, delivery_task)
            courier.assign_order(order.order_id)
            
            distance_increase = min_cost - original_dist
            return max(0, distance_increase)  # 确保非负
            
        except Exception as e:
            # 如果添加任务失败，回滚订单状态
            logger.error(f"添加订单{order.order_id}到骑手{courier.courier_id}失败: {e}，回滚状态")
            order.status = original_status
            order.assigned_courier_id = original_courier_id
            order.assigned_time = original_assigned_time
            return -1.0
    
    def _alns_insert_order(self, order: Any, courier: Any) -> float:
        """
        使用ALNS风格的路径优化将订单插入到骑手路线中
        
        先执行贪婪插入，然后使用2-opt优化路线
        
        Args:
            order: 订单对象
            courier: 骑手对象
        
        Returns:
            距离增加量（米），如果无法分配则返回-1
        """
        # 记录插入前的原路线距离（用于计算真实增量）
        original_route_dist = self._calculate_route_total_distance(
            courier.current_route, courier.current_node
        ) if courier.current_route else 0.0
        
        # 首先使用贪婪插入
        greedy_increase = self._greedy_insert_order(order, courier)
        
        if greedy_increase < 0:
            return -1.0
        
        # 如果路线任务数小于等于4，不需要优化
        if len(courier.current_route) <= 4:
            return greedy_increase
        
        # 执行2-opt优化
        route = courier.current_route.copy()
        improved = True
        iteration = 0
        
        while improved and iteration < self.alns_max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(route) - 2):
                for j in range(i + 2, len(route)):
                    # 检查交换是否违反取送约束
                    if self._is_2opt_swap_valid(route, i, j):
                        # 计算交换前后的总距离
                        old_dist = self._calculate_route_total_distance(route, courier.current_node)
                        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                        new_dist = self._calculate_route_total_distance(new_route, courier.current_node)
                        
                        if new_dist < old_dist:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        
        # 更新骑手路线
        courier.current_route = route
        
        # 计算优化后的真实距离增量（优化后总距离 - 插入前原路线距离）
        final_dist = self._calculate_route_total_distance(route, courier.current_node)
        actual_increase = final_dist - original_route_dist
        
        logger.debug(f"骑手{courier.courier_id}路线已优化(ALNS 2-opt, {iteration}次迭代), "
                    f"贪婪增量={greedy_increase:.0f}m, 优化后增量={actual_increase:.0f}m")
        
        return max(0, actual_increase)
    
    def _is_2opt_swap_valid(self, route: List, i: int, j: int) -> bool:
        """
        检查2-opt交换是否违反取送约束
        
        每个订单的取货必须在配送之前
        
        Args:
            route: 当前路线
            i, j: 交换位置索引
        
        Returns:
            交换是否有效
        """
        # 检查交换后的路线
        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
        
        # 验证每个订单的取送顺序
        pickup_positions = {}
        delivery_positions = {}
        
        for idx, task in enumerate(new_route):
            action, order_id, _ = task
            if action == 'pickup':
                pickup_positions[order_id] = idx
            else:
                delivery_positions[order_id] = idx
        
        # 检查每个订单是否满足先取后送
        for order_id in pickup_positions:
            if order_id in delivery_positions:
                if pickup_positions[order_id] >= delivery_positions[order_id]:
                    return False
        
        return True
    
    def _calculate_route_total_distance(self, route: List, start_node) -> float:
        """
        计算路线总距离
        
        Args:
            route: 路线任务列表
            start_node: 起始节点
        
        Returns:
            总距离（米）
        """
        total = 0.0
        current = start_node
        
        for _, _, node in route:
            try:
                total += self.sim_env.get_distance(current, node)
                current = node
            except Exception:
                total += 10000  # 不可达时给予高惩罚
        
        return total
    
    def _fifo_insert_order(self, order: Any, courier: Any) -> float:
        """
        使用FIFO方式将订单插入到骑手路线末尾
        
        简单追加，不做优化，与GreedyDispatcher行为一致
        
        Args:
            order: 订单对象
            courier: 骑手对象
        
        Returns:
            距离增加量（米），如果无法分配则返回-1
        """
        from ..simulation.entities import OrderStatus
        
        # 订单状态检查
        if order.status != OrderStatus.PENDING:
            return -1.0
        
        # 全局检查：确保订单没有在任何骑手中
        for cid, c in self.sim_env.couriers.items():
            if order.order_id in c.assigned_orders:
                return -1.0
            for task in c.current_route:
                if task[1] == order.order_id:
                    return -1.0
        
        # 容量检查
        if len(courier.assigned_orders) >= courier.max_capacity:
            return -1.0
        
        # 更新订单状态
        order.status = OrderStatus.ASSIGNED
        order.assigned_courier_id = courier.courier_id
        order.assigned_time = self.sim_env.env.now
        
        # FIFO方式：简单追加到路线末尾
        pickup_task = ('pickup', order.order_id, order.merchant_node)
        delivery_task = ('delivery', order.order_id, order.customer_node)
        
        courier.current_route.append(pickup_task)
        courier.current_route.append(delivery_task)
        courier.assign_order(order.order_id)
        
        # 计算距离增加
        # 注意：此时pickup_task和delivery_task已经追加到路线末尾
        # 所以current_route[-2]是pickup_task，current_route[-3]是追加前的最后一个任务
        try:
            if len(courier.current_route) == 2:
                # 第一个订单：从骑手当前位置出发
                last_node = courier.current_node
            else:
                # 追加到现有路线：从追加前的最后一个任务节点出发
                # current_route[-3]是追加前的最后一个任务（因为已经追加了pickup和delivery）
                last_node = courier.current_route[-3][2]
            
            dist = self.sim_env.get_distance(last_node, order.merchant_node)
            dist += self.sim_env.get_distance(order.merchant_node, order.customer_node)
            return dist
        except Exception:
            return 5000.0  # 默认5km
    
    def _insert_order_with_strategy(self, order: Any, courier: Any) -> float:
        """
        根据配置的路径优化策略插入订单
        
        职责边界说明（重要）：
        ┌─────────────────────────────────────────────────────────────────┐
        │ 本方法及其调用的插入方法（greedy/fifo/alns）负责：              │
        │   - 订单状态更新：order.status = ASSIGNED                       │
        │   - 骑手路线更新：courier.current_route.append(...)             │
        │   - 骑手任务列表：courier.assign_order(order_id)                │
        │                                                                  │
        │ 调用方（_execute_discrete_action等）负责：                       │
        │   - 队列更新：pending_orders.remove(order_id)                   │
        │   - 队列更新：assigned_orders.append(order_id)                  │
        │                                                                  │
        │ 这样分层的原因：                                                 │
        │   - 插入可能失败（返回-1），队列更新只在成功时执行              │
        │   - 避免队列状态与订单状态不一致                                 │
        │   - 如果在其他地方调用本方法，必须遵循相同的职责划分            │
        └─────────────────────────────────────────────────────────────────┘
        
        Args:
            order: 订单对象
            courier: 骑手对象
        
        Returns:
            距离增加量（米），如果无法分配则返回-1
        """
        if self.routing_strategy == 'alns':
            return self._alns_insert_order(order, courier)
        elif self.routing_strategy == 'fifo':
            return self._fifo_insert_order(order, courier)
        else:  # 默认 'greedy_insert'
            return self._greedy_insert_order(order, courier)
    
    def _execute_multi_discrete_action(self, actions: np.ndarray) -> Dict[str, Any]:
        """
        执行多离散动作（批量派单）
        
        Args:
            actions: 动作数组，每个元素对应一个待分配订单的派单决策
        
        Returns:
            执行详情
        """
        if not self.sim_env or not self.sim_env.pending_orders:
            return {
                'action_type': 'multi_assign',
                'assignments': [],
                'distance_increase': 0.0
            }
        
        assignments = []
        total_distance_increase = 0.0
        
        # 获取待分配订单（最多max_pending_orders个）
        pending_orders = self.sim_env.pending_orders[:self.max_pending_orders]
        
        for i, order_id in enumerate(pending_orders):
            if i >= len(actions):
                break
            
            raw_action = int(actions[i])
            order = self.sim_env.orders[order_id]
            
            if raw_action == 0:
                # 明确延迟派单
                continue
            
            # 使用_try_assign_with_fallback，与discrete模式保持一致
            # 这样即使action指向不存在的骑手ID，也会自动回退到最近可用骑手
            preferred_courier_id = raw_action
            assigned_courier_id, distance_increase = self._try_assign_with_fallback(
                order, preferred_courier_id
            )
            
            if assigned_courier_id is None:
                # 所有骑手都满载或无法插入，视为延迟
                logger.debug(f"订单{order_id}无法分配（首选骑手{preferred_courier_id}），保持待分配状态")
                continue
            
            # 插入成功，更新队列
            if order_id in self.sim_env.pending_orders:
                self.sim_env.pending_orders.remove(order_id)
            if order_id not in self.sim_env.assigned_orders:
                self.sim_env.assigned_orders.append(order_id)
            # 注意：订单状态已在插入方法内部更新，无需重复调用
            
            # 记录分配详情（包含order对象，用于奖励计算中的商家惩罚）
            is_fallback = assigned_courier_id != preferred_courier_id
            assignments.append({
                'order_id': order_id,
                'order': order,  # 订单对象，用于检查商家信息
                'courier_id': assigned_courier_id,
                'distance_increase': distance_increase,
                'is_fallback': is_fallback
            })
            total_distance_increase += distance_increase
        
        return {
            'action_type': 'multi_assign',
            'assignments': assignments,
            'distance_increase': total_distance_increase
        }
    
    def _execute_continuous_action(self, action: np.ndarray) -> Dict[str, Any]:
        """执行连续动作（基于概率采样）"""
        # TODO: 从概率分布中采样派单决策
        return {
            'action_type': 'probabilistic_assign',
            'assignments': [],
            'distance_increase': 0.0
        }
    
    def _advance_simulation(self):
        """推进仿真到下一个决策时刻"""
        if self.sim_env is None:
            logger.warning("仿真环境未初始化，无法推进")
            return
        
        # 运行仿真一个调度间隔
        dispatch_interval = self.simulation_config.get('dispatch_interval', 30.0)
        
        # 调用仿真环境的step方法推进
        # 保存step_info供_get_step_info使用（透传merchant_wait_events等信息）
        self._last_sim_step_info = self.sim_env.step(dispatch_interval)
        
        logger.debug(f"仿真推进到 {self._last_sim_step_info['current_time']:.1f}s, "
                    f"待分配: {self._last_sim_step_info['pending_orders']}, "
                    f"已完成: {self._last_sim_step_info['completed_orders']}")
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观测（状态向量）
        
        Returns:
            观测向量
        """
        if self.sim_env is None:
            # 返回零向量（环境未初始化）
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 性能优化：复用已缓存的_current_state，避免重复构造
        if self._current_state is None:
            self._current_state = self._get_full_state()
        
        # 编码为向量
        observation = self.state_encoder.encode(self._current_state)
        
        return observation
    
    def _get_full_state(self) -> Dict[str, Any]:
        """
        获取完整环境状态
        
        Returns:
            状态字典
        """
        if self.sim_env is None:
            return {}
        
        # 获取地理边界
        bounds = self._get_geographic_bounds()
        
        current_time = self.sim_env.env.now if self.sim_env.env else 0.0
        pending_orders = [
            self.sim_env.orders[oid] 
            for oid in self.sim_env.pending_orders
        ]
        
        # 计算待处理订单的平均等待时间（用于dense奖励中的等待时间惩罚）
        pending_wait_time = 0.0
        if pending_orders:
            wait_times = [
                max(current_time - order.arrival_time, 0.0)
                for order in pending_orders
            ]
            pending_wait_time = sum(wait_times) / len(wait_times)
        
        state = {
            'current_time': current_time,
            'pending_orders': pending_orders,
            'pending_wait_time': pending_wait_time,  # 平均等待时间（秒）
            'couriers': self.sim_env.couriers,
            'merchants': getattr(self.sim_env, 'merchants', {}),  # 商家信息（用于等餐惩罚）
            'graph': self.sim_env.graph,
            'bounds': bounds
        }
        
        return state
    
    def _get_geographic_bounds(self) -> Tuple[float, float, float, float]:
        """
        获取路网地理边界
        
        Returns:
            (min_x, max_x, min_y, max_y)
        """
        # 默认边界（北京某区域）
        default_bounds = (116.38, 116.44, 39.88, 39.94)
        
        if self.sim_env is None or self.sim_env.graph is None:
            return default_bounds
        
        # 从路网图中提取边界，添加容错处理
        x_coords = []
        y_coords = []
        for _, data in self.sim_env.graph.nodes(data=True):
            try:
                x_coords.append(float(data['x']))
                y_coords.append(float(data['y']))
            except (KeyError, TypeError, ValueError):
                # 跳过缺少x/y属性或格式错误的节点
                continue
        
        # 如果没有有效坐标，返回默认值
        if not x_coords or not y_coords:
            logger.warning("路网图中没有有效的坐标数据，使用默认边界")
            return default_bounds
        
        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
    
    def _get_step_info(self) -> Dict[str, Any]:
        """
        获取步进额外信息
        
        透传SimulationEnvironment.step()返回的信息，包括：
        - merchant_wait_events: 商家等餐事件（用于等餐惩罚）
        - delay_justified: 延迟是否合理
        - 其他仿真级别的事件信息
        
        Returns:
            信息字典
        """
        if self.sim_env is None:
            return {}
        
        # 从sim_env的step_info透传原始信息（包含merchant_wait_events等）
        base_info = getattr(self, '_last_sim_step_info', {}) or {}
        
        # 检查是否有订单完成
        completed_orders = self._extract_completed_orders()
        
        info = {
            **base_info,  # 透传merchant_wait_events, delay_justified等
            'current_time': self.sim_env.env.now if self.sim_env.env else 0.0,
            'pending_count': len(self.sim_env.pending_orders),
            'completed_orders': completed_orders,
            'total_completed': len(self.sim_env.completed_orders),
            'timeout_count': self.sim_env.stats.get('timeout_orders', 0)
        }
        
        return info
    
    def _extract_completed_orders(self) -> List[Dict[str, Any]]:
        """
        提取自上一步以来完成的订单
        
        Returns:
            完成订单信息列表
        """
        if self.sim_env is None:
            return []
        
        completed_orders = []
        
        # 检查自上一步以来新完成的订单
        if hasattr(self, '_last_completed_count'):
            new_completed = self.sim_env.completed_orders[self._last_completed_count:]
            
            for order_id in new_completed:
                order = self.sim_env.orders[order_id]
                completion_time = order.delivery_complete_time
                
                # 计算slack_time（提前完成时间）
                # slack_time > 0 表示提前完成，用于稀疏奖励中的提前送达奖励
                slack_time = 0.0
                if completion_time is not None and hasattr(order, 'latest_delivery_time'):
                    slack_time = order.latest_delivery_time - completion_time
                
                completed_orders.append({
                    'order_id': order_id,
                    'completion_time': completion_time,
                    'is_timeout': order.is_timeout(completion_time) if completion_time else False,
                    'wait_time': completion_time - order.arrival_time if completion_time else 0,
                    'assigned_courier': order.assigned_courier_id,
                    'slack_time': slack_time  # 提前完成时间（秒）
                })
        
        # 更新完成计数
        self._last_completed_count = len(self.sim_env.completed_orders)
        
        return completed_orders
    
    def _check_termination(self) -> bool:
        """
        检查Episode是否自然终止
        
        Episode终止条件：
        1. 所有订单都已进入系统（当前时间 >= 最晚订单到达时间）
        2. 没有活跃订单（pending == 0 且 assigned == 0）
        
        这样可以避免"中间空档"导致的提前终止：
        - 某时刻已到达的订单都送完了，但还有订单未到达
        - 如果只检查pending和assigned，会错误地认为episode结束
        
        注意：
        - 如果max_arrival_time > simulation_duration，此条件不会触发
        - 此时依赖_check_truncation（超时截断）来结束Episode
        - 这是预期行为，因为订单到达时间已在load_orders时调整到仿真范围内
        
        Returns:
            是否终止
        """
        if self.sim_env is None:
            return True
        
        # 容错：订单为空时直接终止
        total_orders = len(self.sim_env.orders)
        if total_orders == 0:
            logger.warning("订单为空，Episode终止")
            return True
        
        # 检查是否还有活跃订单（待分配或正在配送）
        pending_count = len(self.sim_env.pending_orders)
        assigned_count = len(self.sim_env.assigned_orders)
        
        # 如果还有活跃订单，不终止
        if pending_count > 0 or assigned_count > 0:
            return False
        
        # 检查是否所有订单都已进入系统（当前时间 >= 最晚订单到达时间）
        # 这避免了"中间空档"导致的提前终止
        current_time = self.sim_env.env.now if self.sim_env.env else 0.0
        
        # 容错：安全计算max_arrival_time
        try:
            max_arrival_time = max(
                order.arrival_time for order in self.sim_env.orders.values()
            )
        except (ValueError, AttributeError) as e:
            # 订单列表为空或格式错误
            logger.warning(f"计算max_arrival_time失败: {e}，Episode终止")
            return True
        
        # 只有当前时间超过最晚订单到达时间，且没有活跃订单时才终止
        if current_time < max_arrival_time:
            # 还有订单未到达，不终止
            return False
        
        # 所有订单都已进入系统，且没有活跃订单，可以终止
        return True
    
    def _check_truncation(self) -> bool:
        """
        检查Episode是否被截断（超时等）
        
        Returns:
            是否截断
        """
        if self.sim_env is None:
            return False
        
        # 达到最大仿真时长
        max_duration = self.simulation_config.get('simulation_duration', 86400)
        current_time = self.sim_env.env.now if self.sim_env.env else 0.0
        
        return current_time >= max_duration
    
    def render(self, mode: str = 'human'):
        """
        渲染环境（可选）
        
        Args:
            mode: 渲染模式
        """
        if mode == 'human':
            # 打印当前状态摘要
            if self.sim_env:
                print(f"\n[Time: {self.sim_env.env.now:.1f}s]")
                print(f"  Pending Orders: {len(self.sim_env.pending_orders)}")
                print(f"  Completed Orders: {len(self.sim_env.completed_orders)}")
                print(f"  Timeout Orders: {self.sim_env.stats.get('timeout_orders', 0)}")
        
        elif mode == 'rgb_array':
            # 返回可视化图像
            # TODO: 实现图像渲染
            pass
    
    def close(self):
        """关闭环境，清理资源"""
        if self.sim_env:
            # 清理仿真环境
            self.sim_env = None
        
        logger.info("环境已关闭")


def test_rl_environment():
    """测试RL环境"""
    print("="*60)
    print("测试 DeliveryRLEnvironment")
    print("="*60)
    
    # 配置
    simulation_config = {
        'total_orders': 100,
        'num_couriers': 10,
        'simulation_duration': 7200,  # 2小时
        'dispatch_interval': 30
    }
    
    rl_config = {
        'state_encoder': {
            'max_pending_orders': 10,
            'max_couriers': 10,
            'grid_size': 5
        },
        'reward_calculator': {
            'reward_type': 'dense'
        },
        'action_mode': 'discrete'
    }
    
    # 创建环境
    env = DeliveryRLEnvironment(simulation_config, rl_config)
    
    print(f"\n环境信息:")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 测试reset
    print(f"\n测试 reset()...")
    obs, info = env.reset()
    print(f"  观测形状: {obs.shape}")
    print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Info: {info}")
    
    # 测试step（模拟）
    print(f"\n测试 step()...")
    try:
        action = env.action_space.sample()
        print(f"  随机动作: {action}")
        # obs, reward, terminated, truncated, info = env.step(action)
        # print(f"  奖励: {reward:.4f}")
        # print(f"  终止: {terminated}, 截断: {truncated}")
    except Exception as e:
        print(f"  Step测试跳过（仿真环境未完全集成）: {str(e)}")
    
    # 关闭环境
    env.close()
    
    print("\n测试完成！")
    print("\n注意：完整功能需要在Phase 2中集成实际仿真环境")


if __name__ == "__main__":
    test_rl_environment()

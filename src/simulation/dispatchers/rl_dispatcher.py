"""
RL Dispatcher - 基于强化学习的调度器
使用训练好的PPO模型进行派单决策
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入Stable-Baselines3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3不可用，RLDispatcher将回退到Greedy策略")

# 导入状态编码器
try:
    from src.rl.state_representation import StateEncoder
    STATE_ENCODER_AVAILABLE = True
except ImportError:
    STATE_ENCODER_AVAILABLE = False
    logger.warning("StateEncoder不可用")


class RLDispatcher:
    """
    强化学习调度器
    
    使用训练好的PPO模型进行派单决策：
    1. 将仿真状态编码为RL观测向量
    2. 使用PPO模型预测最优动作
    3. 将动作转换为骑手-订单分配
    
    分层决策架构：
    - Level 1 (RL): 派单决策（订单分配给哪个骑手）
    - Level 2 (启发式): 路径优化（由骑手自动执行）
    """
    
    def __init__(self, 
                 env, 
                 model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化RL调度器
        
        Args:
            env: SimulationEnvironment实例
            model_path: 训练好的PPO模型路径（.zip文件）
            config: 调度器配置
        """
        self.env = env
        self.config = config or {}
        self.model_path = model_path
        self.model = None
        
        # 状态编码器配置 - 必须与训练时的配置一致
        self.max_pending_orders = self.config.get('max_pending_orders', 50)
        self.max_couriers = self.config.get('max_couriers', 50)  # 与训练配置一致
        
        # Hybrid架构配置
        self.include_merchant_features = self.config.get('include_merchant_features', True)
        self.routing_optimizer = self.config.get('routing_optimizer', 'greedy')  # greedy 或 alns
        
        # 初始化真正的状态编码器
        # 从config读取所有参数，确保与训练时的配置一致
        if STATE_ENCODER_AVAILABLE:
            state_encoder_config = {
                'max_pending_orders': self.max_pending_orders,
                'max_couriers': self.max_couriers,
                'grid_size': self.config.get('grid_size', 10),  # 从配置读取，默认10
                'max_merchants': self.config.get('max_merchants', 100),  # 从配置读取，默认100
                'include_merchant_features': self.include_merchant_features  # 支持消融
            }
            self.state_encoder = StateEncoder(state_encoder_config)
            logger.info(f"StateEncoder初始化，状态维度: {self.state_encoder.state_dim}, "
                       f"grid_size: {state_encoder_config['grid_size']}, "
                       f"商家特征: {self.include_merchant_features}")
        else:
            self.state_encoder = None
            logger.warning("StateEncoder不可用，RL模型将无法正常工作")
        
        # 统计计数器
        self.dispatch_count = 0
        self.rl_decisions = 0
        self.fallback_decisions = 0
        self.route_optimizations = 0
        
        # 加载模型
        if model_path and SB3_AVAILABLE:
            self._load_model(model_path)
        else:
            if not SB3_AVAILABLE:
                logger.warning("Stable-Baselines3不可用，使用Greedy回退策略")
            elif not model_path:
                logger.warning("未提供模型路径，使用Greedy回退策略")
        
        logger.info("RLDispatcher初始化完成")
        logger.info(f"  模型路径: {model_path}")
        logger.info(f"  模型加载: {'成功' if self.model else '失败/回退'}")
    
    def _load_model(self, model_path: str) -> bool:
        """
        加载训练好的PPO模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否成功加载
        """
        try:
            path = Path(model_path)
            if path.exists():
                self.model = PPO.load(str(path))
                logger.info(f"成功加载RL模型: {path}")
                return True
            else:
                # 尝试添加.zip后缀
                zip_path = Path(str(model_path) + '.zip')
                if zip_path.exists():
                    self.model = PPO.load(str(zip_path))
                    logger.info(f"成功加载RL模型: {zip_path}")
                    return True
                else:
                    logger.error(f"模型文件不存在: {model_path}")
                    return False
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def dispatch_pending_orders(self) -> int:
        """
        调度所有待分配订单
        
        使用RL模型或回退到Greedy策略
        
        Returns:
            成功分配的订单数
        """
        if len(self.env.pending_orders) == 0:
            return 0
        
        num_pending = len(self.env.pending_orders)
        logger.info(
            f"[{self.env.env.now:.1f}s] RL调度启动，"
            f"待分配订单: {num_pending}"
        )
        
        # 如果模型可用，使用RL决策
        if self.model is not None:
            assigned = self._dispatch_with_rl()
        else:
            # 回退到Greedy策略
            assigned = self._dispatch_with_greedy()
        
        self.dispatch_count += 1
        
        logger.info(
            f"[{self.env.env.now:.1f}s] RL调度完成，"
            f"分配 {assigned} 个订单"
        )
        
        return assigned
    
    def _dispatch_with_rl(self) -> int:
        """
        使用RL模型进行调度
        
        支持两种动作空间模式：
        - discrete: 为单个订单选择骑手
        - multi_discrete: 批量为多个订单决策（训练时使用的模式）
        
        Returns:
            成功分配的订单数
        """
        assigned_count = 0
        pending_copy = list(self.env.pending_orders)
        
        if not pending_copy:
            return 0
        
        # 编码当前状态
        observation = self._encode_state()
        
        # 使用模型预测动作
        action, _ = self.model.predict(observation, deterministic=True)
        
        # 检查动作类型（multi_discrete返回数组，discrete返回标量）
        if isinstance(action, np.ndarray) and action.ndim > 0 and len(action) > 1:
            # multi_discrete模式：action是一个数组，每个元素对应一个订单槽位的骑手分配
            assigned_count = self._decode_multi_discrete_action(action, pending_copy)
        else:
            # discrete模式：action是单个值
            action_scalar = int(action) if isinstance(action, np.ndarray) else action
            for order_id in pending_copy:
                courier_id = self._decode_action(action_scalar, order_id)
                
                if courier_id is not None:
                    if self._assign_order(order_id, courier_id):
                        assigned_count += 1
                        self.rl_decisions += 1
                else:
                    logger.debug(f"RL决策：订单{order_id}暂不分配")
        
        return assigned_count
    
    def _decode_multi_discrete_action(self, action: np.ndarray, pending_orders: List[int]) -> int:
        """
        解码multi_discrete动作
        
        动作空间设计（与训练时一致）：
        - action[i] 表示第i个订单槽位的分配决策
        - action[i] < num_couriers: 分配给对应骑手
        - action[i] == num_couriers: 延迟派单（不分配）
        
        Args:
            action: RL模型输出的动作数组，形状为 (max_pending_orders,)
            pending_orders: 当前待分配订单ID列表
            
        Returns:
            成功分配的订单数
        """
        assigned_count = 0
        couriers = list(self.env.couriers.values())
        num_couriers = len(couriers)
        
        # 遍历待分配订单
        for i, order_id in enumerate(pending_orders):
            if i >= len(action):
                # 超出动作数组范围，跳过
                break
            
            action_i = int(action[i])
            
            # 检查是否延迟派单
            if action_i >= num_couriers:
                logger.debug(f"RL决策：订单{order_id}延迟派单 (action={action_i})")
                continue
            
            # 获取目标骑手
            if action_i < len(couriers):
                courier = couriers[action_i]
                
                # 检查骑手是否可以接单
                if courier.can_accept_new_order():
                    if self._assign_order(order_id, courier.courier_id):
                        assigned_count += 1
                        self.rl_decisions += 1
                        logger.debug(f"RL决策：订单{order_id}分配给骑手{courier.courier_id}")
                else:
                    # 骑手不可用，尝试找替代
                    available = [c for c in couriers if c.can_accept_new_order()]
                    if available:
                        fallback = min(available, key=lambda c: len(c.assigned_orders))
                        if self._assign_order(order_id, fallback.courier_id):
                            assigned_count += 1
                            self.fallback_decisions += 1
                            logger.debug(f"RL决策：订单{order_id}回退分配给骑手{fallback.courier_id}")
        
        return assigned_count
    
    def _dispatch_with_greedy(self) -> int:
        """
        回退到Greedy策略进行调度
        
        Returns:
            成功分配的订单数
        """
        assigned_count = 0
        pending_copy = list(self.env.pending_orders)
        
        for order_id in pending_copy:
            order = self.env.orders[order_id]
            
            # 找最近的可用骑手
            available_couriers = [
                c for c in self.env.couriers.values()
                if c.is_available()
            ]
            
            if not available_couriers:
                continue
            
            # 选择最近骑手
            nearest = self._find_nearest_courier(available_couriers, order.merchant_node)
            
            if nearest is not None:
                if self._assign_order(order_id, nearest.courier_id):
                    assigned_count += 1
                    self.fallback_decisions += 1
        
        return assigned_count
    
    def _encode_state(self) -> np.ndarray:
        """
        编码当前仿真状态为RL观测向量
        
        使用与训练一致的StateEncoder来确保维度匹配
        
        Returns:
            状态向量 (numpy array)
        """
        if self.state_encoder is None:
            raise RuntimeError("StateEncoder不可用，无法编码状态")
        
        # 构建环境状态字典，供StateEncoder使用
        current_time = self.env.env.now
        
        # 获取待分配订单列表
        pending_orders = []
        for order_id in self.env.pending_orders:
            order = self.env.orders.get(order_id)
            if order:
                pending_orders.append(order)
        
        # 获取地理边界
        bounds = self._get_geographic_bounds()
        
        # 构建状态字典
        env_state = {
            'current_time': current_time,
            'pending_orders': pending_orders,
            'couriers': self.env.couriers,
            'bounds': bounds,
            'merchants': getattr(self.env, 'merchants', {})
        }
        
        # 使用StateEncoder编码
        state_vector = self.state_encoder.encode(env_state)
        
        return state_vector
    
    def _get_geographic_bounds(self) -> tuple:
        """
        获取地理边界（用于空间归一化）
        
        Returns:
            (min_x, max_x, min_y, max_y)
        """
        # 尝试从订单中提取边界
        all_x = []
        all_y = []
        
        for order in self.env.orders.values():
            if hasattr(order, 'merchant_lng') and order.merchant_lng:
                all_x.append(order.merchant_lng)
                all_y.append(order.merchant_lat)
            if hasattr(order, 'customer_lng') and order.customer_lng:
                all_x.append(order.customer_lng)
                all_y.append(order.customer_lat)
        
        if all_x and all_y:
            return (min(all_x), max(all_x), min(all_y), max(all_y))
        else:
            # 默认上海区域边界
            return (121.3, 121.6, 31.1, 31.4)
    
    def _decode_action(self, action: int, order_id: int) -> Optional[int]:
        """
        将RL动作解码为骑手ID
        
        动作空间设计：
        - action < num_couriers: 分配给对应骑手
        - action >= num_couriers: 延迟派单（不分配）
        
        Args:
            action: RL模型输出的动作
            order_id: 当前订单ID
            
        Returns:
            骑手ID，如果不分配则返回None
        """
        couriers = list(self.env.couriers.values())
        num_couriers = len(couriers)
        
        # 如果动作表示延迟派单
        if action >= num_couriers:
            return None
        
        # 获取对应骑手
        if action < len(couriers):
            courier = couriers[action]
            # 检查骑手是否可以接单
            if courier.can_accept_new_order():
                return courier.courier_id
        
        # 如果指定骑手不可用，尝试找替代
        available = [c for c in couriers if c.can_accept_new_order()]
        if available:
            # 选择负载最小的骑手
            return min(available, key=lambda c: len(c.assigned_orders)).courier_id
        
        return None
    
    def _find_nearest_courier(self, couriers: List, target_node) -> Optional[Any]:
        """
        找到最近的骑手
        
        Args:
            couriers: 骑手列表
            target_node: 目标节点（商家位置）
            
        Returns:
            最近的骑手，如果都不可达则返回None
        """
        best_courier = None
        best_distance = float('inf')
        
        for courier in couriers:
            try:
                distance = self.env.get_distance(courier.current_node, target_node)
                if distance < best_distance:
                    best_distance = distance
                    best_courier = courier
            except Exception:
                continue
        
        return best_courier
    
    def _assign_order(self, order_id: int, courier_id: int) -> bool:
        """
        将订单分配给骑手
        
        Args:
            order_id: 订单ID
            courier_id: 骑手ID
            
        Returns:
            是否分配成功
        """
        from ..entities import OrderStatus
        
        order = self.env.orders.get(order_id)
        courier = self.env.couriers.get(courier_id)
        
        if order is None or courier is None:
            return False
        
        # 检查骑手容量
        if not courier.can_accept_new_order():
            return False
        
        # 更新订单状态
        order.status = OrderStatus.ASSIGNED
        order.assigned_courier_id = courier_id
        order.assigned_time = self.env.env.now
        
        # 更新骑手
        courier.assign_order(order_id)
        
        # 添加任务到骑手路线
        courier.current_route.append(('pickup', order_id, order.merchant_node))
        courier.current_route.append(('delivery', order_id, order.customer_node))
        
        # Hybrid架构：调用底层路径优化器
        self._optimize_route(courier)
        
        # 更新环境队列
        if order_id in self.env.pending_orders:
            self.env.pending_orders.remove(order_id)
        self.env.assigned_orders.append(order_id)
        
        # 记录事件
        self.env.record_event(
            'order_assigned',
            order_id,
            {
                'courier_id': courier_id,
                'merchant_node': order.merchant_node,
                'customer_node': order.customer_node,
                'assignment_time': self.env.env.now,
                'method': 'RL-PPO'
            }
        )
        
        logger.debug(
            f"[{self.env.env.now:.1f}s] RL分配: 订单{order_id} -> 骑手{courier_id}"
        )
        
        return True
    
    def _optimize_route(self, courier) -> None:
        """
        Hybrid架构的底层路径优化
        
        根据routing_optimizer配置选择优化策略:
        - greedy: 简单的最近邻插入（保持当前顺序，无优化）
        - alns: 使用ALNS重新排序路线
        
        Args:
            courier: 骑手对象
        """
        if len(courier.current_route) <= 2:
            # 路线太短，无需优化
            return
        
        if self.routing_optimizer == 'greedy':
            # Greedy模式：使用贪婪插入优化
            self._greedy_route_optimization(courier)
        elif self.routing_optimizer == 'alns':
            # ALNS模式：使用ALNS重新优化路线
            self._alns_route_optimization(courier)
        
        self.route_optimizations += 1
    
    def _greedy_route_optimization(self, courier) -> None:
        """
        贪婪路径优化：按最近邻顺序重排路线
        
        确保满足取送约束（每个订单先取后送）
        
        Args:
            courier: 骑手对象
        """
        if len(courier.current_route) <= 2:
            return
        
        # 提取所有订单及其取送任务
        order_tasks = {}  # order_id -> {'pickup': task, 'delivery': task}
        for task in courier.current_route:
            action, order_id, node = task
            if order_id not in order_tasks:
                order_tasks[order_id] = {}
            order_tasks[order_id][action] = task
        
        # 使用最近邻启发式重排
        optimized_route = []
        current_node = courier.current_node
        remaining_orders = set(order_tasks.keys())
        picked_up = set()  # 已取货的订单
        
        while remaining_orders or picked_up:
            best_task = None
            best_distance = float('inf')
            
            # 优先考虑已取货订单的配送
            for order_id in list(picked_up):
                # 检查该订单是否有delivery任务
                if 'delivery' not in order_tasks[order_id]:
                    picked_up.discard(order_id)
                    continue
                task = order_tasks[order_id]['delivery']
                try:
                    dist = self.env.get_distance(current_node, task[2])
                    if dist < best_distance:
                        best_distance = dist
                        best_task = task
                except Exception:
                    continue
            
            # 考虑取货任务
            if best_task is None or (remaining_orders and best_distance > 1000):
                for order_id in list(remaining_orders):
                    # 检查该订单是否有pickup任务（可能已经完成）
                    if 'pickup' not in order_tasks[order_id]:
                        continue
                    task = order_tasks[order_id]['pickup']
                    try:
                        dist = self.env.get_distance(current_node, task[2])
                        if dist < best_distance:
                            best_distance = dist
                            best_task = task
                    except Exception:
                        continue
            
            if best_task is None:
                break
            
            optimized_route.append(best_task)
            current_node = best_task[2]
            
            action, order_id, _ = best_task
            if action == 'pickup':
                remaining_orders.discard(order_id)
                picked_up.add(order_id)
            else:
                picked_up.discard(order_id)
        
        # 更新骑手路线
        if len(optimized_route) == len(courier.current_route):
            courier.current_route = optimized_route
            logger.debug(f"骑手{courier.courier_id}路线已优化(Greedy)")
    
    def _alns_route_optimization(self, courier) -> None:
        """
        ALNS路径优化：使用破坏-修复算子优化路线
        
        Args:
            courier: 骑手对象
        """
        if len(courier.current_route) <= 4:
            # 任务太少，用贪婪即可
            self._greedy_route_optimization(courier)
            return
        
        # 尝试导入ALNS相关逻辑
        try:
            # 简化版ALNS：执行2-opt交换
            route = courier.current_route.copy()
            improved = True
            max_iterations = 10
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                
                for i in range(len(route) - 2):
                    for j in range(i + 2, len(route)):
                        # 检查交换是否违反取送约束
                        if self._is_swap_valid(route, i, j):
                            # 计算交换前后的总距离
                            old_dist = self._calculate_route_distance(route, courier.current_node)
                            new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                            new_dist = self._calculate_route_distance(new_route, courier.current_node)
                            
                            if new_dist < old_dist:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
            
            courier.current_route = route
            logger.debug(f"骑手{courier.courier_id}路线已优化(ALNS 2-opt, {iteration}次迭代)")
        except Exception as e:
            logger.warning(f"ALNS优化失败，回退到Greedy: {e}")
            self._greedy_route_optimization(courier)
    
    def _is_swap_valid(self, route: List, i: int, j: int) -> bool:
        """
        检查2-opt交换是否违反取送约束
        
        每个订单的取货必须在配送之前
        """
        # 检查交换后的路线
        new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
        
        # 验证每个订单的取送顺序
        pickup_positions = {}
        delivery_positions = {}
        
        for idx, (action, order_id, _) in enumerate(new_route):
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
    
    def _calculate_route_distance(self, route: List, start_node) -> float:
        """计算路线总距离"""
        total = 0.0
        current = start_node
        
        for _, _, node in route:
            try:
                total += self.env.get_distance(current, node)
                current = node
            except Exception:
                total += 10000  # 不可达时给予高惩罚
        
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            统计信息字典
        """
        total_decisions = self.rl_decisions + self.fallback_decisions
        return {
            'dispatch_count': self.dispatch_count,
            'rl_decisions': self.rl_decisions,
            'fallback_decisions': self.fallback_decisions,
            'route_optimizations': self.route_optimizations,
            'rl_ratio': self.rl_decisions / max(total_decisions, 1),
            'model_loaded': self.model is not None,
            'routing_optimizer': self.routing_optimizer,
            'include_merchant_features': self.include_merchant_features
        }

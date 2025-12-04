"""
ALNS Dispatcher - Day 6 实现
自适应大邻域搜索 (Adaptive Large Neighborhood Search) 调度器

基于alns库实现动态VRP问题求解
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

# 仅支持alns>=7.x
from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations
from alns.select import RouletteWheel

# 【修复】将实体类导入移到文件顶部，避免函数内重复导入
from src.simulation.entities import OrderStatus, CourierStatus

logger = logging.getLogger(__name__)


class VRPSolution(State):
    """
    VRP问题的解表示
    
    解结构：
    - routes: {courier_id: [(action, order_id, node_id), ...]}
    - unassigned: [order_ids]
    - fixed_orders: 已分配订单集合，不参与destroy/repair操作
    - objective_value: 总成本
    """
    
    def __init__(self, routes: Dict[int, List[Tuple[str, int, int]]], 
                 unassigned: List[int], env_ref,
                 fixed_orders: Optional[Set[int]] = None):
        """
        初始化解
        
        Args:
            routes: 骑手路线字典
            unassigned: 未分配订单列表
            env_ref: 仿真环境引用
            fixed_orders: 已分配订单集合（不可移动）
        """
        self.routes = routes
        self.unassigned = unassigned
        self.env = env_ref
        self.fixed_orders = fixed_orders or set()  # 已分配订单，不参与优化
        self._objective_value = None
    
    def objective(self) -> float:
        """计算目标函数值（缓存）"""
        if self._objective_value is None:
            self._objective_value = self._calculate_objective()
        return self._objective_value
    
    def _calculate_objective(self) -> float:
        """
        计算目标函数
        
        组成：
        1. 总行驶距离
        2. 时间窗违背惩罚
        3. 未分配订单惩罚
        """
        total_cost = 0.0
        
        # 1. 行驶距离成本
        for courier_id, route in self.routes.items():
            if not route:
                continue
            
            courier = self.env.couriers[courier_id]
            current_pos = courier.current_node
            
            for action, order_id, target_node in route:
                try:
                    dist = self.env.get_distance(current_pos, target_node)
                    total_cost += dist
                    current_pos = target_node
                except Exception:
                    # 节点不可达，给予大惩罚
                    total_cost += 999999.0
        
        # 2. 时间窗违背惩罚
        current_time = self.env.env.now
        timeout_penalty = 0.0
        
        for courier_id, route in self.routes.items():
            if not route:
                continue
            
            courier = self.env.couriers[courier_id]
            current_pos = courier.current_node
            
            # 【修复3】考虑骑手当前任务剩余时间
            # 如果骑手正在移动，需要加上剩余移动时间
            simulated_time = current_time
            if hasattr(courier, 'busy_until') and courier.busy_until > current_time:
                simulated_time = courier.busy_until
            
            for action, order_id, target_node in route:
                try:
                    # 行程时间
                    travel_time = self.env.get_travel_time(
                        current_pos, target_node, courier.speed_kph
                    )
                    arrival_time = simulated_time + travel_time
                    
                    # 检查时间窗
                    order = self.env.orders[order_id]
                    
                    # 考虑等待时间（Early Arrival）
                    # 如果骑手提前到达，需要等待到最早开始时间
                    if action == 'pickup':
                        earliest_time = getattr(order, 'earliest_pickup_time', 0)
                        start_time = max(arrival_time, earliest_time)
                        # 等待时间惩罚（骑手运力浪费）
                        # 【统一系数】与局部估算保持一致，1秒等待成本=1单位
                        wait_time = start_time - arrival_time
                        if wait_time > 0:
                            timeout_penalty += wait_time * 1.0  # 每秒等待惩罚1单位
                    else:
                        start_time = arrival_time
                    
                    # 【问题2修复】动态获取服务时间，而非硬编码
                    if action == 'pickup':
                        service_time = getattr(order, 'pickup_service_time', 120.0)
                    else:
                        service_time = getattr(order, 'delivery_service_time', 120.0)
                    
                    simulated_time = start_time + service_time
                    
                    if action == 'delivery':
                        if simulated_time > order.latest_delivery_time:
                            # 超时惩罚
                            overtime = simulated_time - order.latest_delivery_time
                            timeout_penalty += overtime * 10.0  # 每秒10单位惩罚
                    
                    current_pos = target_node
                except Exception:
                    # 计算失败，给予惩罚
                    timeout_penalty += 10000.0
        
        total_cost += timeout_penalty
        
        # 3. 未分配订单惩罚
        unassigned_penalty = len(self.unassigned) * 50000.0
        total_cost += unassigned_penalty
        
        return total_cost
    
    def copy(self):
        """浅拷贝解（性能优化：避免deepcopy）"""
        # 使用字典推导 + 列表切片实现浅拷贝，比deepcopy快得多
        new_routes = {k: v[:] for k, v in self.routes.items()}
        return VRPSolution(
            routes=new_routes,
            unassigned=self.unassigned.copy(),
            env_ref=self.env,
            fixed_orders=self.fixed_orders.copy()
        )
    
    def get_assigned_orders(self) -> Set[int]:
        """获取所有已分配的订单ID"""
        assigned = set()
        for route in self.routes.values():
            for action, order_id, node_id in route:
                assigned.add(order_id)
        return assigned
    
    def get_movable_orders(self) -> Set[int]:
        """获取可移动的订单ID（不包括fixed_orders）"""
        all_assigned = self.get_assigned_orders()
        return all_assigned - self.fixed_orders


class ALNSDispatcher:
    """
    ALNS调度器
    
    使用自适应大邻域搜索求解动态VRP问题
    """
    
    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        """
        初始化ALNS调度器
        
        Args:
            env: SimulationEnvironment实例
            config: 配置字典
        """
        self.env = env
        self.config = config or {}
        
        # ALNS参数
        self.iterations = self.config.get('iterations', 200)
        self.destroy_degree_min = self.config.get('destroy_degree_min', 0.1)
        self.destroy_degree_max = self.config.get('destroy_degree_max', 0.4)
        self.temperature_start = self.config.get('temperature_start', 10000.0)
        self.temperature_end = self.config.get('temperature_end', 1.0)
        self.temperature_decay = self.config.get('temperature_decay', 0.95)
        
        # 权重参数
        self.weight_best = self.config.get('weight_best', 10)
        self.weight_better = self.config.get('weight_better', 5)
        self.weight_accepted = self.config.get('weight_accepted', 1)
        self.decay = self.config.get('decay', 0.8)
        
        # 统计
        self.dispatch_count = 0
        self.solve_success_count = 0
        self.solve_failure_count = 0
        self.total_solve_time = 0.0
        
        logger.info("ALNS调度器初始化完成")
        logger.info(f"  迭代次数: {self.iterations}")
        logger.info(f"  破坏程度: {self.destroy_degree_min:.0%} - {self.destroy_degree_max:.0%}")
        logger.info(f"  温度范围: {self.temperature_start:.0f} -> {self.temperature_end:.1f}")
    
    def dispatch_pending_orders(self) -> int:
        """
        调度所有待分配订单（主接口方法）
        
        Returns:
            成功分配的订单数
        """
        if len(self.env.pending_orders) == 0:
            return 0
        
        num_pending = len(self.env.pending_orders)
        logger.info(
            f"[{self.env.env.now:.1f}s] ALNS调度启动，"
            f"待分配订单: {num_pending}"
        )
        
        try:
            import time
            solve_start_time = time.time()
            
            # 1. 构建初始解
            initial_solution = self._build_initial_solution()
            
            if initial_solution is None:
                logger.warning("无法构建初始解")
                return 0
            
            logger.info(f"初始解目标值: {initial_solution.objective():.0f}")
            
            # 2. 初始化ALNS框架
            self._init_alns()
            
            # 3. 使用ALNS优化（alns>=7.x API）
            # 【修复Issue3】使用模拟退火接受准则，允许一定概率接受较差解，避免局部最优
            accept = SimulatedAnnealing(
                start_temperature=self.temperature_start,
                end_temperature=self.temperature_end,
                step=self.temperature_decay
            )
            
            # 停止准则：最大迭代次数
            stop = MaxIterations(self.iterations)
            
            # 算子选择：轮盘赌
            op_select = RouletteWheel(
                [self.weight_best, self.weight_better, self.weight_accepted, 0],
                self.decay, 3, 3
            )
            
            result = self.alns.iterate(
                initial_solution,
                op_select,
                accept,
                stop
            )
            
            best_solution = result.best_state
            
            solve_time = time.time() - solve_start_time
            self.total_solve_time += solve_time
            
            improvement = (1 - best_solution.objective()/initial_solution.objective())*100
            logger.info(
                f"ALNS优化完成 - "
                f"初始: {initial_solution.objective():.0f}, "
                f"最优: {best_solution.objective():.0f}, "
                f"改进: {improvement:.1f}%, "
                f"耗时: {solve_time:.2f}秒"
            )
            
            # 4. 应用解到环境
            assigned_count = self._apply_solution_to_env(best_solution)
            
            self.dispatch_count += 1
            self.solve_success_count += 1
            
            logger.info(
                f"[{self.env.env.now:.1f}s] ALNS调度完成，"
                f"分配 {assigned_count} 个订单"
            )
            
            return assigned_count
            
        except Exception as e:
            logger.error(f"ALNS调度过程出错: {str(e)}")
            logger.exception("详细错误:")
            self.solve_failure_count += 1
            return 0
    
    def _init_alns(self):
        """初始化ALNS框架"""
        np.random.seed(self.config.get('random_seed', 42))
        
        # 使用numpy的RandomState（alns 7.x需要numpy的random）
        rng_state = np.random.RandomState(self.config.get('random_seed', 42))
        self.alns = ALNS(rng_state)
        
        # 注册Destroy算子
        self.alns.add_destroy_operator(self._random_removal)
        self.alns.add_destroy_operator(self._worst_removal)
        self.alns.add_destroy_operator(self._shaw_removal)
        
        # 注册Repair算子
        self.alns.add_repair_operator(self._greedy_insertion)
        self.alns.add_repair_operator(self._regret_insertion)
        self.alns.add_repair_operator(self._time_oriented_insertion)
    
    # ==================== 辅助方法 ====================
    
    def _build_initial_solution(self) -> Optional[VRPSolution]:
        """
        构建初始解（使用贪心策略）
        
        Returns:
            初始解，如果无法构建则返回None
        """
        # 获取可用骑手（有剩余容量的骑手）
        # 【修复】将"是否参与搜索"和"是否还能装更多订单"分开
        # 所有有任务的骑手都应参与搜索，以便ALNS能做跨骑手负载均衡
        # 是否能插入新单由容量检查决定
        active_couriers = [
            c for c in self.env.couriers.values()
            if len(c.current_route) > 0 or c.can_accept_new_order()
        ]
        
        # 可以接受新订单的骑手（用于贪心初始分配）
        available_couriers = [
            c for c in self.env.couriers.values()
            if c.can_accept_new_order()
        ]
        
        if not active_couriers:
            logger.warning("没有活跃骑手")
            return None
        
        # 获取待分配订单（过滤不可达订单）
        reachable_orders = self._filter_reachable_orders(
            list(self.env.pending_orders)
        )
        
        if not reachable_orders and not any(len(c.current_route) > 0 for c in active_couriers):
            logger.warning("没有可达的待分配订单且没有活跃路线")
            return None
        
        # 初始化路线：记录骑手当前已分配订单的任务
        # 【修复2】只有PICKED_UP（在途）订单才是fixed，ASSIGNED订单允许重分配
        routes = {}
        fixed_orders = set()  # 只包含在途订单（已取货未送达）
        
        # 【修复】所有活跃骑手都纳入搜索，包括满载骑手
        for courier in active_couriers:
            # 保留骑手当前路线中的已分配订单任务
            filtered_route = []
            for action, order_id, node_id in courier.current_route:
                order = self.env.orders.get(order_id)
                if order is None:
                    continue
                # 【问题3修复】使用枚举比较，而非.value字符串
                # 保留待执行的任务
                if action == 'pickup' and order.status == OrderStatus.ASSIGNED:
                    filtered_route.append((action, order_id, node_id))
                    # ASSIGNED订单不标记为fixed，允许被重分配
                elif action == 'delivery' and order.status in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                    filtered_route.append((action, order_id, node_id))
                    # 只有PICKED_UP的订单才是fixed（在途，不可移动）
                    if order.status == OrderStatus.PICKED_UP:
                        fixed_orders.add(order_id)
            routes[courier.courier_id] = filtered_route
        
        # 使用贪心策略将pending订单分配到路线中
        unassigned = []
        
        for order_id in reachable_orders:
            order = self.env.orders[order_id]
            
            # 找最佳骑手和插入位置
            best_courier = None
            best_p_pos = None
            best_d_pos = None
            min_cost = float('inf')
            
            for courier in available_couriers:
                # 确保骑手在routes中（可能是空闲骑手，不在active_couriers中）
                if courier.courier_id not in routes:
                    routes[courier.courier_id] = []
                route = routes[courier.courier_id]
                
                # 【修复1】获取骑手初始背包负载
                initial_load = self._get_courier_initial_load(courier)
                
                # 【修复3】删除过早跳过检查，直接在_find_best_pd_insertion中精确计算
                # 查找最佳P&D插入位置对（内部会检查容量约束）
                p_pos, d_pos, cost = self._find_best_pd_insertion(
                    courier, route, order, courier.max_capacity, initial_load
                )
                
                if p_pos is not None and cost < min_cost:
                    min_cost = cost
                    best_courier = courier
                    best_p_pos = p_pos
                    best_d_pos = d_pos
            
            if best_courier is not None:
                # 【修复1】在最佳位置插入P和D（允许拼单）
                route = routes[best_courier.courier_id]
                route.insert(best_p_pos, ('pickup', order_id, order.merchant_node))
                # delivery位置需要+1因为pickup已插入
                route.insert(best_d_pos + 1, ('delivery', order_id, order.customer_node))
            else:
                unassigned.append(order_id)
        
        # 创建解时传入fixed_orders
        solution = VRPSolution(routes, unassigned, self.env, fixed_orders)
        
        logger.debug(
            f"初始解构建完成 - "
            f"分配: {len(reachable_orders)-len(unassigned)}, "
            f"未分配: {len(unassigned)}, "
            f"固定订单: {len(fixed_orders)}"
        )
        
        return solution
    
    def _filter_reachable_orders(self, order_ids: List[int]) -> List[int]:
        """
        过滤可达订单（节点在距离矩阵中或GPS模式下）
        GPS模式下所有订单都是可达的
        """
        # GPS模式下所有订单都可达
        if self.env.use_gps_coords:
            return list(order_ids)
        
        # 路网模式：检查节点是否在距离矩阵中
        reachable = []
        node_mapping_keys = set(str(k) for k in self.env.node_to_idx.keys())
        
        for order_id in order_ids:
            order = self.env.orders[order_id]
            merchant_str = str(order.merchant_node)
            customer_str = str(order.customer_node)
            
            if merchant_str in node_mapping_keys and customer_str in node_mapping_keys:
                reachable.append(order_id)
        
        return reachable
    
    def _get_courier_initial_load(self, courier) -> int:
        """
        【新增】获取骑手当前背包中的订单数（已取货但未送达）
        
        Args:
            courier: 骑手对象
        
        Returns:
            骑手背包中的订单数量
        """
        initial_load = 0
        for order_id in courier.assigned_orders:
            order = self.env.orders.get(order_id)
            if order and order.status == OrderStatus.PICKED_UP:
                initial_load += 1
        return initial_load
    
    def _calculate_max_load(self, route: List, initial_load: int = 0) -> int:
        """
        【修复】计算路线的最大瞬时负载
        
        Args:
            route: 路线任务列表
            initial_load: 骑手初始背包负载（已取货但未送达的订单数）
        
        Returns:
            路线执行过程中的最大同时负载数
        """
        # 【修复1】从初始负载开始计算，而不是从0开始
        current_load = initial_load
        max_load = initial_load
        
        for action, order_id, node_id in route:
            if action == 'pickup':
                current_load += 1
            elif action == 'delivery':
                current_load -= 1
            max_load = max(max_load, current_load)
        
        return max_load
    
    def _check_capacity_with_insertion(self, route: List, pickup_pos: int, delivery_pos: int, 
                                        max_capacity: int, initial_load: int = 0) -> bool:
        """
        【修复】检查在指定位置插入P&D后是否满足容量约束
        
        Args:
            route: 当前路线
            pickup_pos: 取货位置
            delivery_pos: 送货位置（必须 >= pickup_pos）
            max_capacity: 最大容量
            initial_load: 骑手初始背包负载
        
        Returns:
            True如果满足容量约束
        """
        # 构建插入后的路线并计算max_load
        test_route = route[:]
        test_route.insert(pickup_pos, ('pickup', -1, None))
        test_route.insert(delivery_pos + 1, ('delivery', -1, None))  # +1因为pickup已经插入
        
        # 【修复1】从初始负载开始计算
        current_load = initial_load
        max_load = initial_load
        
        for action, _, _ in test_route:
            if action == 'pickup':
                current_load += 1
            elif action == 'delivery':
                current_load -= 1
            max_load = max(max_load, current_load)
        
        return max_load <= max_capacity
    
    def _find_best_pd_insertion(
        self, courier, route: List, order, max_capacity: int, initial_load: int = 0,
        is_executing_first_task: bool = False
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        【修复】查找最佳的Pickup和Delivery插入位置对
        
        允许Pickup和Delivery分开插入，实现拼单优化
        
        Args:
            courier: 骑手对象
            route: 当前路线
            order: 要插入的订单
            max_capacity: 最大容量
            initial_load: 骑手初始背包负载
            is_executing_first_task: 骑手是否正在执行route[0]任务
        
        Returns:
            (best_pickup_pos, best_delivery_pos, min_cost)
            如果无法插入返回 (None, None, inf)
        """
        best_pickup_pos = None
        best_delivery_pos = None
        min_cost = float('inf')
        
        n = len(route)
        
        # 【修复1】如果骑手正在执行第一个任务，则不能在索引0处插入
        # 否则会让骑手"掉头"，在物理上不合理
        start_index = 1 if (is_executing_first_task and n > 0) else 0
        
        # 遍历所有可能的(pickup_pos, delivery_pos)组合，满足 pickup_pos <= delivery_pos
        for p_pos in range(start_index, n + 1):
            for d_pos in range(p_pos, n + 2):  # delivery必须在pickup之后或同位置后
                # 检查容量约束时传入初始负载
                if not self._check_capacity_with_insertion(route, p_pos, d_pos, max_capacity, initial_load):
                    continue
                
                # 计算插入成本（包含时间窗惩罚）
                cost = self._calculate_pd_insertion_cost(courier, route, order, p_pos, d_pos)
                
                if cost < min_cost:
                    min_cost = cost
                    best_pickup_pos = p_pos
                    best_delivery_pos = d_pos
        
        return best_pickup_pos, best_delivery_pos, min_cost
    
    def _calculate_pd_insertion_cost(
        self, courier, route: List, order, pickup_pos: int, delivery_pos: int
    ) -> float:
        """
        【修复3】计算在指定位置分开插入Pickup和Delivery的成本（包含时间窗惩罚）
        
        Args:
            courier: 骑手
            route: 当前路线
            order: 订单
            pickup_pos: 取货插入位置
            delivery_pos: 送货插入位置（原路线索引，不是插入后）
        
        Returns:
            成本增量（距离成本 + 时间窗惩罚）
        """
        try:
            # === 1. 计算距离成本增量 ===
            # 获取pickup位置前后节点
            if pickup_pos == 0:
                prev_p = courier.current_node
            else:
                prev_p = route[pickup_pos - 1][2]
            
            if pickup_pos < len(route):
                next_p = route[pickup_pos][2]
            else:
                next_p = None
            
            # 获取delivery位置前后节点（考虑pickup已插入）
            if delivery_pos == pickup_pos:
                prev_d = order.merchant_node
            elif delivery_pos == 0:
                prev_d = courier.current_node
            elif delivery_pos <= pickup_pos:
                prev_d = route[delivery_pos - 1][2]
            else:
                if delivery_pos - 1 < pickup_pos:
                    prev_d = route[delivery_pos - 1][2]
                elif delivery_pos - 1 == pickup_pos:
                    prev_d = order.merchant_node
                else:
                    prev_d = route[delivery_pos - 2][2]
            
            if delivery_pos < len(route):
                next_d = route[delivery_pos][2]
            else:
                next_d = None
            
            # 计算原距离成本
            original_dist = 0.0
            if next_p is not None and pickup_pos < len(route):
                original_dist += self.env.get_distance(prev_p, next_p)
            if next_d is not None and delivery_pos < len(route) and delivery_pos != pickup_pos:
                if delivery_pos > pickup_pos:
                    d_prev = route[delivery_pos - 1][2] if delivery_pos > 0 else courier.current_node
                    d_next = route[delivery_pos][2] if delivery_pos < len(route) else None
                    if d_next:
                        original_dist += self.env.get_distance(d_prev, d_next)
            
            # 计算新距离成本
            new_dist = 0.0
            new_dist += self.env.get_distance(prev_p, order.merchant_node)
            
            if delivery_pos == pickup_pos:
                new_dist += self.env.get_distance(order.merchant_node, order.customer_node)
                if next_p is not None:
                    new_dist += self.env.get_distance(order.customer_node, next_p)
            else:
                if next_p is not None:
                    new_dist += self.env.get_distance(order.merchant_node, next_p)
                new_dist += self.env.get_distance(prev_d, order.customer_node)
                if next_d is not None:
                    new_dist += self.env.get_distance(order.customer_node, next_d)
            
            distance_cost = new_dist - original_dist
            
            # === 2. 【修复3】估算时间窗惩罚 ===
            # 模拟插入后该订单的送达时间
            time_penalty = self._estimate_delivery_time_penalty(
                courier, route, order, pickup_pos, delivery_pos
            )
            
            # 总成本 = 距离成本 + 时间窗惩罚
            return distance_cost + time_penalty
            
        except Exception as e:
            logger.debug(f"计算P&D插入成本失败: {e}")
            return 999999.0
    
    def _estimate_delivery_time_penalty(
        self, courier, route: List, order, pickup_pos: int, delivery_pos: int
    ) -> float:
        """
        【修复3&4】估算插入订单后的时间窗惩罚
        
        Args:
            courier: 骑手
            route: 当前路线
            order: 要插入的订单
            pickup_pos: 取货插入位置
            delivery_pos: 送货插入位置
        
        Returns:
            时间窗惩罚（超时秒数 * 惩罚系数）
        """
        try:
            current_time = self.env.env.now
            
            # 骑手起始可用时间
            simulated_time = current_time
            if hasattr(courier, 'busy_until') and courier.busy_until > current_time:
                simulated_time = courier.busy_until
            
            current_pos = courier.current_node
            
            # 构建插入后的路线
            test_route = route[:]
            test_route.insert(pickup_pos, ('pickup', order.order_id, order.merchant_node))
            test_route.insert(delivery_pos + 1, ('delivery', order.order_id, order.customer_node))
            
            # 【问题5修复】模拟执行路线，计算所有订单的超时惩罚（与_calculate_route_time_penalty对称）
            total_wait_time = 0.0
            total_overtime_penalty = 0.0
            
            # 记录每个订单的送达时间
            delivery_times = {}
            
            for action, oid, target_node in test_route:
                # 行程时间
                travel_time = self.env.get_travel_time(current_pos, target_node, courier.speed_kph)
                arrival_time = simulated_time + travel_time
                
                # 考虑等待时间
                o = self.env.orders.get(oid)
                if o and action == 'pickup':
                    earliest_time = getattr(o, 'earliest_pickup_time', 0)
                    start_time = max(arrival_time, earliest_time)
                    wait_time = start_time - arrival_time
                    if wait_time > 0:
                        total_wait_time += wait_time
                else:
                    start_time = arrival_time
                
                # 动态获取服务时间
                if o:
                    if action == 'pickup':
                        service_time = getattr(o, 'pickup_service_time', 120.0)
                    else:
                        service_time = getattr(o, 'delivery_service_time', 120.0)
                else:
                    service_time = 120.0
                
                simulated_time = start_time + service_time
                current_pos = target_node
                
                # 【问题5核心修复】记录所有订单的送达时间，而不只是新订单
                if action == 'delivery':
                    delivery_times[oid] = simulated_time
            
            # 【问题5核心修复】计算所有订单的超时惩罚（包括旧订单）
            # 这样与_calculate_route_time_penalty的计算逻辑对称
            for oid, delivery_time in delivery_times.items():
                o = self.env.orders.get(oid)
                if o and delivery_time > o.latest_delivery_time:
                    overtime = delivery_time - o.latest_delivery_time
                    total_overtime_penalty += overtime * 10.0
            
            # 总惩罚 = 超时惩罚 + 等待时间惩罚
            penalty = total_overtime_penalty + total_wait_time * 1.0
            
            return penalty
            
        except Exception as e:
            # 【统一异常处理】与objective计算保持一致，返回大惩罚
            logger.debug(f"计算时间惩罚异常：订单{order.order_id}, 错误: {e}")
            return 10000.0
    
    def _calculate_route_time_penalty(self, courier, route: List) -> float:
        """
        【Issue1修复】计算当前路线的时间惩罚（不含插入）
        用于计算增量成本时作为基准值
        
        Args:
            courier: 骑手
            route: 当前路线
        
        Returns:
            当前路线的总时间惩罚
        """
        if not route:
            return 0.0
        
        try:
            current_time = self.env.env.now
            simulated_time = current_time
            if hasattr(courier, 'busy_until') and courier.busy_until > current_time:
                simulated_time = courier.busy_until
            
            current_pos = courier.current_node
            total_penalty = 0.0
            total_wait_time = 0.0
            
            for action, oid, target_node in route:
                travel_time = self.env.get_travel_time(current_pos, target_node, courier.speed_kph)
                arrival_time = simulated_time + travel_time
                
                o = self.env.orders.get(oid)
                if o and action == 'pickup':
                    earliest_time = getattr(o, 'earliest_pickup_time', 0)
                    start_time = max(arrival_time, earliest_time)
                    wait_time = start_time - arrival_time
                    if wait_time > 0:
                        total_wait_time += wait_time
                else:
                    start_time = arrival_time
                
                if o:
                    if action == 'pickup':
                        service_time = getattr(o, 'pickup_service_time', 120.0)
                    else:
                        service_time = getattr(o, 'delivery_service_time', 120.0)
                else:
                    service_time = 120.0
                
                simulated_time = start_time + service_time
                current_pos = target_node
                
                # 计算超时惩罚
                if o and action == 'delivery':
                    if simulated_time > o.latest_delivery_time:
                        overtime = simulated_time - o.latest_delivery_time
                        total_penalty += overtime * 10.0
            
            # 等待时间惩罚
            total_penalty += total_wait_time * 1.0
            
            return total_penalty
            
        except Exception as e:
            # 【统一异常处理】与objective计算保持一致，返回大惩罚
            logger.debug(f"计算路线时间惩罚异常: {e}")
            return 10000.0
    
    def _apply_solution_to_env(self, solution: VRPSolution) -> int:
        """
        将ALNS解应用到仿真环境
        
        Args:
            solution: ALNS求解得到的最优解
        
        Returns:
            成功分配的订单数
        """
        assigned_count = 0
        
        # 第一步：收集所有骑手应该负责的订单
        courier_orders = {}
        for courier_id, new_route in solution.routes.items():
            orders_in_route = set()
            for action, order_id, node_id in new_route:
                orders_in_route.add(order_id)
            courier_orders[courier_id] = orders_in_route
        
        # 【修复4】第1.5步：检测订单重分配，从原骑手移除
        # 构建订单->骑手的映射（当前状态）
        current_order_courier = {}
        for cid, c in self.env.couriers.items():
            for oid in c.assigned_orders:
                current_order_courier[oid] = cid
        
        # 检测需要重分配的订单
        orders_to_reassign = []
        for new_courier_id, orders in courier_orders.items():
            for order_id in orders:
                if order_id in current_order_courier:
                    old_courier_id = current_order_courier[order_id]
                    if old_courier_id != new_courier_id:
                        orders_to_reassign.append((order_id, old_courier_id, new_courier_id))
        
        # 【问题1修复】完整的订单重分配逻辑
        for order_id, old_courier_id, new_courier_id in orders_to_reassign:
            old_courier = self.env.couriers[old_courier_id]
            new_courier = self.env.couriers[new_courier_id]
            order = self.env.orders.get(order_id)
            
            if order is None:
                continue
            
            # 只能重分配尚未取货的订单（ASSIGNED状态）
            if order.status == OrderStatus.PICKED_UP:
                logger.warning(
                    f"订单 {order_id} 已取货，无法从骑手 {old_courier_id} 移动到 {new_courier_id}"
                )
                continue
            
            # 1. 从原骑手移除
            if order_id in old_courier.assigned_orders:
                old_courier.assigned_orders.remove(order_id)
            
            # 从原骑手路线中移除
            old_courier.current_route = [
                task for task in old_courier.current_route
                if task[1] != order_id
            ]
            
            # 2. 【问题1核心修复】将ASSIGNED订单分配给新骑手
            if order.status == OrderStatus.ASSIGNED:
                # 更新订单归属信息
                order.assigned_courier_id = new_courier_id
                
                # 添加到新骑手的assigned_orders
                if order_id not in new_courier.assigned_orders:
                    new_courier.assign_order(order_id)
                
                # 记录重分配事件
                self.env.record_event(
                    'order_reassigned',
                    order_id,
                    {
                        'old_courier_id': old_courier_id,
                        'new_courier_id': new_courier_id,
                        'reassignment_time': self.env.env.now,
                        'method': 'ALNS'
                    }
                )
                
                logger.info(
                    f"订单 {order_id} 从骑手 {old_courier_id} 重分配到 {new_courier_id}"
                )
        
        # 第二步：更新每个骑手
        for courier_id, new_route in solution.routes.items():
            courier = self.env.couriers[courier_id]
            target_orders = courier_orders[courier_id]
            
            # 分配新订单
            for order_id in target_orders:
                if order_id not in courier.assigned_orders:
                    if order_id in self.env.pending_orders:
                        order = self.env.orders[order_id]
                        
                        # 验证订单状态：只有PENDING状态的订单才能分配
                        if order.status != OrderStatus.PENDING:
                            logger.warning(
                                f"跳过订单 {order_id}：状态为 {order.status}，不是PENDING"
                            )
                            continue
                        
                        # 验证骑手容量
                        if not courier.can_accept_new_order():
                            logger.warning(
                                f"跳过订单 {order_id}：骑手 {courier_id} 已满载"
                            )
                            continue
                        
                        # 更新订单状态
                        order.assign_to_courier(courier_id, self.env.env.now)
                        
                        # 更新骑手状态
                        courier.assign_order(order_id)
                        
                        # 从待分配队列移除
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
                                'method': 'ALNS'
                            }
                        )
                        
                        assigned_count += 1
            
            # 【修复】更新骑手路线 - 信任ALNS计算结果，完全替换路线
            # 只保留正在执行的那一个任务，其他全部由ALNS路线决定
            
            # 1. 保留正在执行的任务（需要验证其有效性）
            current_task = None
            if courier.status != CourierStatus.IDLE and courier.current_route:
                task = courier.current_route[0]
                task_action, task_order_id, task_node = task
                task_order = self.env.orders.get(task_order_id)
                
                # 【关键修复】如果当前任务的订单被ALNS放进了unassigned，不保留它
                # 否则会造成：路线里有这个任务，但订单状态会被回退为PENDING
                if task_order_id in solution.unassigned:
                    logger.info(
                        f"当前任务订单 {task_order_id} 被ALNS移除，不保留为current_task"
                    )
                # 验证当前任务是否仍然有效
                elif task_order is not None:
                    if task_action == 'pickup' and task_order.status in [OrderStatus.PENDING, OrderStatus.ASSIGNED]:
                        current_task = task
                    elif task_action == 'pickup' and task_order.status == OrderStatus.PICKED_UP:
                        # 【关键修复】pickup刚完成，订单变为PICKED_UP
                        # 需要将delivery任务作为当前任务，而不是丢弃
                        logger.info(f"订单{task_order_id}的pickup已完成，切换到delivery任务")
                        current_task = ('delivery', task_order_id, task_order.customer_node)
                    elif task_action == 'delivery' and task_order.status in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP, OrderStatus.DELIVERING]:
                        # DELIVERING表示正在送货中，保留这个任务让骑手继续执行
                        current_task = task
                    else:
                        logger.warning(f"当前任务无效：{task_action} 订单{task_order_id} 状态{task_order.status}")
            
            # 2. 重组路线：先添加正在执行的有效任务
            new_route_list = []
            if current_task:
                new_route_list.append(current_task)
            
            # 3. 添加ALNS计算的路线
            for action, order_id, node_id in new_route:
                # 跳过完全相同的正在执行任务（防止重复）
                if current_task and (action, order_id, node_id) == current_task:
                    continue
                
                # 验证订单状态与任务类型是否匹配（防止无效任务）
                order = self.env.orders.get(order_id)
                if order is None:
                    continue
                
                # pickup任务：只有PENDING或ASSIGNED状态才能取货
                # 注意：PICKING_UP表示正在被另一个骑手取货，不能重复
                if action == 'pickup':
                    if order.status not in [OrderStatus.PENDING, OrderStatus.ASSIGNED]:
                        logger.debug(f"跳过pickup任务：订单{order_id}状态为{order.status}")
                        continue
                # delivery任务：ASSIGNED或PICKED_UP状态才能添加
                # ASSIGNED：pickup还没完成，delivery在路线中等待执行
                # PICKED_UP：pickup完成，可以执行delivery
                # DELIVERING/DELIVERED：已经在送或送完，跳过
                elif action == 'delivery':
                    if order.status not in [OrderStatus.ASSIGNED, OrderStatus.PICKED_UP]:
                        logger.debug(f"跳过delivery任务：订单{order_id}状态为{order.status}")
                        continue
                
                new_route_list.append((action, order_id, node_id))
            
            # 4. 应用新路线
            courier.current_route = new_route_list
            
            if target_orders:
                logger.debug(
                    f"骑手 {courier_id} 更新路线，"
                    f"订单: {target_orders}, "
                    f"路线任务数: {len(new_route_list)}"
                )
        
        # ==========================================
        # 【Fix1】处理未分配订单 (The "Dropped" Orders)
        # ==========================================
        # ALNS可能将某些订单移出所有路线，留在unassigned中
        # 需要将这些ASSIGNED状态的订单回退为PENDING，避免"僵尸订单"
        for order_id in solution.unassigned:
            order = self.env.orders.get(order_id)
            if order is None:
                continue
            
            # 只处理ASSIGNED状态的订单（已分配但被ALNS抛弃）
            if order.status == OrderStatus.ASSIGNED:
                # 找到原骑手并移除
                if hasattr(order, 'assigned_courier_id') and order.assigned_courier_id is not None:
                    old_courier = self.env.couriers.get(order.assigned_courier_id)
                    if old_courier and order_id in old_courier.assigned_orders:
                        old_courier.assigned_orders.remove(order_id)
                        logger.warning(
                            f"订单 {order_id} 被 ALNS 暂时抛弃，从骑手 {old_courier.courier_id} 移除"
                        )
                
                # 回退订单状态为PENDING
                order.status = OrderStatus.PENDING
                order.assigned_courier_id = None
                
                # 【问题2修复】确保它在pending_orders中，以便下轮调度
                # 注意：pending_orders是List类型，使用append而非add
                if order_id not in self.env.pending_orders:
                    self.env.pending_orders.append(order_id)
                
                # 从assigned_orders中移除
                if order_id in self.env.assigned_orders:
                    self.env.assigned_orders.remove(order_id)
        
        # ==========================================
        # 【关键修复】确保所有PICKED_UP订单都有delivery任务
        # ==========================================
        # 在ALNS调度期间，pickup任务可能已经完成，但delivery任务丢失
        # 这里检查并强制添加缺失的delivery任务
        for order_id, order in self.env.orders.items():
            if order.status == OrderStatus.PICKED_UP:
                # 检查是否有任何骑手的路线中包含这个订单的delivery任务
                delivery_found = False
                order_courier_id = order.assigned_courier_id
                
                for courier_id, courier in self.env.couriers.items():
                    for action, oid, _ in courier.current_route:
                        if oid == order_id and action == 'delivery':
                            delivery_found = True
                            break
                    if delivery_found:
                        break
                
                if not delivery_found:
                    # delivery任务丢失！需要强制添加
                    # 找到负责这个订单的骑手
                    if order_courier_id is not None:
                        target_courier = self.env.couriers.get(order_courier_id)
                        if target_courier:
                            # 将delivery任务添加到路线开头（优先执行）
                            delivery_task = ('delivery', order_id, order.customer_node)
                            target_courier.current_route.insert(0, delivery_task)
                            
                            # 同步维护 courier.assigned_orders 和 env.assigned_orders
                            if order_id not in target_courier.assigned_orders:
                                target_courier.assign_order(order_id)
                            if order_id not in self.env.assigned_orders:
                                self.env.assigned_orders.append(order_id)
                            
                            logger.warning(
                                f"【修复】PICKED_UP订单{order_id}的delivery任务丢失，"
                                f"强制添加到骑手{order_courier_id}路线开头"
                            )
                        else:
                            # 原骑手不存在，分配给空闲骑手
                            for cid, c in self.env.couriers.items():
                                if c.status == CourierStatus.IDLE or len(c.current_route) == 0:
                                    delivery_task = ('delivery', order_id, order.customer_node)
                                    c.current_route.append(delivery_task)
                                    order.assigned_courier_id = cid
                                    
                                    # 同步维护 courier.assigned_orders 和 env.assigned_orders
                                    if order_id not in c.assigned_orders:
                                        c.assign_order(order_id)
                                    if order_id not in self.env.assigned_orders:
                                        self.env.assigned_orders.append(order_id)
                                    
                                    logger.warning(
                                        f"【修复】PICKED_UP订单{order_id}的delivery任务丢失，"
                                        f"分配给空闲骑手{cid}"
                                    )
                                    break
                    else:
                        # 没有assigned_courier_id，分配给任意骑手
                        for cid, c in self.env.couriers.items():
                            delivery_task = ('delivery', order_id, order.customer_node)
                            c.current_route.append(delivery_task)
                            order.assigned_courier_id = cid
                            
                            # 同步维护 courier.assigned_orders 和 env.assigned_orders
                            if order_id not in c.assigned_orders:
                                c.assign_order(order_id)
                            if order_id not in self.env.assigned_orders:
                                self.env.assigned_orders.append(order_id)
                            
                            logger.warning(
                                f"【修复】PICKED_UP订单{order_id}无归属且delivery任务丢失，"
                                f"分配给骑手{cid}"
                            )
                            break
        
        return assigned_count
    
    # ==================== Destroy算子 ====================
    
    def _random_removal(self, current: VRPSolution, random_state: Any) -> VRPSolution:
        """
        随机移除算子
        随机选择并移除d%的可移动订单（不包括fixed订单）
        """
        destroyed = current.copy()
        
        # 只收集可移动订单（排除fixed订单）
        assigned_orders = list(destroyed.get_movable_orders())
        
        if not assigned_orders:
            return destroyed
        
        # 确定移除数量
        destroy_degree = random_state.uniform(self.destroy_degree_min, self.destroy_degree_max)
        num_to_remove = max(1, int(len(assigned_orders) * destroy_degree))
        num_to_remove = min(num_to_remove, len(assigned_orders))
        
        # 随机选择订单移除
        orders_to_remove = random_state.choice(assigned_orders, num_to_remove, replace=False).tolist()
        
        # 从路线中移除
        for courier_id, route in destroyed.routes.items():
            destroyed.routes[courier_id] = [
                task for task in route 
                if task[1] not in orders_to_remove
            ]
        
        # 添加到未分配列表
        destroyed.unassigned.extend(orders_to_remove)
        
        return destroyed
    
    def _worst_removal(self, current: VRPSolution, random_state: Any) -> VRPSolution:
        """
        最差移除算子
        移除对目标函数贡献最大的可移动订单（不包括fixed订单）
        """
        destroyed = current.copy()
        
        # 只收集可移动订单（排除fixed订单）
        assigned_orders = list(destroyed.get_movable_orders())
        
        if not assigned_orders:
            return destroyed
        
        # 计算每个订单的成本贡献
        order_costs = []
        
        for order_id in assigned_orders:
            # 估算移除该订单的成本贡献
            cost_contribution = self._estimate_order_cost(order_id)
            order_costs.append((order_id, cost_contribution))
        
        # 按成本降序排序
        order_costs.sort(key=lambda x: x[1], reverse=True)
        
        # 移除成本最高的订单
        destroy_degree = random_state.uniform(self.destroy_degree_min, self.destroy_degree_max)
        num_to_remove = max(1, int(len(assigned_orders) * destroy_degree))
        num_to_remove = min(num_to_remove, len(order_costs))
        
        orders_to_remove = [order_id for order_id, _ in order_costs[:num_to_remove]]
        
        # 从路线中移除
        for courier_id, route in destroyed.routes.items():
            destroyed.routes[courier_id] = [
                task for task in route 
                if task[1] not in orders_to_remove
            ]
        
        destroyed.unassigned.extend(orders_to_remove)
        
        return destroyed
    
    def _shaw_removal(self, current: VRPSolution, random_state: Any) -> VRPSolution:
        """
        Shaw移除算子
        移除空间/时间相关性强的可移动订单（不包括fixed订单）
        """
        destroyed = current.copy()
        
        # 只收集可移动订单（排除fixed订单）
        assigned_orders = list(destroyed.get_movable_orders())
        
        if not assigned_orders:
            return destroyed
        
        # 随机选择一个种子订单
        seed_order_id = random_state.choice(assigned_orders)
        seed_order = self.env.orders[seed_order_id]
        
        # 计算其他订单与种子订单的相关性
        relatedness = []
        
        for order_id in assigned_orders:
            if order_id == seed_order_id:
                continue
            
            order = self.env.orders[order_id]
            
            # 空间距离相关性
            try:
                spatial_dist = self.env.get_distance(
                    seed_order.merchant_node, order.merchant_node
                )
            except Exception:
                spatial_dist = 999999.0
            
            # 时间相关性
            time_diff = abs(order.arrival_time - seed_order.arrival_time)
            
            # 综合相关性得分（距离越近，时间越接近，相关性越高）
            relatedness_score = spatial_dist / 1000.0 + time_diff / 60.0
            
            relatedness.append((order_id, relatedness_score))
        
        # 按相关性排序（越相似排越前，数值越小）
        relatedness.sort(key=lambda x: x[1])
        
        # 计算要移除的订单数量
        destroy_degree = random_state.uniform(self.destroy_degree_min, self.destroy_degree_max)
        num_to_remove = max(1, int(len(assigned_orders) * destroy_degree))
        num_to_remove = min(num_to_remove, len(relatedness) + 1)
        
        # 【Issue3修复】幂律随机选择，避免确定性搜索陷入局部最优
        # 倾向于选择前面（更相似）的订单，但也有概率选择后面的
        orders_to_remove = [seed_order_id]
        pool = relatedness[:]
        
        while len(orders_to_remove) < num_to_remove and pool:
            # 幂律随机选择：random^6 倾向于产生接近0的数
            # power=6 是常用经验值，越大越倾向于头部
            rand_idx = int(len(pool) * (random_state.random() ** 6))
            rand_idx = min(rand_idx, len(pool) - 1)
            
            selected = pool.pop(rand_idx)
            orders_to_remove.append(selected[0])
        
        # 从路线中移除
        for courier_id, route in destroyed.routes.items():
            destroyed.routes[courier_id] = [
                task for task in route 
                if task[1] not in orders_to_remove
            ]
        
        destroyed.unassigned.extend(orders_to_remove)
        
        return destroyed
    
    def _estimate_order_cost(self, order_id: int) -> float:
        """
        【Fix2】估算订单的成本贡献（用于Worst Removal）
        使用乘法因子来放大紧急订单的成本，量纲统一
        
        Args:
            order_id: 订单ID
        
        Returns:
            订单的估算成本贡献
        """
        try:
            order = self.env.orders[order_id]
            
            # 基础成本：订单相关的行程距离
            dist = self.env.get_distance(order.merchant_node, order.customer_node)
            
            # 紧急度因子：剩余时间越少，因子越大
            # 剩余1小时(3600s) -> factor = 1.0
            # 剩余0小时 -> factor = 10.0
            # 已经超时 -> factor > 10.0 (更紧急)
            current_time = self.env.env.now
            time_remaining = order.latest_delivery_time - current_time
            
            # 归一化紧急度因子
            # 注意：time_remaining可以是负数（已超时），这样factor会更大
            urgency_factor = 1.0 + max(0, (3600.0 - time_remaining) / 400.0)
            
            # 成本 = 距离 * 紧急度因子
            # 这样紧急/超时订单会被优先移除和重新插入
            return dist * urgency_factor
            
        except Exception:
            return 10000.0
    
    # ==================== Repair算子 ====================
    
    def _greedy_insertion(self, destroyed: VRPSolution, random_state: Any) -> VRPSolution:
        """
        贪心插入算子
        【修复1&2】按最小成本增量插入未分配订单，允许P&D分开插入
        """
        repaired = destroyed.copy()
        
        if not repaired.unassigned:
            return repaired
        
        # 【Fix3】打乱顺序，增加随机性，避免陷入局部最优
        unassigned_orders = repaired.unassigned[:]
        random_state.shuffle(unassigned_orders)
        
        # 逐个插入未分配订单
        for order_id in unassigned_orders:
            order = self.env.orders[order_id]
            
            best_courier_id = None
            best_p_pos = None
            best_d_pos = None
            min_cost_increase = float('inf')
            
            # 尝试插入到每个骑手的路线中
            for courier_id, route in repaired.routes.items():
                courier = self.env.couriers[courier_id]
                
                # 1. 获取骑手初始背包负载
                initial_load = self._get_courier_initial_load(courier)
                
                # 2. 【修复参数】容量预检查（必须传入initial_load）
                current_max_load = self._calculate_max_load(route, initial_load)
                if current_max_load >= courier.max_capacity:
                    continue  # 骑手已满载，无法插入新的P&D对
                
                # 3. 判断骑手是否正在执行第一个任务
                is_executing = (courier.status != CourierStatus.IDLE and len(route) > 0)
                
                # 4. 【Issue1修复】计算原路线时间惩罚（基准值/沉没成本）
                original_route_penalty = self._calculate_route_time_penalty(courier, route)
                
                # 5. 调用P&D分离插入逻辑
                p_pos, d_pos, total_new_cost = self._find_best_pd_insertion(
                    courier, route, order, courier.max_capacity, initial_load, is_executing
                )
                
                # 6. 【Issue1修复】计算真实增量成本 = 新总成本 - 原时间惩罚
                # total_new_cost = 距离增量 + 新时间惩罚
                # 真实增量 = 距离增量 + (新时间惩罚 - 原时间惩罚) = 距离增量 + 时间惩罚增量
                if p_pos is not None:
                    real_cost_increase = total_new_cost - original_route_penalty
                    if real_cost_increase < min_cost_increase:
                        min_cost_increase = real_cost_increase
                        best_courier_id = courier_id
                        best_p_pos = p_pos
                        best_d_pos = d_pos
            
            # 执行插入
            if best_courier_id is not None:
                route = repaired.routes[best_courier_id]
                # 【修复1】在最佳位置分开插入P和D
                route.insert(best_p_pos, ('pickup', order_id, order.merchant_node))
                route.insert(best_d_pos + 1, ('delivery', order_id, order.customer_node))
                repaired.unassigned.remove(order_id)
        
        return repaired
    
    def _regret_insertion(self, destroyed: VRPSolution, random_state: Any) -> VRPSolution:
        """
        后悔插入算子
        【修复1&2】优先插入"后悔值"最大的订单，允许P&D分开插入
        """
        repaired = destroyed.copy()
        
        if not repaired.unassigned:
            return repaired
        
        # 重复插入，直到无订单可插入
        while repaired.unassigned:
            max_regret = -1
            best_order_id = None
            best_insertion = None  # (cost, courier_id, p_pos, d_pos)
            
            # 计算每个未分配订单的后悔值
            for order_id in repaired.unassigned:
                order = self.env.orders[order_id]
                
                # 找到最优和次优插入位置
                insertion_costs = []
                
                for courier_id, route in repaired.routes.items():
                    courier = self.env.couriers[courier_id]
                    
                    # 1. 获取骑手初始背包负载
                    initial_load = self._get_courier_initial_load(courier)
                    
                    # 2. 【修复参数】容量预检查（必须传入initial_load）
                    current_max_load = self._calculate_max_load(route, initial_load)
                    if current_max_load >= courier.max_capacity:
                        continue  # 骑手已满载
                    
                    # 3. 判断骑手是否正在执行第一个任务
                    is_executing = (courier.status != CourierStatus.IDLE and len(route) > 0)
                    
                    # 4. 【Issue1修复】计算原路线时间惩罚（基准值）
                    original_route_penalty = self._calculate_route_time_penalty(courier, route)
                    
                    # 5. 调用P&D分离插入逻辑
                    p_pos, d_pos, total_new_cost = self._find_best_pd_insertion(
                        courier, route, order, courier.max_capacity, initial_load, is_executing
                    )
                    
                    # 6. 【Issue1修复】使用增量成本
                    if p_pos is not None:
                        real_cost = total_new_cost - original_route_penalty
                        insertion_costs.append((real_cost, courier_id, p_pos, d_pos))
                
                if len(insertion_costs) == 0:
                    continue
                
                # 排序找最优和次优
                insertion_costs.sort(key=lambda x: x[0])
                
                best_cost = insertion_costs[0][0]
                second_best_cost = insertion_costs[1][0] if len(insertion_costs) > 1 else best_cost * 2
                
                # 后悔值 = 次优成本 - 最优成本
                regret = second_best_cost - best_cost
                
                if regret > max_regret:
                    max_regret = regret
                    best_order_id = order_id
                    best_insertion = insertion_costs[0]
            
            # 执行插入
            if best_order_id is not None and best_insertion is not None:
                cost, courier_id, p_pos, d_pos = best_insertion
                order = self.env.orders[best_order_id]
                route = repaired.routes[courier_id]
                
                # 【修复1】在最佳位置分开插入P和D
                route.insert(p_pos, ('pickup', best_order_id, order.merchant_node))
                route.insert(d_pos + 1, ('delivery', best_order_id, order.customer_node))
                repaired.unassigned.remove(best_order_id)
            else:
                # 无法插入更多订单
                break
        
        return repaired
    
    def _time_oriented_insertion(self, destroyed: VRPSolution, random_state: Any) -> VRPSolution:
        """
        时间导向插入算子
        【修复1&2】优先插入时间窗紧迫的订单，允许P&D分开插入
        """
        repaired = destroyed.copy()
        
        if not repaired.unassigned:
            return repaired
        
        current_time = self.env.env.now
        
        # 按紧急程度排序未分配订单
        urgency_list = []
        for order_id in repaired.unassigned:
            order = self.env.orders[order_id]
            time_remaining = order.latest_delivery_time - current_time
            urgency_list.append((order_id, time_remaining))
        
        urgency_list.sort(key=lambda x: x[1])  # 最紧急的在前
        
        # 按紧急程度依次插入
        for order_id, _ in urgency_list:
            order = self.env.orders[order_id]
            
            best_courier_id = None
            best_p_pos = None
            best_d_pos = None
            min_cost = float('inf')
            
            for courier_id, route in repaired.routes.items():
                courier = self.env.couriers[courier_id]
                
                # 1. 获取骑手初始背包负载
                initial_load = self._get_courier_initial_load(courier)
                
                # 2. 【修复参数】容量预检查（必须传入initial_load）
                current_max_load = self._calculate_max_load(route, initial_load)
                if current_max_load >= courier.max_capacity:
                    continue  # 骑手已满载
                
                # 3. 判断骑手是否正在执行第一个任务
                is_executing = (courier.status != CourierStatus.IDLE and len(route) > 0)
                
                # 4. 【Issue1修复】计算原路线时间惩罚（基准值）
                original_route_penalty = self._calculate_route_time_penalty(courier, route)
                
                # 5. 调用P&D分离插入逻辑
                p_pos, d_pos, total_new_cost = self._find_best_pd_insertion(
                    courier, route, order, courier.max_capacity, initial_load, is_executing
                )
                
                # 6. 【Issue1修复】使用增量成本
                if p_pos is not None:
                    real_cost = total_new_cost - original_route_penalty
                    if real_cost < min_cost:
                        min_cost = real_cost
                        best_courier_id = courier_id
                        best_p_pos = p_pos
                        best_d_pos = d_pos
            
            if best_courier_id is not None:
                route = repaired.routes[best_courier_id]
                # 【修复1】在最佳位置分开插入P和D
                route.insert(best_p_pos, ('pickup', order_id, order.merchant_node))
                route.insert(best_d_pos + 1, ('delivery', order_id, order.customer_node))
                repaired.unassigned.remove(order_id)
        
        return repaired
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            统计字典
        """
        stats = {
            'dispatcher_type': 'ALNS',
            'dispatch_count': self.dispatch_count,
            'solve_success_count': self.solve_success_count,
            'solve_failure_count': self.solve_failure_count,
            'average_solve_time': (
                self.total_solve_time / self.solve_success_count
                if self.solve_success_count > 0 else 0.0
            ),
            # ALNS特定参数
            'iterations': self.iterations,
            'destroy_degree_min': self.destroy_degree_min,
            'destroy_degree_max': self.destroy_degree_max,
            'temperature_start': self.temperature_start,
            'temperature_end': self.temperature_end
        }
        
        # 计算成功率
        total_attempts = self.solve_success_count + self.solve_failure_count
        if total_attempts > 0:
            stats['solve_success_rate'] = self.solve_success_count / total_attempts
        else:
            stats['solve_success_rate'] = 0.0
        
        return stats

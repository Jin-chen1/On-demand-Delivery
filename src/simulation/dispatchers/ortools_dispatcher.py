"""
OR-Tools VRP Dispatcher - Day 4 实现
使用 Google OR-Tools 求解带时间窗的车辆路径问题（VRPTW）
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

logger = logging.getLogger(__name__)


class ORToolsDispatcher:
    """
    OR-Tools VRP 调度器
    
    功能：
    1. 将 SimPy 状态（待派订单、骑手状态）转换为 VRP 模型
    2. 调用 OR-Tools 求解 VRPTW
    3. 将求解结果转换为骑手路线并更新仿真状态
    
    策略：滚动时域优化（Rolling Horizon Optimization）
    """
    
    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        """
        初始化 OR-Tools 调度器
        
        Args:
            env: SimulationEnvironment 实例
            config: 调度器配置字典
        """
        self.env = env
        self.config = config or {}
        
        # OR-Tools 求解器配置
        self.time_limit_seconds = self.config.get('time_limit_seconds', 5)
        self.first_solution_strategy = self.config.get(
            'first_solution_strategy',
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        self.local_search_metaheuristic = self.config.get(
            'local_search_metaheuristic',
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        
        # 时间窗松弛配置（优化策略1）
        self.soft_time_windows = self.config.get('soft_time_windows', True)
        self.time_window_slack = self.config.get('time_window_slack', 180.0)  # 默认3分钟松弛（从5分钟优化）
        
        # 分批处理配置（优化策略2）
        self.enable_batching = self.config.get('enable_batching', True)
        self.batch_strategy = self.config.get('batch_strategy', 'adaptive')  # 'fixed', 'adaptive', 'priority'
        self.max_batch_size = self.config.get('max_batch_size', 10)
        self.min_batch_size = self.config.get('min_batch_size', 5)
        
        # 动态插入配置（优化策略3）
        self.allow_insertion_to_active = self.config.get('allow_insertion_to_active', True)  # 允许向非空闲骑手插入订单
        
        # 离线规划模式配置（优化策略4）
        self.offline_mode = self.config.get('offline_mode', False)
        self.offline_time_limit = self.config.get('offline_time_limit', 300)  # 离线规划时间限制（秒）
        self.precomputed_routes = {}  # 预计算的路线 {courier_id: [(order_id, action), ...]}
        self.precomputed_assignments = {}  # 预计算的分配 {order_id: courier_id}
        self.offline_plan_ready = False
        
        # 【新增】距离/时间矩阵缓存（优化策略5）
        self.enable_distance_cache = self.config.get('enable_distance_cache', True)
        self.distance_cache = {}  # {(from_node, to_node): distance}
        self.time_cache = {}  # {(from_node, to_node): time}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 统计计数器
        self.dispatch_count = 0
        self.solve_success_count = 0
        self.solve_failure_count = 0
        self.total_solve_time = 0.0
        self.batch_count = 0  # 分批次数统计
        self.offline_planning_time = 0.0  # 离线规划耗时
        
        mode_str = "离线规划" if self.offline_mode else "在线调度"
        logger.info(f"OR-Tools 调度器初始化完成 ({mode_str}模式)")
        logger.info(f"  时间限制: {self.time_limit_seconds}秒")
        logger.info(f"  时间窗松弛: {self.time_window_slack:.0f}秒")
        if self.offline_mode:
            logger.info(f"  离线规划时间: {self.offline_time_limit}秒")
        else:
            logger.info(f"  动态插入: {'启用' if self.allow_insertion_to_active else '禁用（仅分配给IDLE骑手）'}")
            logger.info(f"  分批处理: {'启用' if self.enable_batching else '禁用'}")
            if self.enable_batching:
                logger.info(f"  分批策略: {self.batch_strategy}")
                logger.info(f"  批次大小: {self.min_batch_size}-{self.max_batch_size}")
    
    def dispatch_pending_orders(self) -> int:
        """
        调度所有待分配订单（接口方法，与 Greedy 保持一致）
        支持分批处理优化和离线规划模式
        
        Returns:
            成功分配的订单数
        """
        if len(self.env.pending_orders) == 0:
            return 0
        
        num_pending = len(self.env.pending_orders)
        logger.info(
            f"[{self.env.env.now:.1f}s] OR-Tools 调度启动，"
            f"待分配订单: {num_pending}"
        )
        
        try:
            # 离线规划模式：使用预计算的路线
            if self.offline_mode:
                return self._dispatch_from_precomputed()
            
            # 在线调度模式
            # 判断是否需要分批处理
            if self.enable_batching and num_pending > self.max_batch_size:
                # 分批处理模式
                return self._dispatch_with_batching()
            else:
                # 全量处理模式
                return self._dispatch_single_batch()
            
        except Exception as e:
            logger.error(f"OR-Tools 调度过程出错: {str(e)}")
            logger.exception("详细错误:")
            self.solve_failure_count += 1
            return 0
    
    def precompute_routes(self, order_ids: List[int] = None) -> bool:
        """
        离线预计算所有订单的最优路线
        在仿真开始前调用，使用更长的求解时间获得更优解
        
        Args:
            order_ids: 要规划的订单ID列表，None表示使用所有已加载订单
            
        Returns:
            是否成功完成预计算
        """
        import time
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("OR-Tools 离线规划开始")
        logger.info("="*60)
        
        # 获取所有订单
        if order_ids is None:
            order_ids = list(self.env.orders.keys())
        
        logger.info(f"  订单数量: {len(order_ids)}")
        logger.info(f"  骑手数量: {len(self.env.couriers)}")
        logger.info(f"  求解时间限制: {self.offline_time_limit}秒")
        
        # 临时保存原始时间限制
        original_time_limit = self.time_limit_seconds
        self.time_limit_seconds = self.offline_time_limit
        
        try:
            # 过滤可达订单
            reachable_orders = self._filter_reachable_orders(order_ids)
            logger.info(f"  可达订单: {len(reachable_orders)}")
            
            if not reachable_orders:
                logger.warning("没有可达订单，离线规划终止")
                return False
            
            # 获取所有骑手
            couriers = list(self.env.couriers.values())
            if not couriers:
                logger.warning("没有可用骑手，离线规划终止")
                return False
            
            # 构建完整的VRP数据模型（包含所有订单）
            vrp_data = self._build_vrp_data_model_offline(reachable_orders, couriers)
            
            if vrp_data is None:
                logger.warning("无法构建VRP模型")
                return False
            
            # 求解VRP
            logger.info(f"开始求解VRP（时限: {self.time_limit_seconds}秒）...")
            solution = self._solve_vrp(vrp_data)
            
            if solution is None:
                logger.warning("OR-Tools 求解失败")
                return False
            
            # 解析解并存储预计算结果
            self._parse_offline_solution(vrp_data, solution)
            
            self.offline_plan_ready = True
            self.offline_planning_time = time.time() - start_time
            
            logger.info("="*60)
            logger.info("OR-Tools 离线规划完成")
            logger.info(f"  规划耗时: {self.offline_planning_time:.1f}秒")
            logger.info(f"  分配订单: {len(self.precomputed_assignments)}")
            logger.info(f"  骑手路线: {len(self.precomputed_routes)}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"离线规划出错: {str(e)}")
            logger.exception("详细错误:")
            return False
        finally:
            # 恢复原始时间限制
            self.time_limit_seconds = original_time_limit
    
    def _build_vrp_data_model_offline(
        self, 
        order_ids: List[int], 
        couriers: List
    ) -> Optional[Dict[str, Any]]:
        """
        为离线规划构建VRP数据模型
        与在线模式类似，但使用所有订单和骑手
        """
        from ..entities import CourierStatus
        
        if not order_ids or not couriers:
            return None
        
        # 构建位置列表
        locations, location_to_idx, order_location_map = self._build_location_list(
            order_ids, couriers
        )
        
        if len(locations) < 2:
            return None
        
        # 构建距离和时间矩阵
        distance_matrix = self._build_distance_matrix_for_vrp(locations)
        time_matrix = self._build_time_matrix_for_vrp(locations)
        
        # 构建时间窗（离线模式使用完整时间窗）
        time_windows = self._build_time_windows_offline(locations, order_location_map)
        
        # 构建取送对
        pickups_deliveries = self._build_pickups_deliveries(order_ids, order_location_map)
        
        # 构建车辆信息
        num_vehicles = len(couriers)
        vehicle_start_indices = []
        vehicle_end_indices = []
        vehicle_capacities = []
        
        for courier in couriers:
            if courier.current_node in location_to_idx:
                start_idx = location_to_idx[courier.current_node]
            else:
                start_idx = 0
            vehicle_start_indices.append(start_idx)
            vehicle_end_indices.append(0)  # 终点回到depot
            vehicle_capacities.append(courier.max_capacity)
        
        # 构建需求列表（取货+1，送货-1）
        demands = [0] * len(locations)  # depot需求为0
        for order_id, loc_map in order_location_map.items():
            demands[loc_map['pickup']] = 1   # 取货增加负载
            demands[loc_map['delivery']] = -1  # 送货减少负载
        
        data = {
            'locations': locations,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix,
            'time_windows': time_windows,
            'pickups_deliveries': pickups_deliveries,
            'demands': demands,
            'vehicle_capacities': vehicle_capacities,
            'num_vehicles': num_vehicles,
            'vehicle_start_indices': vehicle_start_indices,
            'vehicle_end_indices': vehicle_end_indices,
            'depot': 0,
            'order_location_map': order_location_map,
            'available_couriers': couriers,
            'reachable_orders': order_ids
        }
        
        logger.info(
            f"离线VRP模型构建完成 - "
            f"位置数: {len(locations)}, "
            f"订单数: {len(order_ids)}, "
            f"车辆数: {num_vehicles}"
        )
        
        return data
    
    def _build_time_windows_offline(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]]
    ) -> List[Tuple[float, float]]:
        """
        为离线规划构建时间窗（使用宽松时间窗确保可行解）
        离线规划优先保证找到可行解，而非严格遵守时间窗
        """
        # 离线规划使用非常宽松的时间窗，确保有解
        # 所有位置都使用相同的大时间窗
        max_time = 86400.0  # 24小时
        time_windows = [(0.0, max_time)] * len(locations)
        
        # 只为订单设置基本的顺序约束（取货在送货之前）
        for order_id, loc_map in order_location_map.items():
            order = self.env.orders[order_id]
            
            # 取货点：任何时间都可以取货
            pickup_idx = loc_map['pickup']
            earliest_pickup = 0.0
            time_windows[pickup_idx] = (earliest_pickup, max_time)
            
            # 送货点：取货后才能送货
            delivery_idx = loc_map['delivery']
            earliest_delivery = 60.0  # 至少取货后1分钟
            time_windows[delivery_idx] = (earliest_delivery, max_time)
        
        return time_windows
    
    def _parse_offline_solution(self, vrp_data: Dict, solution_dict: Dict) -> None:
        """
        解析离线规划解，存储预计算的路线和分配
        
        Args:
            vrp_data: VRP数据模型
            solution_dict: _solve_vrp返回的解字典，包含solution, manager, routing
        """
        solution = solution_dict['solution']
        manager = solution_dict['manager']
        routing = solution_dict['routing']
        
        order_location_map = vrp_data['order_location_map']
        couriers = vrp_data['available_couriers']
        reachable_orders = vrp_data['reachable_orders']
        
        # 反向映射：位置索引 -> 订单ID
        idx_to_order = {}
        for order_id, loc_map in order_location_map.items():
            idx_to_order[loc_map['pickup']] = (order_id, 'pickup')
            idx_to_order[loc_map['delivery']] = (order_id, 'delivery')
        
        self.precomputed_routes = {}
        self.precomputed_assignments = {}
        
        # 解析每个骑手的路线
        for vehicle_id in range(vrp_data['num_vehicles']):
            courier = couriers[vehicle_id]
            route = []
            
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                if node_index in idx_to_order:
                    order_id, action = idx_to_order[node_index]
                    route.append((order_id, action))
                    
                    if action == 'pickup':
                        self.precomputed_assignments[order_id] = courier.courier_id
                
                index = solution.Value(routing.NextVar(index))
            
            if route:
                self.precomputed_routes[courier.courier_id] = route
        
        logger.info(
            f"离线解析完成 - "
            f"分配{len(self.precomputed_assignments)}个订单到"
            f"{len(self.precomputed_routes)}个骑手"
        )
    
    def _dispatch_from_precomputed(self) -> int:
        """
        从预计算的路线中分配订单
        在仿真运行时调用，快速分配
        """
        from ..entities import OrderStatus, CourierStatus
        
        if not self.offline_plan_ready:
            logger.warning("离线规划未完成，尝试实时规划...")
            # 回退到实时规划
            return self._dispatch_single_batch()
        
        assigned_count = 0
        current_time = self.env.env.now
        
        # 遍历待分配订单
        for order_id in list(self.env.pending_orders):
            order = self.env.orders[order_id]
            
            # 检查是否有预计算分配
            if order_id not in self.precomputed_assignments:
                continue
            
            courier_id = self.precomputed_assignments[order_id]
            
            # 检查骑手是否可用
            if courier_id not in self.env.couriers:
                continue
            
            courier = self.env.couriers[courier_id]
            
            # 检查骑手状态和容量
            if len(courier.assigned_orders) >= courier.max_capacity:
                continue
            
            # 执行分配
            order.status = OrderStatus.ASSIGNED
            order.assigned_courier_id = courier_id
            order.assigned_time = current_time
            
            courier.assign_order(order_id)
            
            # 添加任务到骑手路线（先取货后送货）
            courier.current_route.append(('pickup', order_id, order.merchant_node))
            courier.current_route.append(('delivery', order_id, order.customer_node))
            
            logger.debug(
                f"[{current_time:.1f}s] 骑手{courier_id}分配订单{order_id}, "
                f"路线长度: {len(courier.current_route)}"
            )
            
            # 更新状态
            if order_id in self.env.pending_orders:
                self.env.pending_orders.remove(order_id)
            self.env.assigned_orders.append(order_id)
            
            self.env.record_event(
                'order_assigned',
                order_id,
                {
                    'courier_id': courier_id,
                    'merchant_node': order.merchant_node,
                    'customer_node': order.customer_node,
                    'assignment_time': current_time,
                    'method': 'OR-Tools-Offline'
                }
            )
            
            assigned_count += 1
        
        if assigned_count > 0:
            self.dispatch_count += 1
            self.solve_success_count += 1
            logger.info(
                f"[{current_time:.1f}s] OR-Tools离线分配完成，"
                f"分配 {assigned_count} 个订单"
            )
        
        return assigned_count
    
    def _dispatch_single_batch(self) -> int:
        """
        单批次调度（全量处理）
        
        Returns:
            成功分配的订单数
        """
        # 步骤1: 构建 VRP 数据模型
        vrp_data = self._build_vrp_data_model()
        
        if vrp_data is None:
            logger.warning("无法构建 VRP 模型（可能无可用骑手或订单节点不可达）")
            return 0
        
        # 步骤2: 创建并求解 VRP 模型
        solution = self._solve_vrp(vrp_data)
        
        if solution is None:
            logger.warning("OR-Tools 求解失败，本次调度未分配订单")
            self.solve_failure_count += 1
            return 0
        
        # 步骤3: 解析解并分配给骑手
        assigned_count = self._apply_solution(vrp_data, solution)
        
        self.dispatch_count += 1
        self.solve_success_count += 1
        
        logger.info(
            f"[{self.env.env.now:.1f}s] OR-Tools 调度完成，"
            f"分配 {assigned_count} 个订单"
        )
        
        return assigned_count
    
    def _dispatch_with_batching(self) -> int:
        """
        分批调度（优化策略2）
        
        Returns:
            成功分配的订单总数
        """
        logger.info(f"启用分批处理模式（策略: {self.batch_strategy}）")
        
        # 获取并排序待分配订单
        pending_order_ids = list(self.env.pending_orders)
        
        # 按优先级排序（紧急订单优先）
        sorted_order_ids = self._prioritize_orders(pending_order_ids)
        
        # 确定批次划分
        batches = self._create_batches(sorted_order_ids)
        
        logger.info(f"订单分为 {len(batches)} 个批次处理")
        
        total_assigned = 0
        
        # 逐批处理
        for batch_idx, batch_order_ids in enumerate(batches, 1):
            logger.info(
                f"  处理批次 {batch_idx}/{len(batches)}，"
                f"包含 {len(batch_order_ids)} 个订单"
            )
            
            # 为这一批构建和求解 VRP
            assigned = self._solve_and_assign_batch(batch_order_ids)
            total_assigned += assigned
            
            self.batch_count += 1
            
            if assigned == 0:
                logger.warning(f"批次 {batch_idx} 求解失败，未分配订单")
        
        self.dispatch_count += 1
        
        logger.info(
            f"[{self.env.env.now:.1f}s] 分批调度完成，"
            f"总计分配 {total_assigned} 个订单"
        )
        
        return total_assigned
    
    def _prioritize_orders(self, order_ids: List[int]) -> List[int]:
        """
        按紧急程度排序订单（优先级策略）
        
        Args:
            order_ids: 订单ID列表
        
        Returns:
            排序后的订单ID列表（最紧急的在前）
        """
        current_time = self.env.env.now
        
        orders_with_urgency = []
        for oid in order_ids:
            order = self.env.orders[oid]
            # 计算剩余时间
            time_remaining = order.latest_delivery_time - current_time
            orders_with_urgency.append((oid, time_remaining))
        
        # 按剩余时间升序排序（最紧急的在前）
        orders_with_urgency.sort(key=lambda x: x[1])
        
        return [oid for oid, _ in orders_with_urgency]
    
    def _create_batches(self, order_ids: List[int]) -> List[List[int]]:
        """
        创建订单批次
        
        Args:
            order_ids: 已排序的订单ID列表
        
        Returns:
            批次列表，每个批次是订单ID列表
        """
        if self.batch_strategy == 'fixed':
            # 固定批次大小策略
            return self._create_fixed_batches(order_ids)
        elif self.batch_strategy == 'adaptive':
            # 自适应批次大小策略
            return self._create_adaptive_batches(order_ids)
        else:
            # 默认使用自适应策略
            return self._create_adaptive_batches(order_ids)
    
    def _create_fixed_batches(self, order_ids: List[int]) -> List[List[int]]:
        """
        固定批次大小策略
        
        Args:
            order_ids: 订单ID列表
        
        Returns:
            批次列表
        """
        batches = []
        batch_size = self.max_batch_size
        
        for i in range(0, len(order_ids), batch_size):
            batch = order_ids[i:i+batch_size]
            batches.append(batch)
        
        return batches
    
    def _create_adaptive_batches(self, order_ids: List[int]) -> List[List[int]]:
        """
        自适应批次大小策略（根据订单数和骑手数动态调整）
        
        Args:
            order_ids: 订单ID列表
        
        Returns:
            批次列表
        """
        num_orders = len(order_ids)
        # 根据allow_insertion_to_active配置选择骑手筛选策略
        # 与_build_vrp_data_model保持一致
        if self.allow_insertion_to_active:
            available_couriers = [c for c in self.env.couriers.values() if c.can_accept_new_order()]
        else:
            available_couriers = [c for c in self.env.couriers.values() if c.is_available()]
        num_couriers = len(available_couriers)
        
        if num_couriers == 0:
            # 无可用骑手，返回空批次
            return []
        
        # 计算订单/骑手比例
        ratio = num_orders / num_couriers
        
        # 动态确定批次大小
        if ratio > 3:
            # 订单远多于骑手，使用小批次快速响应
            batch_size = max(self.min_batch_size, min(8, num_couriers * 2))
        elif ratio > 1.5:
            # 中等负载，使用标准批次
            batch_size = self.max_batch_size
        else:
            # 订单少，可以使用大批次或全量
            batch_size = min(num_orders, self.max_batch_size * 2)
        
        # 创建批次
        batches = []
        for i in range(0, num_orders, batch_size):
            batch = order_ids[i:i+batch_size]
            batches.append(batch)
        
        logger.debug(
            f"自适应分批: {num_orders}订单, {num_couriers}骑手, "
            f"比例={ratio:.1f}, 批次大小={batch_size}, 批次数={len(batches)}"
        )
        
        return batches
    
    def _solve_and_assign_batch(self, batch_order_ids: List[int]) -> int:
        """
        求解并分配单个批次的订单
        
        Args:
            batch_order_ids: 批次内的订单ID列表
        
        Returns:
            成功分配的订单数
        """
        # 构建这一批订单的 VRP 模型
        vrp_data = self._build_vrp_data_model(order_subset=batch_order_ids)
        
        if vrp_data is None:
            logger.warning(f"批次构建失败（可能无可用骑手或节点不可达）")
            return 0
        
        # 求解
        solution = self._solve_vrp(vrp_data)
        
        if solution is None:
            self.solve_failure_count += 1
            return 0
        
        # 应用解
        assigned_count = self._apply_solution(vrp_data, solution)
        self.solve_success_count += 1
        
        return assigned_count
    
    def _build_vrp_data_model(self, order_subset: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        构建 OR-Tools VRP 数据模型
        
        【修复】正确处理骑手背包中的订单：
        1. PICKED_UP状态（在途订单）：只建模送货点，锁定给当前骑手
        2. ASSIGNED状态（已分配未取货）：重新加入调度，锁定给当前骑手
        3. 正确计算初始负载（只统计PICKED_UP状态的订单）
        
        Args:
            order_subset: 订单ID子集，如果为None则使用所有待分配订单
        
        返回数据结构：
        {
            'locations': [node_ids],  # 位置节点列表
            'distance_matrix': [[...]],  # 距离矩阵（米）
            'time_matrix': [[...]],  # 时间矩阵（秒）
            'time_windows': [(earliest, latest), ...],  # 时间窗
            'pickups_deliveries': [(pickup_idx, delivery_idx), ...],  # 取送对
            'demands': [0, 1, -1, ...],  # 需求（pickup=+1, delivery=-1）
            'num_vehicles': int,  # 可用车辆数
            'vehicle_capacities': [capacity, ...],  # 车辆容量（已扣减在途订单）
            'vehicle_start_nodes': [node, ...],  # 车辆起始位置
            'depot': 0,  # depot索引（虚拟）
            'on_board_orders': {...},  # 在途订单信息（PICKED_UP状态）
            'vehicle_locked_orders': {...}  # 骑手锁定的订单（包括ASSIGNED和PICKED_UP）
        }
        
        Returns:
            VRP 数据字典，如果无法构建则返回 None
        """
        from ..entities import OrderStatus
        
        logger.debug("开始构建 VRP 数据模型...")
        
        # 根据配置选择骑手筛选策略
        if self.allow_insertion_to_active:
            # 启用动态插入：仅检查容量，不限制状态
            available_couriers = [
                c for c in self.env.couriers.values()
                if c.can_accept_new_order()
            ]
            logger.debug(f"动态插入模式：考虑所有未满载骑手（包括非 IDLE）")
        else:
            # 传统模式：仅分配给IDLE骑手
            available_couriers = [
                c for c in self.env.couriers.values()
                if c.is_available()
            ]
            logger.debug(f"传统模式：仅考虑 IDLE 骑手")
        
        if not available_couriers:
            logger.warning("没有可用骑手")
            return None
        
        # 确定要处理的订单（使用子集或全部待分配订单）
        target_orders = order_subset if order_subset is not None else list(self.env.pending_orders)
        
        # 【修复】收集骑手背包中的订单，分类处理
        # on_board_orders: PICKED_UP状态，已在车上，只需送货
        # assigned_orders_to_replan: ASSIGNED状态，需重新加入调度但锁定骑手
        on_board_orders = {}  # {order_id: {'courier_idx': int, 'courier_id': int, 'delivery_node': int}}
        vehicle_locked_orders = {}  # {vehicle_idx: [order_ids]}
        courier_initial_loads = []  # 每个骑手的初始负载（只统计PICKED_UP）
        assigned_orders_to_replan = {}  # {order_id: courier_idx} ASSIGNED状态订单的锁定关系
        
        for idx, courier in enumerate(available_couriers):
            vehicle_locked_orders[idx] = []
            
            # 【修复】初始负载只统计PICKED_UP状态的订单（真正在车上的）
            initial_load = 0
            
            for order_id in courier.assigned_orders:
                if order_id not in self.env.orders:
                    continue
                order = self.env.orders[order_id]
                
                # 情况1: PICKED_UP状态（在途订单）- 已在车上，只需送货
                if order.status == OrderStatus.PICKED_UP:
                    initial_load += 1  # 计入初始负载
                    on_board_orders[order_id] = {
                        'courier_idx': idx,
                        'courier_id': courier.courier_id,
                        'delivery_node': order.customer_node
                    }
                    vehicle_locked_orders[idx].append(order_id)
                    logger.debug(
                        f"骑手{courier.courier_id}背包中有在途订单{order_id}（PICKED_UP），"
                        f"需送往{order.customer_node}"
                    )
                
                # 【修复】情况2: ASSIGNED状态（已分配未取货）- 重新加入调度，锁定骑手
                elif order.status == OrderStatus.ASSIGNED:
                    # 将其加回target_orders，让VRP决定最佳顺路顺序
                    if order_id not in target_orders:
                        target_orders.append(order_id)
                    # 记录锁定关系，防止被分配给别的骑手
                    assigned_orders_to_replan[order_id] = idx
                    vehicle_locked_orders[idx].append(order_id)
                    logger.debug(
                        f"骑手{courier.courier_id}有已分配订单{order_id}（ASSIGNED），"
                        f"重新加入调度并锁定"
                    )
            
            courier_initial_loads.append(initial_load)
        
        # 【修复】在收集完ASSIGNED订单后，再过滤可达订单
        # 这样ASSIGNED订单也会被包含在过滤范围内
        reachable_orders = self._filter_reachable_orders(target_orders)
        
        if not reachable_orders and not on_board_orders:
            logger.warning("没有可达的待分配订单且没有在途订单")
            return None
        
        logger.info(
            f"可用骑手: {len(available_couriers)}, "
            f"可达订单: {len(reachable_orders)}, "
            f"在途订单: {len(on_board_orders)}, "
            f"已分配待重规划: {len(assigned_orders_to_replan)}"
        )
        
        # 构建位置列表和映射（包含在途订单的送货点）
        locations, location_to_idx, order_location_map = self._build_location_list_with_onboard(
            reachable_orders, available_couriers, on_board_orders
        )
        
        # 构建距离和时间矩阵
        distance_matrix = self._build_distance_matrix_for_vrp(locations)
        time_matrix = self._build_time_matrix_for_vrp(locations)
        
        # 构建时间窗（包含在途订单）
        time_windows = self._build_time_windows_with_onboard(
            locations, order_location_map, on_board_orders
        )
        
        # 构建取送对（包含在途订单的特殊处理）
        pickups_deliveries = self._build_pickups_deliveries_with_onboard(
            reachable_orders, order_location_map, on_board_orders, 
            available_couriers, location_to_idx
        )
        
        # 构建需求向量
        demands = self._build_demands_with_onboard(
            locations, order_location_map, on_board_orders
        )
        
        # 【新增】构建服务时间映射和原始截止时间映射
        service_times, original_deadlines = self._build_service_times_and_deadlines(
            locations, order_location_map, on_board_orders
        )
        
        # 【修复】车辆容量：使用最大容量（不是剩余容量）
        # 起始负载通过 courier_initial_loads 在 _solve_vrp 中设置
        vehicle_capacities = [c.max_capacity for c in available_couriers]
        
        vehicle_start_indices = [
            location_to_idx.get(c.current_node, 0)
            for c in available_couriers
        ]
        # 结束位置：回到depot（虚拟节点，距离为0）
        vehicle_end_indices = [0] * len(available_couriers)
        
        data = {
            'locations': locations,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix,
            'time_windows': time_windows,
            'pickups_deliveries': pickups_deliveries,
            'demands': demands,
            'num_vehicles': len(available_couriers),
            'vehicle_capacities': vehicle_capacities,
            'vehicle_start_indices': vehicle_start_indices,
            'vehicle_end_indices': vehicle_end_indices,
            'depot': 0,
            # 额外信息用于解析
            'location_to_idx': location_to_idx,
            'order_location_map': order_location_map,
            'available_couriers': available_couriers,
            'reachable_orders': reachable_orders,
            # 【新增】在途订单信息
            'on_board_orders': on_board_orders,
            'vehicle_locked_orders': vehicle_locked_orders,
            'courier_initial_loads': courier_initial_loads,
            # 【新增】ASSIGNED状态订单的锁定关系
            'assigned_orders_to_replan': assigned_orders_to_replan,
            # 【新增】服务时间和原始截止时间
            'service_times': service_times,
            'original_deadlines': original_deadlines
        }
        
        logger.debug(
            f"VRP 模型构建完成 - 位置数: {len(locations)}, "
            f"车辆数: {data['num_vehicles']}, "
            f"在途订单: {len(on_board_orders)}"
        )
        
        return data
    
    def _filter_reachable_orders(self, order_ids: List[int]) -> List[int]:
        """
        过滤可达且未分配的订单
        GPS模式下所有订单都是可达的，路网模式下检查节点是否在距离矩阵中
        
        Args:
            order_ids: 订单ID列表
        
        Returns:
            可达且未分配的订单ID列表
        """
        from ..entities import OrderStatus
        
        reachable = []
        
        # GPS模式下所有订单都可达
        use_gps = getattr(self.env, 'use_gps_coords', False)
        
        if use_gps:
            # GPS模式：只检查订单状态
            for order_id in order_ids:
                order = self.env.orders[order_id]
                if order.status == OrderStatus.PENDING:
                    reachable.append(order_id)
                else:
                    logger.debug(f"订单 {order_id} 已被分配，跳过（状态: {order.status}）")
        else:
            # 路网模式：检查节点是否在距离矩阵中
            node_mapping_keys = set(str(k) for k in self.env.node_to_idx.keys())
            
            for order_id in order_ids:
                order = self.env.orders[order_id]
                
                # 检查订单是否已经被分配
                if order.status != OrderStatus.PENDING:
                    logger.debug(f"订单 {order_id} 已被分配，跳过（状态: {order.status}）")
                    continue
                
                merchant_str = str(order.merchant_node)
                customer_str = str(order.customer_node)
                
                if merchant_str in node_mapping_keys and customer_str in node_mapping_keys:
                    reachable.append(order_id)
                else:
                    logger.debug(
                        f"订单 {order_id} 不可达 "
                        f"(merchant: {merchant_str in node_mapping_keys}, "
                        f"customer: {customer_str in node_mapping_keys})"
                    )
        
        return reachable
    
    def _build_location_list(
        self, 
        order_ids: List[int],
        couriers: List
    ) -> Tuple[List[int], Dict[int, int], Dict[int, Dict[str, int]]]:
        """
        构建位置列表和映射
        
        位置顺序：[depot] + [courier_starts] + [pickup1, delivery1, pickup2, delivery2, ...]
        
        Returns:
            (locations, location_to_idx, order_location_map)
        """
        locations = [0]  # depot 虚拟节点
        location_to_idx = {0: 0}
        order_location_map = {}
        
        idx = 1
        
        # 添加骑手起始位置
        for courier in couriers:
            if courier.current_node not in location_to_idx:
                locations.append(courier.current_node)
                location_to_idx[courier.current_node] = idx
                idx += 1
        
        # 添加订单的取送点
        for order_id in order_ids:
            order = self.env.orders[order_id]
            
            # 添加取货点
            if order.merchant_node not in location_to_idx:
                locations.append(order.merchant_node)
                location_to_idx[order.merchant_node] = idx
                idx += 1
            
            pickup_idx = location_to_idx[order.merchant_node]
            
            # 添加送货点
            if order.customer_node not in location_to_idx:
                locations.append(order.customer_node)
                location_to_idx[order.customer_node] = idx
                idx += 1
            
            delivery_idx = location_to_idx[order.customer_node]
            
            order_location_map[order_id] = {
                'pickup': pickup_idx,
                'delivery': delivery_idx
            }
        
        return locations, location_to_idx, order_location_map
    
    def _build_location_list_with_onboard(
        self, 
        order_ids: List[int],
        couriers: List,
        on_board_orders: Dict[int, Dict[str, Any]]
    ) -> Tuple[List[int], Dict[int, int], Dict[int, Dict[str, int]]]:
        """
        构建位置列表和映射（包含在途订单）
        
        【修复】在途订单的处理：
        - 在途订单只添加送货点，不添加虚拟取货点
        - 在途订单不参与 Pickup/Delivery 约束，只需锁定送货点到指定骑手
        
        位置顺序：[depot] + [courier_starts] + [pickup1, delivery1, ...] + [onboard_deliveries]
        
        Args:
            order_ids: 待分配订单ID列表
            couriers: 可用骑手列表
            on_board_orders: 在途订单信息 {order_id: {'courier_idx': int, 'delivery_node': int}}
        
        Returns:
            (locations, location_to_idx, order_location_map)
        """
        locations = [0]  # depot 虚拟节点
        location_to_idx = {0: 0}
        order_location_map = {}
        
        idx = 1
        
        # 添加骑手起始位置
        courier_start_indices = {}  # {courier_idx: location_idx}
        for c_idx, courier in enumerate(couriers):
            if courier.current_node not in location_to_idx:
                locations.append(courier.current_node)
                location_to_idx[courier.current_node] = idx
                courier_start_indices[c_idx] = idx
                idx += 1
            else:
                courier_start_indices[c_idx] = location_to_idx[courier.current_node]
        
        # 添加待分配订单的取送点（包括PENDING和ASSIGNED状态）
        for order_id in order_ids:
            order = self.env.orders[order_id]
            
            # 添加取货点
            if order.merchant_node not in location_to_idx:
                locations.append(order.merchant_node)
                location_to_idx[order.merchant_node] = idx
                idx += 1
            
            pickup_idx = location_to_idx[order.merchant_node]
            
            # 添加送货点
            if order.customer_node not in location_to_idx:
                locations.append(order.customer_node)
                location_to_idx[order.customer_node] = idx
                idx += 1
            
            delivery_idx = location_to_idx[order.customer_node]
            
            order_location_map[order_id] = {
                'pickup': pickup_idx,
                'delivery': delivery_idx,
                'is_onboard': False
            }
        
        # 【修复】在途订单只添加送货点，不添加虚拟取货点
        # 在途订单已经在车上，不需要"取货"动作
        for order_id, onboard_info in on_board_orders.items():
            order = self.env.orders[order_id]
            courier_idx = onboard_info['courier_idx']
            
            # 【修复】只添加送货点，不设置pickup
            if order.customer_node not in location_to_idx:
                locations.append(order.customer_node)
                location_to_idx[order.customer_node] = idx
                idx += 1
            
            delivery_idx = location_to_idx[order.customer_node]
            
            order_location_map[order_id] = {
                'pickup': -1,  # 【修复】标记无取货点
                'delivery': delivery_idx,
                'is_onboard': True,
                'locked_vehicle': courier_idx
            }
        
        return locations, location_to_idx, order_location_map
    
    def _build_time_windows_with_onboard(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        on_board_orders: Dict[int, Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """
        构建时间窗列表（包含在途订单）
        
        【修复】在途订单只有送货点（pickup = -1），不设置取货点时间窗
        
        Args:
            locations: 位置节点列表
            order_location_map: 订单位置映射
            on_board_orders: 在途订单信息
        
        Returns:
            时间窗列表 [(earliest, latest), ...]
        """
        current_time = self.env.env.now
        time_windows = [(0.0, 999999.0)] * len(locations)  # 默认宽松时间窗
        
        # 为每个订单的取送点设置时间窗
        for order_id, loc_map in order_location_map.items():
            order = self.env.orders[order_id]
            is_onboard = loc_map.get('is_onboard', False)
            pickup_idx = loc_map.get('pickup', -1)
            delivery_idx = loc_map.get('delivery', -1)
            
            if is_onboard:
                # 【修复】在途订单：只有送货点（pickup = -1）
                # 送货点时间窗：尽快送达
                if delivery_idx >= 0:
                    earliest_delivery = 0.0  # 可以立即送
                    latest_delivery = order.latest_delivery_time - current_time
                    
                    if self.soft_time_windows:
                        latest_delivery += self.time_window_slack
                    
                    latest_delivery = max(latest_delivery, earliest_delivery + 60.0)
                    time_windows[delivery_idx] = (earliest_delivery, latest_delivery)
            else:
                # 待分配订单（PENDING和ASSIGNED）：正常时间窗
                if pickup_idx >= 0:
                    earliest_pickup = max(order.earliest_pickup_time - current_time, 0.0)
                    latest_pickup = order.latest_delivery_time - current_time - 300.0
                    
                    if self.soft_time_windows:
                        latest_pickup += self.time_window_slack
                    
                    latest_pickup = max(latest_pickup, earliest_pickup + 60.0)
                    time_windows[pickup_idx] = (earliest_pickup, latest_pickup)
                
                # 送货点时间窗
                if delivery_idx >= 0:
                    earliest_pickup = max(order.earliest_pickup_time - current_time, 0.0)
                    earliest_delivery = earliest_pickup + 120.0
                    latest_delivery = order.latest_delivery_time - current_time
                    
                    if self.soft_time_windows:
                        latest_delivery += self.time_window_slack
                    
                    latest_delivery = max(latest_delivery, earliest_delivery + 60.0)
                    time_windows[delivery_idx] = (earliest_delivery, latest_delivery)
        
        return time_windows
    
    def _build_pickups_deliveries_with_onboard(
        self,
        order_ids: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        on_board_orders: Dict[int, Dict[str, Any]],
        couriers: List,
        location_to_idx: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        构建取送对列表
        
        【修复】在途订单不添加取送对约束：
        - 在途订单已在车上，不需要 Pickup/Delivery 约束
        - 只需在 _solve_vrp 中锁定送货点到指定骑手
        
        Args:
            order_ids: 待分配订单ID列表
            order_location_map: 订单位置映射
            on_board_orders: 在途订单信息
            couriers: 可用骑手列表
            location_to_idx: 位置到索引的映射
        
        Returns:
            取送对列表 [(pickup_idx, delivery_idx), ...]
        """
        pickups_deliveries = []
        
        # 【修复】只添加待分配订单（PENDING和ASSIGNED）的取送对
        # 在途订单（PICKED_UP）不添加取送对
        for order_id in order_ids:
            # 跳过在途订单
            if order_id in on_board_orders:
                continue
            
            loc_map = order_location_map.get(order_id)
            if loc_map and loc_map.get('pickup', -1) >= 0:
                pickups_deliveries.append((loc_map['pickup'], loc_map['delivery']))
        
        return pickups_deliveries
    
    def _build_demands_with_onboard(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        on_board_orders: Dict[int, Dict[str, Any]]
    ) -> List[int]:
        """
        构建需求向量（包含在途订单）
        
        【修复】容量建模说明（真实负载法）：
        - Vehicle Capacity = 最大容量（例如 10）
        - Start Load = 初始负载（在途订单数，在 _solve_vrp 中设置）
        - 待分配订单：Pickup = +1（增加负载），Delivery = -1（减少负载）
        - 在途订单：只有 Delivery = -1（卸货减少负载）
        - 在途订单没有 Pickup 节点（pickup = -1）
        
        示例：骑手初始负载=2，最大容量=5
        - 送货（在途）：2 + (-1) = 1 ✓
        - 取货（新单）：1 + (+1) = 2 ✓
        - 送货（新单）：2 + (-1) = 1 ✓
        
        Args:
            locations: 位置节点列表
            order_location_map: 订单位置映射
            on_board_orders: 在途订单信息
        
        Returns:
            需求列表 [0, 1, -1, ...]
        """
        demands = [0] * len(locations)
        
        for order_id, loc_map in order_location_map.items():
            is_onboard = loc_map.get('is_onboard', False)
            pickup_idx = loc_map.get('pickup', -1)
            delivery_idx = loc_map.get('delivery', -1)
            
            if is_onboard:
                # 在途订单：只有送货点，demand = -1（卸货减少负载）
                if delivery_idx >= 0:
                    demands[delivery_idx] = -1
            else:
                # 待分配订单：正常demand
                if pickup_idx >= 0:
                    demands[pickup_idx] = 1   # 取货增加负载
                if delivery_idx >= 0:
                    demands[delivery_idx] = -1  # 送货减少负载
        
        return demands
    
    def _build_service_times_and_deadlines(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        on_board_orders: Dict[int, Dict[str, Any]]
    ) -> Tuple[List[int], Dict[int, int]]:
        """
        构建服务时间映射和原始截止时间映射
        
        【新增】服务时间：商家出餐/取货等待、联系顾客/上楼配送的时间
        在 O2O 配送中，服务时间通常占据总时间的 30%-50%
        
        Args:
            locations: 位置节点列表
            order_location_map: 订单位置映射
            on_board_orders: 在途订单信息
        
        Returns:
            service_times: 每个位置的服务时间（秒）
            original_deadlines: 每个位置的原始截止时间（相对于当前时间的秒数）
        """
        current_time = self.env.env.now
        
        # 默认服务时间配置（秒）
        default_pickup_service_time = self.config.get('pickup_service_time', 180)  # 取货等待3分钟
        default_delivery_service_time = self.config.get('delivery_service_time', 120)  # 送货2分钟
        
        # 初始化服务时间列表（depot 和骑手起点的服务时间为0）
        service_times = [0] * len(locations)
        
        # 原始截止时间映射（只记录送货点的截止时间）
        original_deadlines = {}
        
        for order_id, loc_map in order_location_map.items():
            order = self.env.orders[order_id]
            is_onboard = loc_map.get('is_onboard', False)
            pickup_idx = loc_map.get('pickup', -1)
            delivery_idx = loc_map.get('delivery', -1)
            
            # 获取订单的服务时间（如果订单有自定义值则使用，否则用默认值）
            pickup_service = getattr(order, 'pickup_service_time', default_pickup_service_time)
            delivery_service = getattr(order, 'delivery_service_time', default_delivery_service_time)
            
            if is_onboard:
                # 在途订单：只有送货点
                if delivery_idx >= 0:
                    service_times[delivery_idx] = int(delivery_service)
                    # 记录原始截止时间（相对于当前时间）
                    original_deadlines[delivery_idx] = int(order.latest_delivery_time - current_time)
            else:
                # 待分配订单：取货点和送货点
                if pickup_idx >= 0:
                    service_times[pickup_idx] = int(pickup_service)
                if delivery_idx >= 0:
                    service_times[delivery_idx] = int(delivery_service)
                    # 记录原始截止时间（相对于当前时间）
                    original_deadlines[delivery_idx] = int(order.latest_delivery_time - current_time)
        
        return service_times, original_deadlines
    
    def _build_distance_matrix_for_vrp(self, locations: List[int]) -> List[List[float]]:
        """
        为 VRP 构建距离矩阵
        
        【优化】使用缓存减少重复计算
        
        Args:
            locations: 位置节点列表
        
        Returns:
            距离矩阵（米）
        """
        n = len(locations)
        distance_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0.0
                elif locations[i] == 0 or locations[j] == 0:
                    # depot 到任何位置距离为0
                    distance_matrix[i][j] = 0.0
                else:
                    from_node = locations[i]
                    to_node = locations[j]
                    cache_key = (from_node, to_node)
                    
                    # 检查缓存
                    if self.enable_distance_cache and cache_key in self.distance_cache:
                        distance_matrix[i][j] = self.distance_cache[cache_key]
                        self.cache_hits += 1
                    else:
                        try:
                            dist = self.env.get_distance(from_node, to_node)
                            distance_matrix[i][j] = float(dist)
                            # 存入缓存
                            if self.enable_distance_cache:
                                self.distance_cache[cache_key] = float(dist)
                            self.cache_misses += 1
                        except Exception as e:
                            logger.warning(f"无法获取距离 {from_node} -> {to_node}: {e}")
                            distance_matrix[i][j] = 999999.0
        
        return distance_matrix
    
    def _build_time_matrix_for_vrp(self, locations: List[int]) -> List[List[float]]:
        """
        为 VRP 构建时间矩阵
        
        【优化】使用缓存减少重复计算
        
        Args:
            locations: 位置节点列表
        
        Returns:
            时间矩阵（秒）
        """
        n = len(locations)
        time_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    time_matrix[i][j] = 0.0
                elif locations[i] == 0 or locations[j] == 0:
                    time_matrix[i][j] = 0.0
                else:
                    from_node = locations[i]
                    to_node = locations[j]
                    cache_key = (from_node, to_node)
                    
                    # 检查缓存
                    if self.enable_distance_cache and cache_key in self.time_cache:
                        time_matrix[i][j] = self.time_cache[cache_key]
                        self.cache_hits += 1
                    else:
                        try:
                            travel_time = self.env.get_travel_time(
                                from_node, 
                                to_node,
                                speed_kph=None  # 使用预计算的时间
                            )
                            time_matrix[i][j] = float(travel_time)
                            # 存入缓存
                            if self.enable_distance_cache:
                                self.time_cache[cache_key] = float(travel_time)
                            self.cache_misses += 1
                        except Exception as e:
                            logger.warning(f"无法获取时间 {from_node} -> {to_node}: {e}")
                            time_matrix[i][j] = 999999.0
        
        return time_matrix
    
    def clear_cache(self) -> None:
        """清空距离/时间缓存"""
        self.distance_cache.clear()
        self.time_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.debug("距离/时间缓存已清空")
    
    def _build_time_windows(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]]
    ) -> List[Tuple[float, float]]:
        """
        构建时间窗列表
        
        Args:
            locations: 位置节点列表
            order_location_map: 订单位置映射
        
        Returns:
            时间窗列表 [(earliest, latest), ...]
        """
        current_time = self.env.env.now
        time_windows = [(0.0, 999999.0)] * len(locations)  # 默认宽松时间窗
        
        # 为每个订单的取送点设置时间窗
        for order_id, loc_map in order_location_map.items():
            order = self.env.orders[order_id]
            
            # 取货点时间窗
            pickup_idx = loc_map['pickup']
            earliest_pickup = max(order.earliest_pickup_time - current_time, 0.0)
            latest_pickup = order.latest_delivery_time - current_time - 300.0  # 留出配送时间
            
            if self.soft_time_windows:
                latest_pickup += self.time_window_slack
            
            # 确保时间窗有效（earliest <= latest）
            latest_pickup = max(latest_pickup, earliest_pickup + 60.0)  # 至少1分钟窗口
            
            time_windows[pickup_idx] = (earliest_pickup, latest_pickup)
            
            # 送货点时间窗
            delivery_idx = loc_map['delivery']
            earliest_delivery = earliest_pickup + 120.0  # 至少取货后2分钟
            latest_delivery = order.latest_delivery_time - current_time
            
            if self.soft_time_windows:
                latest_delivery += self.time_window_slack
            
            # 确保时间窗有效
            latest_delivery = max(latest_delivery, earliest_delivery + 60.0)
            
            time_windows[delivery_idx] = (earliest_delivery, latest_delivery)
        
        return time_windows
    
    def _build_pickups_deliveries(
        self,
        order_ids: List[int],
        order_location_map: Dict[int, Dict[str, int]]
    ) -> List[Tuple[int, int]]:
        """
        构建取送对列表
        
        Args:
            order_ids: 订单ID列表
            order_location_map: 订单位置映射
        
        Returns:
            取送对列表 [(pickup_idx, delivery_idx), ...]
        """
        pickups_deliveries = []
        
        for order_id in order_ids:
            loc_map = order_location_map[order_id]
            pickups_deliveries.append((loc_map['pickup'], loc_map['delivery']))
        
        return pickups_deliveries
    
    def _build_demands(
        self,
        locations: List[int],
        order_location_map: Dict[int, Dict[str, int]]
    ) -> List[int]:
        """
        构建需求向量（用于容量约束）
        
        Args:
            locations: 位置节点列表
            order_location_map: 订单位置映射
        
        Returns:
            需求列表 [0, 1, -1, ...]，pickup=+1, delivery=-1
        """
        demands = [0] * len(locations)
        
        for order_id, loc_map in order_location_map.items():
            demands[loc_map['pickup']] = 1   # 取货增加1单位
            demands[loc_map['delivery']] = -1  # 送货减少1单位
        
        return demands
    
    def _create_routing_model(self, vrp_data: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
        """
        创建 OR-Tools 路由模型
        
        Args:
            vrp_data: VRP 数据模型
        
        Returns:
            (manager, routing) 元组，失败则返回 (None, None)
        """
        try:
            num_locations = len(vrp_data['locations'])
            num_vehicles = vrp_data['num_vehicles']
            depot = vrp_data.get('depot', 0)
            
            # 创建路由索引管理器
            # 注意：使用depot作为所有骑手的统一起始/结束位置
            # 骑手实际位置的影响通过距离/时间矩阵中depot到各位置的距离来体现
            manager = pywrapcp.RoutingIndexManager(
                num_locations,
                num_vehicles,
                depot
            )
            
            # 创建路由模型
            routing = pywrapcp.RoutingModel(manager)
            
            logger.debug(f"路由模型创建成功 - 位置: {num_locations}, 车辆: {num_vehicles}, depot: {depot}")
            return manager, routing
            
        except Exception as e:
            logger.error(f"创建路由模型失败: {str(e)}")
            return None, None
    
    def _solve_vrp(self, vrp_data: Dict[str, Any]) -> Optional[Any]:
        """
        使用 OR-Tools 求解 VRP
        
        Args:
            vrp_data: VRP 数据模型
        
        Returns:
            solution 对象，如果求解失败则返回 None
        """
        import time
        solve_start_time = time.time()
        
        logger.debug("开始创建 OR-Tools 路由模型...")
        
        # 1. 创建路由索引管理器和路由模型
        manager, routing = self._create_routing_model(vrp_data)
        
        if manager is None or routing is None:
            return None
        
        # 2. 注册距离回调
        def distance_callback(from_index, to_index):
            """距离回调函数"""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(vrp_data['distance_matrix'][from_node][to_node])
        
        distance_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
        
        # 3. 注册时间回调
        # 【修复】添加服务时间：Time(A->B) = Service(A) + Travel(A->B)
        # 服务时间包括：商家出餐/取货等待、联系顾客/上楼配送
        service_times = vrp_data.get('service_times', [0] * len(vrp_data['locations']))
        
        def time_callback(from_index, to_index):
            """时间回调函数（包含服务时间）"""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            # 基础行驶时间
            travel_time = int(vrp_data['time_matrix'][from_node][to_node])
            
            # 加上 from_node 的服务时间
            # OR-Tools 的 Arc Cost 包含离开节点的时间
            service_time = service_times[from_node] if from_node < len(service_times) else 0
            
            return travel_time + service_time
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # 4. 添加时间维度约束
        time_dimension_name = 'Time'
        routing.AddDimension(
            time_callback_index,
            999999,  # slack（等待时间上限）
            999999,  # maximum time per vehicle
            False,   # Don't force start cumul to zero
            time_dimension_name
        )
        time_dimension = routing.GetDimensionOrDie(time_dimension_name)
        
        # 【修复】设置时间窗约束 - 使用真正的软时间窗
        # 软时间窗允许超时，但会产生惩罚成本
        soft_penalty = self.config.get('soft_time_window_penalty', 1000)  # 超时惩罚系数
        
        # 【修复】获取原始截止时间映射，用于正确设置软时间窗惩罚
        # 软时间窗惩罚应该基于"原始承诺时间"，而不是"被迫延后的时间"
        original_deadlines = vrp_data.get('original_deadlines', {})
        
        for location_idx, time_window in enumerate(vrp_data['time_windows']):
            if location_idx == 0:  # skip depot
                continue
            index = manager.NodeToIndex(location_idx)
            
            earliest = int(time_window[0])
            latest = int(time_window[1])  # 物理可行的最晚时间（可能被延后）
            
            if self.soft_time_windows:
                # 【修复】真正的软时间窗：
                # 硬边界设置为物理可行的时间范围
                max_horizon = 999999
                time_dimension.CumulVar(index).SetRange(earliest, max_horizon)
                
                # 【关键修复】软上界使用原始截止时间，而非延后后的时间
                # 这样即使订单已经超时，惩罚也会正确计算
                if location_idx in original_deadlines:
                    original_deadline = original_deadlines[location_idx]
                    # 即使原始截止时间已过（负值），也设置为软约束
                    # 这样超时的订单会有更高的惩罚，促使调度器优先处理
                    soft_deadline = max(0, original_deadline)  # 至少为0
                    time_dimension.SetCumulVarSoftUpperBound(index, soft_deadline, soft_penalty)
                else:
                    # 非送货点（如取货点）使用延后后的时间
                    time_dimension.SetCumulVarSoftUpperBound(index, latest, soft_penalty)
            else:
                # 硬时间窗
                time_dimension.CumulVar(index).SetRange(earliest, latest)
        
        # 5. 添加取送约束
        for pickup_idx, delivery_idx in vrp_data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(pickup_idx)
            delivery_index = manager.NodeToIndex(delivery_idx)
            
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            
            # 同一车辆必须完成取送
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            
            # 取货必须在送货之前
            routing.solver().Add(
                time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
            )
        
        # 【修复】6. 添加订单锁定约束
        on_board_orders = vrp_data.get('on_board_orders', {})
        order_location_map = vrp_data.get('order_location_map', {})
        assigned_orders_to_replan = vrp_data.get('assigned_orders_to_replan', {})
        
        # 6.1 在途订单（PICKED_UP）：只锁定送货点到指定骑手
        # 不添加 PickupAndDelivery 约束，因为已经在车上了
        for order_id, onboard_info in on_board_orders.items():
            if order_id not in order_location_map:
                continue
            
            locked_vehicle = onboard_info['courier_idx']
            loc_map = order_location_map[order_id]
            
            # 【修复】只锁定送货点，不添加取送对约束
            delivery_idx = loc_map.get('delivery', -1)
            if delivery_idx >= 0:
                delivery_index = manager.NodeToIndex(delivery_idx)
                routing.solver().Add(
                    routing.VehicleVar(delivery_index) == locked_vehicle
                )
                logger.debug(
                    f"在途订单 {order_id}（PICKED_UP）送货点锁定到骑手 {locked_vehicle}"
                )
        
        # 6.2 【新增】ASSIGNED订单：锁定取货点和送货点到指定骑手
        # 这些订单已分配但未取货，需要锁定给当前骑手
        for order_id, locked_vehicle in assigned_orders_to_replan.items():
            if order_id not in order_location_map:
                continue
            
            loc_map = order_location_map[order_id]
            pickup_idx = loc_map.get('pickup', -1)
            delivery_idx = loc_map.get('delivery', -1)
            
            # 锁定取货点
            if pickup_idx >= 0:
                pickup_index = manager.NodeToIndex(pickup_idx)
                routing.solver().Add(
                    routing.VehicleVar(pickup_index) == locked_vehicle
                )
            
            # 锁定送货点（通过取送对约束已经隐式锁定，但显式添加更安全）
            if delivery_idx >= 0:
                delivery_index = manager.NodeToIndex(delivery_idx)
                routing.solver().Add(
                    routing.VehicleVar(delivery_index) == locked_vehicle
                )
            
            logger.debug(
                f"已分配订单 {order_id}（ASSIGNED）锁定到骑手 {locked_vehicle}"
            )
        
        # 7. 添加容量约束
        def demand_callback(from_index):
            """需求回调函数"""
            from_node = manager.IndexToNode(from_index)
            return vrp_data['demands'][from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            vrp_data['vehicle_capacities'],  # 使用最大容量
            False,  # 【关键修复】设为 False，允许非零起始值
            'Capacity'
        )
        
        # 【关键修复】手动设置每个车辆的起始负载
        # 解决"负负载崩溃"问题：在途订单的送货 Demand=-1，如果从0开始会变成-1，违反非负约束
        capacity_dimension = routing.GetDimensionOrDie('Capacity')
        initial_loads = vrp_data.get('courier_initial_loads', [0] * vrp_data['num_vehicles'])
        vehicle_capacities = vrp_data['vehicle_capacities']
        
        for vehicle_id in range(vrp_data['num_vehicles']):
            start_index = routing.Start(vehicle_id)
            raw_load = int(initial_loads[vehicle_id])
            max_cap = vehicle_capacities[vehicle_id]
            
            # 【防御性修复】防止因数据异常导致的 Infeasible
            # 如果初始负载超过容量上限，强制截断以避免求解器崩溃
            if raw_load > max_cap:
                logger.error(
                    f"检测到骑手 {vehicle_id} 初始负载({raw_load}) > 容量({max_cap})，"
                    f"强制截断以避免求解器崩溃"
                )
                safe_load = max_cap
            else:
                safe_load = raw_load
            
            # 锁定起始节点的累积变量为初始负载
            # 这样在途订单送货时：safe_load + (-1) = safe_load - 1 >= 0
            capacity_dimension.CumulVar(start_index).SetRange(safe_load, safe_load)
            
            if safe_load > 0:
                logger.debug(f"车辆 {vehicle_id} 起始负载设置为 {safe_load}")
        
        # 8. 设置搜索参数
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.first_solution_strategy
        search_parameters.local_search_metaheuristic = self.local_search_metaheuristic
        search_parameters.time_limit.seconds = self.time_limit_seconds
        search_parameters.log_search = False  # 关闭搜索日志
        
        # 8. 求解
        logger.info(f"开始求解 VRP（时间限制: {self.time_limit_seconds}秒）...")
        solution = routing.SolveWithParameters(search_parameters)
        
        solve_time = time.time() - solve_start_time
        self.total_solve_time += solve_time
        
        if solution:
            status = routing.status()
            status_names = {
                0: 'ROUTING_NOT_SOLVED',
                1: 'ROUTING_SUCCESS',
                2: 'ROUTING_FAIL',
                3: 'ROUTING_FAIL_TIMEOUT',
                4: 'ROUTING_INVALID'
            }
            logger.info(
                f"求解完成 - 状态: {status_names.get(status, 'UNKNOWN')}, "
                f"耗时: {solve_time:.2f}秒, "
                f"目标值: {solution.ObjectiveValue():.0f}"
            )
            
            # 保存求解结果供后续使用
            return {
                'solution': solution,
                'manager': manager,
                'routing': routing,
                'solve_time': solve_time,
                'objective_value': solution.ObjectiveValue()
            }
        else:
            logger.warning(f"求解失败，耗时: {solve_time:.2f}秒")
            return None
    
    def _apply_solution(
        self,
        vrp_data: Dict[str, Any],
        solution_dict: Dict[str, Any]
    ) -> int:
        """
        应用求解结果到仿真环境
        
        【修复】路线更新逻辑：
        1. 用VRP计算的新路线完全替换旧路线
        2. 保留骑手正在执行的第一个任务（如果有）
        3. 正确处理在途订单（PICKED_UP）：只添加送货任务，不重新分配
        4. 正确处理已分配订单（ASSIGNED）：不重复分配，只更新路线
        
        Args:
            vrp_data: VRP 数据模型
            solution_dict: OR-Tools 解字典
        
        Returns:
            成功分配的订单数
        """
        from ..entities import OrderStatus
        
        logger.debug("解析并应用 VRP 解...")
        
        solution = solution_dict['solution']
        manager = solution_dict['manager']
        routing = solution_dict['routing']
        
        available_couriers = vrp_data['available_couriers']
        locations = vrp_data['locations']
        order_location_map = vrp_data['order_location_map']
        on_board_orders = vrp_data.get('on_board_orders', {})
        assigned_orders_to_replan = vrp_data.get('assigned_orders_to_replan', {})
        
        assigned_orders = set()
        
        # 遍历每个车辆的路线
        for vehicle_id in range(vrp_data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = []
            
            logger.debug(f"提取车辆 {vehicle_id} 的路线...")
            
            # 遍历路线
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                
                # 跳过 depot
                if node_index != 0:
                    route.append(node_index)
                
                index = solution.Value(routing.NextVar(index))
            
            # 如果路线为空，跳过
            if not route:
                logger.debug(f"车辆 {vehicle_id} 无任务")
                continue
            
            # 将路线转换为订单任务
            courier = available_couriers[vehicle_id]
            tasks = self._route_to_tasks_with_onboard(
                route, order_location_map, locations, on_board_orders
            )
            
            if not tasks:
                continue
            
            # 【修复】更安全的路线拼接逻辑
            # 保留骑手正在执行的第一个任务（如果骑手正在移动中）
            # 但要防止死锁：满载时不能保留取货任务
            old_route = courier.current_route.copy()
            current_task = None
            
            if old_route and courier.status not in [
                self._get_idle_status()
            ]:
                potential_task = old_route[0]
                action_type, _, _ = potential_task
                
                # 【死锁预防检查】
                # 如果是取货任务，且骑手已满载，则绝不能保留该任务！
                # 必须让位给 VRP 计算出的（必然是送货的）新任务
                is_full = len(courier.assigned_orders) >= courier.max_capacity
                
                if action_type == 'pickup' and is_full:
                    logger.warning(
                        f"骑手 {courier.courier_id} 满载({len(courier.assigned_orders)}/{courier.max_capacity})"
                        f"但正在执行取货任务，强制丢弃旧任务以防死锁，执行 VRP 新方案"
                    )
                    current_task = None
                else:
                    current_task = potential_task
                    logger.debug(
                        f"骑手 {courier.courier_id} 正在执行任务 {current_task}，保留该任务"
                    )
            
            # 清空旧路线
            courier.current_route.clear()
            
            # 如果有正在执行的任务，先添加回去
            if current_task:
                courier.current_route.append(current_task)
            
            # 添加VRP计算的新路线
            new_order_count = 0
            for action, order_id, node_id in tasks:
                order = self.env.orders[order_id]
                is_onboard = order_id in on_board_orders
                
                # 跳过正在执行的任务（避免重复）
                if current_task and (action, order_id, node_id) == current_task:
                    continue
                
                # 在途订单：只添加送货任务到路线，不重新分配
                if is_onboard:
                    if action == 'delivery':
                        # 检查是否已在路线中
                        if not any(t[0] == 'delivery' and t[1] == order_id 
                                   for t in courier.current_route):
                            courier.current_route.append((action, order_id, node_id))
                    # 跳过在途订单的虚拟取货
                    continue
                
                # 【修复】检查订单是否是已分配待重规划的订单（ASSIGNED状态）
                is_assigned_replan = order_id in assigned_orders_to_replan
                
                # 待分配订单：正常处理
                if action == 'pickup' and order_id not in assigned_orders:
                    # 【修复】ASSIGNED状态订单：不重复分配，只更新路线
                    if is_assigned_replan:
                        # 订单已经分配给该骑手，只需添加到路线
                        logger.debug(
                            f"订单 {order_id}（ASSIGNED）已分配给骑手 {courier.courier_id}，更新路线"
                        )
                        assigned_orders.add(order_id)
                        # 不增加 new_order_count，因为不是新分配
                    elif order.status == OrderStatus.PENDING:
                        # PENDING状态订单：正常分配流程
                        # 【修复】移除容量检查
                        # OR-Tools 的 Capacity Dimension 已经保证了路线的物理可行性
                        # 它允许"先送后取"的智能方案：骑手满载时先送货腾出空间，再取新货
                        # 如果在这里检查 len(assigned_orders) >= max_capacity，
                        # 会错误拦截这种合法的"先送后取"方案
                        
                        # 更新订单状态
                        order.assign_to_courier(courier.courier_id, self.env.env.now)
                        
                        # 更新骑手状态
                        courier.assign_order(order_id)
                        
                        # 从待分配队列移除
                        if order_id in self.env.pending_orders:
                            self.env.pending_orders.remove(order_id)
                            self.env.assigned_orders.append(order_id)
                        
                        # 记录事件
                        self.env.record_event(
                            'order_assigned',
                            order_id,
                            {
                                'courier_id': courier.courier_id,
                                'dispatcher': 'OR-Tools',
                                'assignment_time': self.env.env.now,
                                'route_mode': 'replace'  # 标记为替换模式
                            }
                        )
                        
                        assigned_orders.add(order_id)
                        new_order_count += 1
                    else:
                        # 其他状态（如PICKING_UP等）：跳过
                        logger.debug(
                            f"订单 {order_id} 状态为 {order.status}，跳过分配"
                        )
                        continue
                
                # 添加任务到骑手路线
                courier.current_route.append((action, order_id, node_id))
            
            logger.info(
                f"[{self.env.env.now:.1f}s] 骑手 {courier.courier_id} "
                f"路线更新: {len(old_route)} -> {len(courier.current_route)} 个任务, "
                f"新分配 {new_order_count} 个订单"
            )
        
        logger.info(f"成功分配 {len(assigned_orders)} 个订单到 {vrp_data['num_vehicles']} 个骑手")
        
        return len(assigned_orders)
    
    def _get_idle_status(self):
        """获取IDLE状态枚举值"""
        from ..entities import CourierStatus
        return CourierStatus.IDLE
    
    def _route_to_tasks_with_onboard(
        self,
        route: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        locations: List[int],
        on_board_orders: Dict[int, Dict[str, Any]]
    ) -> List[Tuple[str, int, int]]:
        """
        将路线节点索引转换为任务列表（包含在途订单处理）
        
        【修复】在途订单只有送货点（pickup = -1），不添加取货任务
        
        Args:
            route: 路线节点索引列表 [location_idx, ...]
            order_location_map: 订单位置映射
            locations: 位置节点列表
            on_board_orders: 在途订单信息
        
        Returns:
            任务列表 [(action, order_id, node_id), ...]
        """
        tasks = []
        
        # 反向映射：location_idx -> (order_id, action)
        idx_to_order_action = {}
        for order_id, loc_map in order_location_map.items():
            is_onboard = loc_map.get('is_onboard', False)
            pickup_idx = loc_map.get('pickup', -1)
            delivery_idx = loc_map.get('delivery', -1)
            
            if is_onboard:
                # 【修复】在途订单：只映射送货点（pickup = -1，无取货点）
                if delivery_idx >= 0:
                    idx_to_order_action[delivery_idx] = (order_id, 'delivery')
            else:
                # 待分配订单（PENDING和ASSIGNED）：正常映射取货和送货
                if pickup_idx >= 0:
                    idx_to_order_action[pickup_idx] = (order_id, 'pickup')
                if delivery_idx >= 0:
                    idx_to_order_action[delivery_idx] = (order_id, 'delivery')
        
        # 转换路线
        for location_idx in route:
            if location_idx in idx_to_order_action:
                order_id, action = idx_to_order_action[location_idx]
                node_id = locations[location_idx]
                tasks.append((action, order_id, node_id))
        
        return tasks
    
    def _route_to_tasks(
        self,
        route: List[int],
        order_location_map: Dict[int, Dict[str, int]],
        locations: List[int]
    ) -> List[Tuple[str, int, int]]:
        """
        将路线节点索引转换为任务列表
        
        Args:
            route: 路线节点索引列表 [location_idx, ...]
            order_location_map: 订单位置映射 {order_id: {'pickup': idx, 'delivery': idx}}
            locations: 位置节点列表
        
        Returns:
            任务列表 [(action, order_id, node_id), ...]
        """
        tasks = []
        
        # 反向映射：location_idx -> (order_id, action)
        idx_to_order_action = {}
        for order_id, loc_map in order_location_map.items():
            idx_to_order_action[loc_map['pickup']] = (order_id, 'pickup')
            idx_to_order_action[loc_map['delivery']] = (order_id, 'delivery')
        
        # 转换路线
        for location_idx in route:
            if location_idx in idx_to_order_action:
                order_id, action = idx_to_order_action[location_idx]
                node_id = locations[location_idx]
                tasks.append((action, order_id, node_id))
        
        return tasks
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取调度器统计信息
        
        Returns:
            统计字典
        """
        stats = {
            'dispatcher_type': 'OR-Tools',
            'dispatch_count': self.dispatch_count,
            'solve_success_count': self.solve_success_count,
            'solve_failure_count': self.solve_failure_count,
            'average_solve_time': (
                self.total_solve_time / self.solve_success_count
                if self.solve_success_count > 0 else 0.0
            ),
            # 优化策略统计
            'batching_enabled': self.enable_batching,
            'batch_count': self.batch_count,
            'batch_strategy': self.batch_strategy if self.enable_batching else 'N/A',
            'time_window_slack': self.time_window_slack,
            'soft_time_windows': self.soft_time_windows,
            # 缓存统计
            'cache_enabled': self.enable_distance_cache,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            'distance_cache_size': len(self.distance_cache),
            'time_cache_size': len(self.time_cache)
        }
        
        # 计算成功率
        total_attempts = self.solve_success_count + self.solve_failure_count
        if total_attempts > 0:
            stats['solve_success_rate'] = self.solve_success_count / total_attempts
        else:
            stats['solve_success_rate'] = 0.0
        
        return stats

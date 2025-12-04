"""
Greedy Dispatcher - Day 3 实现
贪心调度策略：最近空闲骑手 + 最小增加距离插入
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GreedyDispatcher:
    """
    贪心调度器
    策略：
    1. 找最近的空闲骑手（基于当前位置到商家距离）
    2. 简单FIFO路径（先取货后送货）
    """
    
    def __init__(self, env):
        """
        初始化调度器
        
        Args:
            env: SimulationEnvironment实例
        """
        self.env = env
        self.dispatch_count = 0
    
    def dispatch_pending_orders(self) -> int:
        """
        调度所有待分配订单
        
        Returns:
            成功分配的订单数
        """
        dispatched_count = 0
        pending_copy = self.env.pending_orders.copy()  # 避免迭代时修改
        
        for order_id in pending_copy:
            if self._dispatch_single_order(order_id):
                dispatched_count += 1
        
        if dispatched_count > 0:
            logger.info(
                f"[{self.env.env.now:.1f}s] Greedy调度完成，"
                f"分配 {dispatched_count}/{len(pending_copy)} 个订单"
            )
        
        return dispatched_count
    
    def _dispatch_single_order(self, order_id: int) -> bool:
        """
        调度单个订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            是否成功分配
        """
        order = self.env.orders[order_id]
        
        # 找所有空闲骑手
        available_couriers = [
            c for c in self.env.couriers.values() 
            if c.is_available()
        ]
        
        if not available_couriers:
            logger.debug(
                f"[{self.env.env.now:.1f}s] 无可用骑手，"
                f"订单 {order_id} 继续等待"
            )
            return False
        
        # 选择最近的骑手（基于当前位置到商家的距离）
        nearest_courier = self._find_nearest_courier(
            available_couriers, 
            order.merchant_node
        )
        
        # 如果找不到可达的骑手（节点不在距离矩阵中），从队列移除该订单
        if nearest_courier is None:
            logger.warning(
                f"[{self.env.env.now:.1f}s] 订单 {order_id} "
                f"的商家节点 {order.merchant_node} 不在路网采样中，"
                f"从待分配队列移除"
            )
            # 从待分配队列中移除，避免阻塞其他订单
            if order_id in self.env.pending_orders:
                self.env.pending_orders.remove(order_id)
            # 标记订单为取消状态
            from ..entities import OrderStatus
            order.status = OrderStatus.CANCELLED
            return False
        
        # 分配订单
        self._assign_order_to_courier(order, nearest_courier)
        
        return True
    
    def _find_nearest_courier(self, couriers: List, target_node: int):
        """
        找到距离目标节点最近的骑手
        
        Args:
            couriers: 骑手列表
            target_node: 目标节点ID
        
        Returns:
            最近的骑手对象，如果所有骑手都无法到达则返回None
        """
        min_distance = float('inf')
        nearest_courier = None
        
        for courier in couriers:
            try:
                distance = self.env.get_distance(
                    courier.current_node, 
                    target_node
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_courier = courier
            except Exception as e:
                # 节点不在映射中，跳过这个骑手
                continue
        
        if nearest_courier is None:
            logger.warning(
                f"无法为目标节点 {target_node} 找到可达的骑手 "
                f"(可能节点不在距离矩阵中)"
            )
        
        return nearest_courier
    
    def _assign_order_to_courier(self, order, courier) -> None:
        """
        将订单分配给骑手
        
        Args:
            order: Order对象
            courier: Courier对象
        """
        # 更新订单状态
        order.assign_to_courier(courier.courier_id, self.env.env.now)
        
        # 更新骑手状态
        courier.assign_order(order.order_id)
        
        # 简单路径：先取货，后送货（FIFO）
        courier.current_route.append(('pickup', order.order_id, order.merchant_node))
        courier.current_route.append(('delivery', order.order_id, order.customer_node))
        
        # 从待分配队列移除
        if order.order_id in self.env.pending_orders:
            self.env.pending_orders.remove(order.order_id)
            self.env.assigned_orders.append(order.order_id)
        
        # 记录事件
        self.env.record_event(
            'order_assigned',
            order.order_id,
            {
                'courier_id': courier.courier_id,
                'merchant_node': order.merchant_node,
                'customer_node': order.customer_node,
                'assignment_time': self.env.env.now
            }
        )
        
        self.dispatch_count += 1
        
        logger.info(
            f"[{self.env.env.now:.1f}s] 订单 {order.order_id} → "
            f"骑手 {courier.courier_id} "
            f"(距离: {self.env.get_distance(courier.current_node, order.merchant_node):.0f}m)"
        )

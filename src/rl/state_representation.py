"""
状态表示模块
将仿真环境的状态编码为RL Agent可用的向量形式

状态空间设计（参考研究大纲）：
- 时间特征：当前时间、高峰/平峰指示
- 订单特征：待分配订单的位置、剩余时间窗、紧迫度
- 骑手特征：位置、当前负载、剩余容量、工作时长
- 全局特征：系统负载（订单/骑手比）、热力图
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class StateEncoder:
    """
    状态编码器
    将复杂的仿真状态转换为固定维度的向量
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化状态编码器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 状态维度配置
        self.max_pending_orders = self.config.get('max_pending_orders', 50)
        self.max_couriers = self.config.get('max_couriers', 50)
        self.grid_size = self.config.get('grid_size', 10)  # 空间网格划分
        
        # Day 22: 商家状态特征配置
        self.max_merchants = self.config.get('max_merchants', 100)
        self.include_merchant_features = self.config.get('include_merchant_features', True)
        
        # 计算状态空间维度
        self.state_dim = self._calculate_state_dimension()
        
        logger.info(f"StateEncoder初始化完成，状态维度: {self.state_dim}")
        if self.include_merchant_features:
            logger.info(f"  商家特征已启用，max_merchants: {self.max_merchants}")
    
    def _calculate_state_dimension(self) -> int:
        """
        计算状态空间总维度
        
        Returns:
            状态向量维度
        """
        dim = 0
        
        # 1. 时间特征 (3维)
        dim += 3  # [当前时间归一化, sin(time_of_day), cos(time_of_day)]
        
        # 2. 全局特征 (5维 + 2维商家特征 = 7维)
        dim += 7  # [待分配订单数, 空闲骑手数, 忙碌骑手数, 订单/骑手比, 平均超时风险, 平均商家利用率, 平均等餐时间]
        
        # 3. 订单特征 (max_pending_orders * 9)
        # 每个订单: [x坐标, y坐标, 剩余时间窗归一化, 准备完成时间, 距离最近骑手距离归一化]
        # Day 22新增: [商家队列位置, 商家利用率, 预估出餐时间, 餐品是否就绪]
        dim += self.max_pending_orders * 9
        
        # 4. 骑手特征 (max_couriers * 4)
        # 每个骑手: [x坐标, y坐标, 当前负载/最大容量, 空闲时长归一化]
        dim += self.max_couriers * 4
        
        # 5. 空间热力图 (grid_size * grid_size)
        # 网格化订单密度
        dim += self.grid_size * self.grid_size
        
        # 6. Day 22新增: 商家繁忙度热力图 (grid_size * grid_size)
        # 消融实验：总是包含这个维度，即使关闭也用0填充
        # 这确保与训练好的模型兼容（固定860维）
        dim += self.grid_size * self.grid_size
        
        return dim
    
    def encode(self, env_state: Dict[str, Any]) -> np.ndarray:
        """
        编码环境状态
        
        Args:
            env_state: 仿真环境状态字典，包含:
                - current_time: 当前仿真时间
                - pending_orders: 待分配订单列表
                - couriers: 骑手字典
                - graph: 路网图
                - bounds: 地理边界 (min_x, max_x, min_y, max_y)
                - merchants: 商家字典 (Day 22新增，可选)
        
        Returns:
            状态向量 (numpy array)
        """
        state_vector = []
        
        # 获取商家字典（如果存在）
        merchants = env_state.get('merchants', {})
        
        # 1. 编码时间特征
        time_features = self._encode_time_features(env_state['current_time'])
        state_vector.extend(time_features)
        
        # 2. 编码全局特征（包含商家特征）
        global_features = self._encode_global_features(env_state, merchants)
        state_vector.extend(global_features)
        
        # 3. 编码订单特征（包含商家备餐状态）
        order_features = self._encode_order_features(
            env_state['pending_orders'],
            env_state['couriers'],
            env_state['bounds'],
            env_state['current_time'],
            merchants
        )
        state_vector.extend(order_features)
        
        # 4. 编码骑手特征
        courier_features = self._encode_courier_features(
            env_state['couriers'],
            env_state['bounds'],
            env_state['current_time']
        )
        state_vector.extend(courier_features)
        
        # 5. 编码空间热力图
        heatmap_features = self._encode_spatial_heatmap(
            env_state['pending_orders'],
            env_state['bounds']
        )
        state_vector.extend(heatmap_features)
        
        # 6. Day 22: 编码商家繁忙度热力图
        # 消融实验支持：即使关闭商家特征，仍输出完整维度（用0填充）
        # 这确保与训练好的模型兼容（模型期望860维输入）
        if self.include_merchant_features:
            merchant_heatmap = self._encode_merchant_heatmap(merchants, env_state['bounds'])
            state_vector.extend(merchant_heatmap)
        else:
            # 消融模式：用0填充商家热力图维度
            zero_heatmap = [0.0] * (self.grid_size * self.grid_size)
            state_vector.extend(zero_heatmap)
            # 首次调用时输出INFO级别日志，便于确认消融生效
            if not hasattr(self, '_ablation_logged'):
                logger.info("【消融模式】商家特征已关闭：全局特征、订单特征、热力图均使用零填充")
                self._ablation_logged = True
        
        return np.array(state_vector, dtype=np.float32)
    
    def _encode_time_features(self, current_time: float) -> List[float]:
        """
        编码时间特征
        
        Args:
            current_time: 当前仿真时间（秒）
        
        Returns:
            时间特征向量 [归一化时间, sin(时刻), cos(时刻)]
        """
        # 归一化到 [0, 1]（假设24小时仿真）
        time_normalized = (current_time % 86400) / 86400
        
        # 周期性编码（捕捉早晚高峰）
        time_radians = 2 * np.pi * time_normalized
        time_sin = np.sin(time_radians)
        time_cos = np.cos(time_radians)
        
        return [time_normalized, time_sin, time_cos]
    
    def _encode_global_features(self, env_state: Dict[str, Any], 
                                 merchants: Dict[int, Any] = None) -> List[float]:
        """
        编码全局系统特征（Day 22: 扩展商家特征）
        
        Returns:
            全局特征向量 (7维)
        """
        pending_orders = env_state['pending_orders']
        couriers = env_state['couriers']
        current_time = env_state['current_time']
        merchants = merchants or {}
        
        # 统计骑手状态
        idle_couriers = sum(1 for c in couriers.values() if len(c.current_route) == 0)
        busy_couriers = len(couriers) - idle_couriers
        
        # 订单/骑手比（归一化）
        order_courier_ratio = len(pending_orders) / max(len(couriers), 1)
        order_courier_ratio_normalized = min(order_courier_ratio / 10.0, 1.0)
        
        # 平均超时风险（0-1）
        if pending_orders:
            timeout_risks = [
                self._calculate_timeout_risk(order, current_time)
                for order in pending_orders
            ]
            avg_timeout_risk = np.mean(timeout_risks)
        else:
            avg_timeout_risk = 0.0
        
        # Day 22: 商家全局特征
        # 修复：检查 include_merchant_features 开关，确保消融实验正确屏蔽商家特征
        if self.include_merchant_features and merchants:
            # 平均商家利用率
            utilizations = [m.get_utilization() if hasattr(m, 'get_utilization') else 0.0 
                           for m in merchants.values()]
            avg_merchant_utilization = np.mean(utilizations) if utilizations else 0.0
            
            # 平均预估等餐时间（归一化到10分钟）
            wait_times = [m.estimate_wait_time() if hasattr(m, 'estimate_wait_time') else 0.0 
                         for m in merchants.values()]
            avg_wait_time = np.mean(wait_times) if wait_times else 0.0
            avg_wait_time_normalized = min(avg_wait_time / 600.0, 1.0)
        else:
            # 消融模式或无商家数据：使用默认值
            avg_merchant_utilization = 0.0
            avg_wait_time_normalized = 0.0
        
        return [
            float(len(pending_orders)) / self.max_pending_orders,  # 归一化订单数
            float(idle_couriers) / max(len(couriers), 1),
            float(busy_couriers) / max(len(couriers), 1),
            order_courier_ratio_normalized,
            avg_timeout_risk,
            avg_merchant_utilization,  # Day 22新增
            avg_wait_time_normalized   # Day 22新增
        ]
    
    def _encode_order_features(self, 
                               pending_orders: List[Any],
                               couriers: Dict[int, Any],
                               bounds: Tuple[float, float, float, float],
                               current_time: float,
                               merchants: Dict[int, Any] = None) -> List[float]:
        """
        编码待分配订单特征（Day 22: 扩展商家备餐状态）
        
        Args:
            pending_orders: 订单对象列表
            couriers: 骑手字典
            bounds: 地理边界 (min_x, max_x, min_y, max_y)
            current_time: 当前时间
            merchants: 商家字典 (Day 22新增)
        
        Returns:
            订单特征向量（固定长度，每订单9维）
        """
        min_x, max_x, min_y, max_y = bounds
        merchants = merchants or {}
        features = []
        
        # 归一化参数
        MAX_QUEUE = 20
        MAX_PREP_TIME = 600.0  # 10分钟
        
        for i in range(self.max_pending_orders):
            if i < len(pending_orders):
                order = pending_orders[i]
                
                # 归一化坐标（确保转换为float）
                x = float(order.merchant_coords[0])
                y = float(order.merchant_coords[1])
                x_norm = (x - min_x) / max(max_x - min_x, 1.0)
                y_norm = (y - min_y) / max(max_y - min_y, 1.0)
                
                # 剩余时间窗（归一化到[0, 1]）
                time_to_deadline = max(order.latest_delivery_time - current_time, 0)
                time_window_normalized = min(time_to_deadline / 3600.0, 1.0)  # 归一化到1小时
                
                # 准备完成时间
                time_to_ready = max(order.earliest_pickup_time - current_time, 0)
                ready_time_normalized = min(time_to_ready / 1800.0, 1.0)  # 归一化到30分钟
                
                # 距离最近骑手的距离（简化版，取欧氏距离）
                min_distance = self._find_nearest_courier_distance(order, couriers, bounds)
                
                # Day 22: 商家备餐状态特征
                # 修复：检查 include_merchant_features 开关，确保消融实验正确屏蔽订单级商家特征
                merchant_id = getattr(order, 'merchant_id', None)
                if self.include_merchant_features and merchant_id and merchant_id in merchants:
                    merchant = merchants[merchant_id]
                    queue_pos = min(getattr(order, 'merchant_queue_position', 0) / MAX_QUEUE, 1.0)
                    utilization = merchant.get_utilization() if hasattr(merchant, 'get_utilization') else 0.0
                    est_ready = getattr(order, 'estimated_ready_time', None)
                    if est_ready:
                        est_ready_norm = min((est_ready - current_time) / MAX_PREP_TIME, 1.0)
                        est_ready_norm = max(est_ready_norm, 0.0)
                    else:
                        est_ready_norm = 0.5
                    food_ready = 1.0 if getattr(order, 'food_ready', False) else 0.0
                else:
                    # 消融模式或无商家信息：使用默认值（零填充）
                    queue_pos = 0.0
                    utilization = 0.0
                    est_ready_norm = 0.5
                    food_ready = 0.0
                
                features.extend([
                    x_norm, y_norm, time_window_normalized, ready_time_normalized, min_distance,
                    queue_pos, utilization, est_ready_norm, food_ready  # Day 22新增4维
                ])
            else:
                # Padding（用0填充，9维）
                features.extend([0.0] * 9)
        
        return features
    
    def _encode_courier_features(self,
                                 couriers: Dict[int, Any],
                                 bounds: Tuple[float, float, float, float],
                                 current_time: float) -> List[float]:
        """
        编码骑手特征
        
        Returns:
            骑手特征向量（固定长度）
        """
        min_x, max_x, min_y, max_y = bounds
        features = []
        
        courier_list = list(couriers.values())
        
        for i in range(self.max_couriers):
            if i < len(courier_list):
                courier = courier_list[i]
                
                # 归一化坐标（确保转换为float）
                x = float(courier.current_coords[0])
                y = float(courier.current_coords[1])
                x_norm = (x - min_x) / max(max_x - min_x, 1.0)
                y_norm = (y - min_y) / max(max_y - min_y, 1.0)
                
                # 当前负载率
                load_ratio = len(courier.current_route) / max(courier.max_capacity, 1)
                
                # 空闲时长（归一化）
                if len(courier.current_route) == 0:
                    # 简化：假设骑手有last_task_complete_time属性
                    idle_time = current_time - getattr(courier, 'last_task_time', current_time)
                    idle_time_normalized = min(idle_time / 600.0, 1.0)  # 归一化到10分钟
                else:
                    idle_time_normalized = 0.0
                
                features.extend([x_norm, y_norm, load_ratio, idle_time_normalized])
            else:
                # Padding
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features
    
    def _encode_spatial_heatmap(self,
                                pending_orders: List[Any],
                                bounds: Tuple[float, float, float, float]) -> List[float]:
        """
        编码空间热力图（订单密度分布）
        
        Returns:
            热力图向量（grid_size * grid_size）
        """
        min_x, max_x, min_y, max_y = bounds
        
        # 初始化网格
        heatmap = np.zeros((self.grid_size, self.grid_size))
        
        # 统计每个网格的订单数
        for order in pending_orders:
            x, y = order.merchant_coords
            
            # 计算网格索引
            grid_x = int((x - min_x) / max(max_x - min_x, 1.0) * self.grid_size)
            grid_y = int((y - min_y) / max(max_y - min_y, 1.0) * self.grid_size)
            
            # 边界检查
            grid_x = max(0, min(grid_x, self.grid_size - 1))
            grid_y = max(0, min(grid_y, self.grid_size - 1))
            
            heatmap[grid_x, grid_y] += 1
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap.flatten().tolist()
    
    def _encode_merchant_heatmap(self,
                                  merchants: Dict[int, Any],
                                  bounds: Tuple[float, float, float, float]) -> List[float]:
        """
        Day 22: 编码商家繁忙度热力图
        
        Args:
            merchants: 商家字典
            bounds: 地理边界 (min_x, max_x, min_y, max_y)
        
        Returns:
            商家繁忙度热力图向量（grid_size * grid_size）
        """
        min_x, max_x, min_y, max_y = bounds
        
        # 初始化网格
        heatmap = np.zeros((self.grid_size, self.grid_size))
        count_map = np.zeros((self.grid_size, self.grid_size))
        
        if not merchants:
            return heatmap.flatten().tolist()
        
        # 统计每个网格的商家繁忙度
        for merchant in merchants.values():
            # 获取商家坐标
            coords = getattr(merchant, 'coords', None)
            if coords is None:
                continue
            
            x, y = coords
            
            # 计算网格索引
            grid_x = int((x - min_x) / max(max_x - min_x, 1.0) * self.grid_size)
            grid_y = int((y - min_y) / max(max_y - min_y, 1.0) * self.grid_size)
            
            # 边界检查
            grid_x = max(0, min(grid_x, self.grid_size - 1))
            grid_y = max(0, min(grid_y, self.grid_size - 1))
            
            # 累加利用率
            utilization = merchant.get_utilization() if hasattr(merchant, 'get_utilization') else 0.0
            heatmap[grid_x, grid_y] += utilization
            count_map[grid_x, grid_y] += 1
        
        # 计算平均利用率
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.where(count_map > 0, heatmap / count_map, 0)
        
        # 利用率已在0-1范围，无需额外归一化
        return heatmap.flatten().tolist()
    
    def _calculate_timeout_risk(self, order: Any, current_time: float) -> float:
        """
        计算订单超时风险（0-1）
        
        Args:
            order: 订单对象
            current_time: 当前时间
        
        Returns:
            超时风险值
        """
        time_to_deadline = max(order.latest_delivery_time - current_time, 0)
        
        # 风险与剩余时间成反比
        # 剩余时间 < 15分钟: 高风险
        # 剩余时间 > 60分钟: 低风险
        risk = 1.0 - min(time_to_deadline / 3600.0, 1.0)
        
        return risk
    
    def _find_nearest_courier_distance(self,
                                       order: Any,
                                       couriers: Dict[int, Any],
                                       bounds: Tuple[float, float, float, float]) -> float:
        """
        找到距离订单最近的骑手（欧氏距离，归一化）
        
        Returns:
            归一化距离 [0, 1]
        """
        if not couriers:
            return 1.0
        
        # 确保坐标转换为float类型
        order_coords = np.array([float(order.merchant_coords[0]), float(order.merchant_coords[1])])
        min_distance = float('inf')
        
        for courier in couriers.values():
            courier_coords = np.array([float(courier.current_coords[0]), float(courier.current_coords[1])])
            distance = np.linalg.norm(order_coords - courier_coords)
            min_distance = min(min_distance, distance)
        
        # 归一化（假设最大合理距离为地图对角线长度）
        min_x, max_x, min_y, max_y = bounds
        max_distance = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        return min(min_distance / max_distance, 1.0)
    
    def get_state_dimension(self) -> int:
        """
        获取状态空间维度
        
        Returns:
            状态向量维度
        """
        return self.state_dim
    
    def get_state_description(self) -> Dict[str, Any]:
        """
        获取状态空间描述（用于文档和调试）
        Day 22: 更新以反映商家特征扩展
        
        Returns:
            状态描述字典
        """
        desc = {
            'total_dimension': self.state_dim,
            'time_features': {
                'dimension': 3,
                'description': '[normalized_time, sin(time), cos(time)]'
            },
            'global_features': {
                'dimension': 7,
                'description': '[pending_orders_ratio, idle_ratio, busy_ratio, order_courier_ratio, avg_timeout_risk, avg_merchant_utilization, avg_wait_time]'
            },
            'order_features': {
                'dimension': self.max_pending_orders * 9,
                'per_order': '[x, y, time_window, ready_time, nearest_courier_distance, queue_pos, utilization, est_ready_time, food_ready]',
                'note': 'Day 22: 新增4维商家备餐状态特征'
            },
            'courier_features': {
                'dimension': self.max_couriers * 4,
                'per_courier': '[x, y, load_ratio, idle_time]'
            },
            'spatial_heatmap': {
                'dimension': self.grid_size * self.grid_size,
                'description': f'{self.grid_size}x{self.grid_size} grid of order density'
            }
        }
        
        if self.include_merchant_features:
            desc['merchant_heatmap'] = {
                'dimension': self.grid_size * self.grid_size,
                'description': f'{self.grid_size}x{self.grid_size} grid of merchant utilization (Day 22)'
            }
        
        return desc


def test_state_encoder():
    """测试状态编码器"""
    print("="*60)
    print("测试 StateEncoder")
    print("="*60)
    
    # 创建编码器
    encoder = StateEncoder({
        'max_pending_orders': 10,
        'max_couriers': 5,
        'grid_size': 5
    })
    
    print(f"\n状态空间维度: {encoder.get_state_dimension()}")
    print(f"\n状态空间描述:")
    import json
    print(json.dumps(encoder.get_state_description(), indent=2))
    
    # 创建模拟状态
    from collections import namedtuple
    Order = namedtuple('Order', ['merchant_coords', 'customer_coords', 'earliest_pickup_time', 'latest_delivery_time'])
    Courier = namedtuple('Courier', ['current_coords', 'current_route', 'max_capacity'])
    
    mock_state = {
        'current_time': 3600.0,  # 1小时
        'pending_orders': [
            Order((116.40, 39.90), (116.41, 39.91), 3700.0, 5400.0),
            Order((116.41, 39.91), (116.42, 39.92), 3800.0, 5500.0),
        ],
        'couriers': {
            1: Courier((116.40, 39.90), [], 5),
            2: Courier((116.42, 39.92), [('pickup', 1, 123)], 5),
        },
        'bounds': (116.38, 116.44, 39.88, 39.94),
        'graph': None  # 简化
    }
    
    # 编码
    state_vector = encoder.encode(mock_state)
    print(f"\n生成的状态向量:")
    print(f"  形状: {state_vector.shape}")
    print(f"  数据类型: {state_vector.dtype}")
    print(f"  前10个元素: {state_vector[:10]}")
    print(f"  范围: [{state_vector.min():.3f}, {state_vector.max():.3f}]")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_state_encoder()

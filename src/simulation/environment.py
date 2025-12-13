"""
SimPy 仿真环境
管理订单、骑手、路网，驱动仿真流程
"""

import simpy
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import json

from .entities import Order, Courier, Merchant, OrderStatus, CourierStatus, SimulationEvent

logger = logging.getLogger(__name__)


class SimulationEnvironment:
    """仿真环境类"""
    
    def __init__(self, 
                 graph: nx.MultiDiGraph,
                 distance_matrix: np.ndarray,
                 time_matrix: np.ndarray,
                 node_mapping: Dict[str, Any],
                 config: Dict[str, Any]):
        """
        初始化仿真环境
        
        Args:
            graph: 路网图
            distance_matrix: 距离矩阵
            time_matrix: 时间矩阵
            node_mapping: 节点映射字典
            config: 仿真配置
        """
        # SimPy 环境
        self.env = simpy.Environment()
        
        # 路网相关
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.node_to_idx = node_mapping.get('node_to_idx', {})
        self.idx_to_node = node_mapping.get('idx_to_node', {})
        self.node_list = node_mapping.get('node_list', [])
        
        # 配置
        self.config = config
        self.simulation_duration = config.get('simulation_duration', 86400)
        self.dispatch_interval = config.get('dispatch_interval', 30.0)  # 调度间隔(秒)
        
        # GPS坐标模式标志（用于支持真实数据集）
        self.use_gps_coords = config.get('use_gps_coords', False)
        self.gps_coords = {}  # {location_id: (lat, lng)}
        
        # Day 24: 交通时变模型配置
        traffic_config = config.get('traffic', {})
        self.enable_time_varying_speed = traffic_config.get('enable_time_varying_speed', True)
        self.morning_peak_hours = traffic_config.get('morning_peak_hours', (7, 9))  # 早高峰
        self.evening_peak_hours = traffic_config.get('evening_peak_hours', (17, 19))  # 晚高峰
        self.lunch_peak_hours = traffic_config.get('lunch_peak_hours', (11, 13))  # 午餐高峰
        self.dinner_peak_hours = traffic_config.get('dinner_peak_hours', (18, 20))  # 晚餐高峰
        self.peak_congestion_factor = traffic_config.get('peak_congestion_factor', 0.6)  # 高峰期拥堵系数
        self.meal_congestion_factor = traffic_config.get('meal_congestion_factor', 0.8)  # 用餐高峰拥堵系数
        self.speed_noise_std = traffic_config.get('speed_noise_std', 0.1)  # 速度随机噪声标准差
        
        # Day 25: 天气场景配置
        weather_config = config.get('weather', {})
        self.weather_scenario = weather_config.get('scenario', 'sunny')
        self.weather_speed_multiplier = weather_config.get('speed_multiplier', 1.0)
        
        # 自动调度控制（RL训练时应设为False，避免与Agent决策冲突）
        self.enable_auto_dispatch = config.get('enable_auto_dispatch', True)
        
        # 实体存储
        self.orders: Dict[int, Order] = {}  # order_id -> Order
        self.couriers: Dict[int, Courier] = {}  # courier_id -> Courier
        self.merchants: Dict[int, Merchant] = {}  # merchant_id -> Merchant (Day 22: 商家不确定性建模)
        
        # 订单队列
        self.pending_orders: List[int] = []  # 待分配订单ID列表
        self.assigned_orders: List[int] = []  # 已分配订单ID列表
        self.completed_orders: List[int] = []  # 已完成订单ID列表
        
        # 事件记录
        self.events: List[SimulationEvent] = []
        
        # Day 27: 商家等餐事件收集（用于RL奖励计算）
        # 每次step()调用后会清空，返回给RL环境
        self._merchant_wait_events: List[Dict[str, Any]] = []
        
        # 统计计数器
        self.stats = {
            'total_orders': 0,
            'completed_orders': 0,
            'timeout_orders': 0,
            'total_couriers': 0
        }
        
        # Day 3/4/6: 初始化调度器（动态导入避免依赖问题）
        dispatcher_type = config.get('dispatcher_type', 'greedy')
        dispatcher_config = config.get('dispatcher_config', {})
        
        if dispatcher_type.lower() == 'ortools':
            from .dispatchers.ortools_dispatcher import ORToolsDispatcher
            self.dispatcher = ORToolsDispatcher(self, dispatcher_config)
            dispatcher_name = 'OR-Tools'
        elif dispatcher_type.lower() == 'alns':
            from .dispatchers.alns_dispatcher import ALNSDispatcher
            self.dispatcher = ALNSDispatcher(self, dispatcher_config)
            dispatcher_name = 'ALNS'
        elif dispatcher_type.lower() == 'rl':
            # RL调度器：使用训练好的PPO模型进行派单决策
            from .dispatchers.rl_dispatcher import RLDispatcher
            model_path = dispatcher_config.get('model_path', None)
            self.dispatcher = RLDispatcher(self, model_path=model_path, config=dispatcher_config)
            dispatcher_name = 'RL-PPO'
        else:
            from .dispatchers.greedy_dispatcher import GreedyDispatcher
            self.dispatcher = GreedyDispatcher(self)
            dispatcher_name = 'Greedy'
        
        logger.info("仿真环境初始化完成")
        logger.info(f"路网节点数: {len(self.node_list)}")
        logger.info(f"仿真时长: {self.simulation_duration}秒 ({self.simulation_duration/3600:.1f}小时)")
        logger.info(f"调度器: {dispatcher_name}, 调度间隔: {self.dispatch_interval}秒")
    
    def load_orders_from_csv(self, orders_file: Path) -> None:
        """
        从CSV文件加载订单数据
        支持两种格式：
        1. 路网节点格式：merchant_node, customer_node, merchant_coords, customer_coords
        2. GPS坐标格式：merchant_lat, merchant_lng, customer_lat, customer_lng
        
        Args:
            orders_file: 订单CSV文件路径
        """
        logger.info(f"从文件加载订单: {orders_file}")
        
        orders_df = pd.read_csv(orders_file)
        
        # 检测订单格式：优先使用配置文件的use_gps_coords设置
        # 只有当配置未明确设置（保持默认False）且订单文件只有GPS格式时，才自动启用GPS模式
        has_gps_columns = 'merchant_lat' in orders_df.columns
        has_node_columns = 'merchant_node' in orders_df.columns
        
        # 如果配置已明确设置use_gps_coords，则使用配置值
        # 如果订单文件同时有GPS和node列，以配置为准
        if has_node_columns and not self.use_gps_coords:
            # 配置为路网模式，且有node列，使用路网模式
            use_gps_format = False
            logger.info("使用路网节点模式（配置: use_gps_coords=False）")
        elif has_gps_columns and not has_node_columns:
            # 只有GPS列，没有node列，强制GPS模式
            use_gps_format = True
            self.use_gps_coords = True
            logger.info("订单文件只有GPS格式，自动启用GPS模式")
        elif self.use_gps_coords and has_gps_columns:
            # 配置为GPS模式，且有GPS列
            use_gps_format = True
            logger.info("使用GPS坐标模式（配置: use_gps_coords=True）")
        else:
            use_gps_format = self.use_gps_coords
            logger.info(f"使用配置的GPS模式: {self.use_gps_coords}")
        
        if use_gps_format:
            
            for _, row in orders_df.iterrows():
                order_id = int(row['order_id'])
                
                # 使用唯一的位置ID（订单ID+类型）
                merchant_loc_id = f"m_{order_id}"
                customer_loc_id = f"c_{order_id}"
                
                # 存储GPS坐标
                merchant_coords = (float(row['merchant_lat']), float(row['merchant_lng']))
                customer_coords = (float(row['customer_lat']), float(row['customer_lng']))
                self.gps_coords[merchant_loc_id] = merchant_coords
                self.gps_coords[customer_loc_id] = customer_coords
                
                # 计算或获取时间字段
                arrival = float(row['arrival_time'])
                prep_time = float(row['preparation_time'])
                
                if 'earliest_pickup_time' in row:
                    pickup_time = float(row['earliest_pickup_time'])
                else:
                    pickup_time = arrival + prep_time
                    
                if 'latest_delivery_time' in row:
                    delivery_time = float(row['latest_delivery_time'])
                elif 'deadline' in row:
                    delivery_time = float(row['deadline'])
                else:
                    delivery_time = arrival + float(row['delivery_window'])

                order = Order(
                    order_id=order_id,
                    arrival_time=arrival,
                    merchant_node=merchant_loc_id,  # 使用位置ID
                    customer_node=customer_loc_id,
                    merchant_coords=merchant_coords,
                    customer_coords=customer_coords,
                    preparation_time=prep_time,
                    delivery_window=float(row['delivery_window']),
                    earliest_pickup_time=pickup_time,
                    latest_delivery_time=delivery_time
                )
                
                self.orders[order.order_id] = order
        else:
            # 路网节点格式
            # 处理坐标列（从字符串转换为元组）
            def parse_coords(coords_str):
                if isinstance(coords_str, str):
                    coords_str = coords_str.strip('()')
                    x, y = map(float, coords_str.split(','))
                    return (x, y)
                return coords_str
            
            # 检测坐标列来源：优先使用merchant_coords，否则使用GPS坐标列
            has_coords_columns = 'merchant_coords' in orders_df.columns
            has_gps_columns = 'merchant_lat' in orders_df.columns and 'merchant_lng' in orders_df.columns
            
            if has_coords_columns:
                orders_df['merchant_coords'] = orders_df['merchant_coords'].apply(parse_coords)
                orders_df['customer_coords'] = orders_df['customer_coords'].apply(parse_coords)
            elif has_gps_columns:
                # 使用GPS坐标作为coords
                logger.info("路网模式：使用GPS坐标列作为坐标来源")
            
            # 检测graph中节点的类型（字符串或整数）
            node_sample = list(self.graph.nodes())[0] if len(self.graph.nodes()) > 0 else None
            use_str_nodes = isinstance(node_sample, str)
            
            # 创建订单对象
            for _, row in orders_df.iterrows():
                # 根据graph节点类型转换节点ID
                # 注意：先转为int再转为str，避免浮点数格式（如"601669728.0"）
                merchant_node = str(int(row['merchant_node'])) if use_str_nodes else int(row['merchant_node'])
                customer_node = str(int(row['customer_node'])) if use_str_nodes else int(row['customer_node'])
                
                # 获取坐标：优先使用merchant_coords列，否则使用GPS坐标
                if has_coords_columns:
                    merchant_coords = row['merchant_coords']
                    customer_coords = row['customer_coords']
                elif has_gps_columns:
                    # GPS坐标格式：(lat, lng)
                    merchant_coords = (float(row['merchant_lat']), float(row['merchant_lng']))
                    customer_coords = (float(row['customer_lat']), float(row['customer_lng']))
                else:
                    # 尝试从路网节点获取坐标
                    try:
                        m_data = self.graph.nodes[merchant_node]
                        c_data = self.graph.nodes[customer_node]
                        merchant_coords = (float(m_data['y']), float(m_data['x']))
                        customer_coords = (float(c_data['y']), float(c_data['x']))
                    except (KeyError, TypeError):
                        merchant_coords = (0.0, 0.0)
                        customer_coords = (0.0, 0.0)
                
                # 计算或获取时间字段
                arrival = float(row['arrival_time'])
                prep_time = float(row['preparation_time'])
                
                if 'earliest_pickup_time' in row:
                    pickup_time = float(row['earliest_pickup_time'])
                else:
                    pickup_time = arrival + prep_time
                    
                if 'latest_delivery_time' in row:
                    delivery_time = float(row['latest_delivery_time'])
                elif 'deadline' in row:
                    delivery_time = float(row['deadline'])
                else:
                    delivery_time = arrival + float(row['delivery_window'])

                order = Order(
                    order_id=int(row['order_id']),
                    arrival_time=arrival,
                    merchant_node=merchant_node,
                    customer_node=customer_node,
                    merchant_coords=merchant_coords,
                    customer_coords=customer_coords,
                    preparation_time=prep_time,
                    delivery_window=float(row['delivery_window']),
                    earliest_pickup_time=pickup_time,
                    latest_delivery_time=delivery_time
                )
                
                self.orders[order.order_id] = order
        
        self.stats['total_orders'] = len(self.orders)
        logger.info(f"加载了 {len(self.orders)} 个订单 (GPS模式: {self.use_gps_coords})")
        
        # 自动调整订单时间到仿真范围内
        self._adjust_order_arrival_times()
        
        # Day 22: 创建商家对象并关联订单
        self._initialize_merchants()
    
    def load_orders_from_raw_data(self, orders_raw: list) -> None:
        """
        从预加载的原始数据加载订单（性能优化）
        
        避免每次reset都从CSV文件读取，直接使用内存中的数据
        
        Args:
            orders_raw: 订单原始数据列表（字典格式，与CSV列对应）
        """
        import copy
        
        logger.debug(f"从预加载数据加载订单: {len(orders_raw)} 条")
        
        # 清空现有订单
        self.orders.clear()
        
        # 检测订单格式
        if not orders_raw:
            logger.warning("预加载订单数据为空")
            return
        
        sample_row = orders_raw[0]
        has_gps_columns = 'merchant_lat' in sample_row
        has_node_columns = 'merchant_node' in sample_row
        
        # 确定使用的格式
        if has_node_columns and not self.use_gps_coords:
            use_gps_format = False
        elif has_gps_columns and not has_node_columns:
            use_gps_format = True
            self.use_gps_coords = True
        else:
            use_gps_format = self.use_gps_coords
        
        # 检测graph中节点的类型
        node_sample = list(self.graph.nodes())[0] if len(self.graph.nodes()) > 0 else None
        use_str_nodes = isinstance(node_sample, str)
        
        for row in orders_raw:
            if use_gps_format:
                order_id = int(row['order_id'])
                merchant_loc_id = f"m_{order_id}"
                customer_loc_id = f"c_{order_id}"
                
                merchant_coords = (float(row['merchant_lat']), float(row['merchant_lng']))
                customer_coords = (float(row['customer_lat']), float(row['customer_lng']))
                self.gps_coords[merchant_loc_id] = merchant_coords
                self.gps_coords[customer_loc_id] = customer_coords
                
                arrival = float(row['arrival_time'])
                prep_time = float(row['preparation_time'])
                
                pickup_time = float(row.get('earliest_pickup_time', arrival + prep_time))
                delivery_time = float(row.get('latest_delivery_time', 
                                             row.get('deadline', arrival + float(row['delivery_window']))))
                
                order = Order(
                    order_id=order_id,
                    arrival_time=arrival,
                    merchant_node=merchant_loc_id,
                    customer_node=customer_loc_id,
                    merchant_coords=merchant_coords,
                    customer_coords=customer_coords,
                    preparation_time=prep_time,
                    delivery_window=float(row['delivery_window']),
                    earliest_pickup_time=pickup_time,
                    latest_delivery_time=delivery_time
                )
            else:
                # 路网节点格式
                merchant_node = str(int(row['merchant_node'])) if use_str_nodes else int(row['merchant_node'])
                customer_node = str(int(row['customer_node'])) if use_str_nodes else int(row['customer_node'])
                
                # 获取坐标
                if 'merchant_coords' in row:
                    coords_str = row['merchant_coords']
                    if isinstance(coords_str, str):
                        coords_str = coords_str.strip('()')
                        x, y = map(float, coords_str.split(','))
                        merchant_coords = (x, y)
                    else:
                        merchant_coords = coords_str
                    
                    coords_str = row['customer_coords']
                    if isinstance(coords_str, str):
                        coords_str = coords_str.strip('()')
                        x, y = map(float, coords_str.split(','))
                        customer_coords = (x, y)
                    else:
                        customer_coords = coords_str
                elif has_gps_columns:
                    merchant_coords = (float(row['merchant_lat']), float(row['merchant_lng']))
                    customer_coords = (float(row['customer_lat']), float(row['customer_lng']))
                else:
                    try:
                        m_data = self.graph.nodes[merchant_node]
                        c_data = self.graph.nodes[customer_node]
                        merchant_coords = (float(m_data['y']), float(m_data['x']))
                        customer_coords = (float(c_data['y']), float(c_data['x']))
                    except (KeyError, TypeError):
                        merchant_coords = (0.0, 0.0)
                        customer_coords = (0.0, 0.0)
                
                arrival = float(row['arrival_time'])
                prep_time = float(row['preparation_time'])
                pickup_time = float(row.get('earliest_pickup_time', arrival + prep_time))
                delivery_time = float(row.get('latest_delivery_time',
                                             row.get('deadline', arrival + float(row['delivery_window']))))
                
                order = Order(
                    order_id=int(row['order_id']),
                    arrival_time=arrival,
                    merchant_node=merchant_node,
                    customer_node=customer_node,
                    merchant_coords=merchant_coords,
                    customer_coords=customer_coords,
                    preparation_time=prep_time,
                    delivery_window=float(row['delivery_window']),
                    earliest_pickup_time=pickup_time,
                    latest_delivery_time=delivery_time
                )
            
            self.orders[order.order_id] = order
        
        self.stats['total_orders'] = len(self.orders)
        logger.debug(f"从预加载数据加载了 {len(self.orders)} 个订单")
        
        # 自动调整订单时间到仿真范围内
        self._adjust_order_arrival_times()
        
        # Day 22: 创建商家对象并关联订单
        self._initialize_merchants()

    def _adjust_order_arrival_times(self):
        """
        调整订单到达时间到仿真时间范围内
        
        如果订单的原始到达时间超出仿真时长，则将所有订单的到达时间
        线性映射到[0, simulation_duration * 0.8]范围内，保留20%的仿真时间用于配送
        """
        if not self.orders:
            return
        
        simulation_duration = self.simulation_duration
        
        arrival_times = [order.arrival_time for order in self.orders.values()]
        min_arrival = min(arrival_times)
        max_arrival = max(arrival_times)
        
        eps = 1e-6
        
        if min_arrival < -eps:
            raise ValueError(f"订单到达时间包含负值: min_arrival={min_arrival:.6f}s")
        
        if max_arrival > simulation_duration + eps:
            out_of_range_orders = [
                order.order_id for order in self.orders.values()
                if order.arrival_time > simulation_duration + eps
            ]
            raise ValueError(
                "订单到达时间超出仿真时长，可能导致订单永远不会到达而被静默丢弃。"
                f"arrival_time范围: {min_arrival:.0f}s - {max_arrival:.0f}s, "
                f"simulation_duration={simulation_duration}s, "
                f"out_of_range_orders(前10个)={out_of_range_orders[:10]} (总数={len(out_of_range_orders)})"
            )
        
        logger.info(f"订单到达时间在仿真范围内，无需调整 (范围: {min_arrival:.0f}s - {max_arrival:.0f}s)")
        return
    
    def _initialize_merchants(self) -> None:
        """
        Day 22: 基于订单位置创建商家对象
        
        将位置相近的订单关联到同一商家，实现商家排队机制
        """
        from collections import defaultdict
        
        # 基于商家位置聚类订单
        # 使用位置哈希（四舍五入到小数点后4位，约10米精度）
        location_to_orders = defaultdict(list)
        
        for order_id, order in self.orders.items():
            if self.use_gps_coords:
                # GPS模式：使用经纬度
                lat, lng = order.merchant_coords
                loc_key = (round(lat, 4), round(lng, 4))
            else:
                # 路网模式：使用节点ID
                loc_key = order.merchant_node
            
            location_to_orders[loc_key].append(order_id)
        
        # 为每个位置创建商家
        merchant_id = 0
        for loc_key, order_ids in location_to_orders.items():
            if self.use_gps_coords:
                coords = loc_key
                node_id = f"merchant_{merchant_id}"
            else:
                node_id = loc_key
                # 获取节点坐标
                sample_order = self.orders[order_ids[0]]
                coords = sample_order.merchant_coords
            
            # 创建商家（使用随机化参数增加差异性）
            merchant = Merchant(
                merchant_id=merchant_id,
                node_id=node_id,
                coords=coords,
                name=f"商家{merchant_id}",
                # 随机化备餐参数（增加商家间差异）
                prep_time_alpha=np.random.uniform(1.5, 2.5),
                prep_time_beta=np.random.uniform(0.015, 0.025),
                service_rate=np.random.uniform(0.008, 0.012),
                num_servers=np.random.randint(1, 4)
            )
            
            self.merchants[merchant_id] = merchant
            
            # 关联订单到商家
            for order_id in order_ids:
                order = self.orders[order_id]
                order.merchant_id = merchant_id
                
                # 设置订单的商家相关参数（继承商家参数）
                order.prep_time_alpha = merchant.prep_time_alpha
                order.prep_time_beta = merchant.prep_time_beta
                
                # 将订单加入商家队列
                merchant.add_order_to_queue(order_id)
                order.merchant_queue_position = merchant.get_queue_length()
                
                # 预估出餐时间
                order.estimated_ready_time = merchant.estimate_ready_time(order.arrival_time)
            
            merchant_id += 1
        
        logger.info(f"创建了 {len(self.merchants)} 个商家，关联 {len(self.orders)} 个订单")
    
    def initialize_couriers(self, num_couriers: int, courier_config: Dict[str, Any]) -> None:
        """
        初始化骑手
        
        Args:
            num_couriers: 骑手数量
            courier_config: 骑手配置
        """
        logger.info(f"初始化 {num_couriers} 个骑手...")
        
        # 获取配置
        speed_mean = courier_config.get('speed', {}).get('mean', 15.0)
        speed_std = courier_config.get('speed', {}).get('std', 2.0)
        speed_min = courier_config.get('speed', {}).get('min', 10.0)
        speed_max = courier_config.get('speed', {}).get('max', 20.0)
        max_capacity = courier_config.get('capacity', {}).get('max_orders', 5)
        
        if self.use_gps_coords:
            # GPS模式：从订单的商家位置中随机选择初始位置
            # 过滤掉坐标异常的位置（如0,0）
            merchant_locs = []
            for loc_id in self.gps_coords.keys():
                if loc_id.startswith('m_'):
                    coords = self.gps_coords[loc_id]
                    # 只选择有效坐标（纬度和经度都大于1）
                    if coords[0] > 1 and coords[1] > 1:
                        merchant_locs.append(loc_id)
            
            if not merchant_locs:
                # 如果没有有效商家位置，使用默认位置（印度中部）
                merchant_locs = ['default']
                self.gps_coords['default'] = (20.0, 77.0)
                logger.warning("没有有效的商家位置，骑手将初始化在默认位置")
            
            for i in range(num_couriers):
                loc_id = merchant_locs[i % len(merchant_locs)]
                coords = self.gps_coords[loc_id]
                
                # 为每个骑手分配速度
                speed = np.clip(
                    np.random.normal(speed_mean, speed_std),
                    speed_min,
                    speed_max
                )
                
                # 创建骑手专属位置ID
                courier_loc_id = f"courier_{i+1}"
                self.gps_coords[courier_loc_id] = coords
                
                courier = Courier(
                    courier_id=i + 1,
                    initial_node=courier_loc_id,
                    initial_coords=coords,
                    max_capacity=max_capacity,
                    speed_kph=speed
                )
                
                self.couriers[courier.courier_id] = courier
        else:
            # 路网模式：从路网节点中随机选择
            initial_nodes = np.random.choice(self.node_list, size=num_couriers, replace=True)
            
            for i in range(num_couriers):
                node_id = initial_nodes[i]
                node_data = self.graph.nodes[node_id]
                coords = (node_data['x'], node_data['y'])
                
                # 为每个骑手分配速度（正态分布）
                speed = np.clip(
                    np.random.normal(speed_mean, speed_std),
                    speed_min,
                    speed_max
                )
                
                courier = Courier(
                    courier_id=i + 1,
                    initial_node=node_id,
                    initial_coords=coords,
                    max_capacity=max_capacity,
                    speed_kph=speed
                )
                
                self.couriers[courier.courier_id] = courier
        
        self.stats['total_couriers'] = len(self.couriers)
        logger.info(f"骑手初始化完成，平均速度: {speed_mean} km/h (GPS模式: {self.use_gps_coords})")
    
    def get_time_varying_speed(self, base_speed: float, current_time: float = None) -> float:
        """
        Day 24: 获取时变交通速度
        
        考虑因素：
        1. 交通高峰期（早晚高峰）
        2. 用餐高峰期（午餐、晚餐）
        3. 随机扰动（模拟交通不确定性）
        4. 天气影响
        
        Args:
            base_speed: 基础速度 (km/h)
            current_time: 当前仿真时间（秒），如果为None则使用self.env.now
        
        Returns:
            实际速度 (km/h)
        """
        if not self.enable_time_varying_speed:
            return base_speed * self.weather_speed_multiplier
        
        if current_time is None:
            current_time = self.env.now
        
        # 计算当前小时（0-23）
        hour = (current_time / 3600) % 24
        
        # 确定拥堵系数
        congestion_factor = 1.0
        
        # 交通高峰期（最大拥堵）
        if self.morning_peak_hours[0] <= hour <= self.morning_peak_hours[1]:
            congestion_factor = self.peak_congestion_factor
        elif self.evening_peak_hours[0] <= hour <= self.evening_peak_hours[1]:
            congestion_factor = self.peak_congestion_factor
        # 用餐高峰期（中等拥堵）
        elif self.lunch_peak_hours[0] <= hour <= self.lunch_peak_hours[1]:
            congestion_factor = self.meal_congestion_factor
        elif self.dinner_peak_hours[0] <= hour <= self.dinner_peak_hours[1]:
            congestion_factor = self.meal_congestion_factor
        
        # 添加随机扰动（模拟交通不确定性）
        noise = np.random.normal(0, self.speed_noise_std)
        actual_factor = congestion_factor * (1 + noise)
        
        # 限制因子范围（避免极端值）
        actual_factor = max(0.4, min(1.2, actual_factor))
        
        # 应用天气影响
        final_speed = base_speed * actual_factor * self.weather_speed_multiplier
        
        # 确保速度不低于基础速度的30%
        return max(final_speed, base_speed * 0.3)
    
    def get_travel_time(self, from_node, to_node, speed_kph: float = None) -> float:
        """
        获取两点间行程时间
        Day 24: 支持时变速度
        支持GPS坐标模式和路网节点模式
        
        Args:
            from_node: 起点节点ID或GPS位置ID
            to_node: 终点节点ID或GPS位置ID
            speed_kph: 基础速度(km/h)，如果为None则使用默认速度15km/h
        
        Returns:
            行程时间（秒）
        """
        # GPS坐标模式：使用距离和速度计算时间
        if self.use_gps_coords:
            distance = self.get_distance(from_node, to_node)  # 米
            actual_speed = self.get_time_varying_speed(speed_kph or 15.0)
            return (distance / 1000) / actual_speed * 3600  # 转换为秒
        
        # 路网节点模式
        from_node_str = str(from_node)
        to_node_str = str(to_node)
        
        if from_node_str not in self.node_to_idx or to_node_str not in self.node_to_idx:
            raise ValueError(f"Node {from_node} or {to_node} not in node mapping")
        
        from_idx = self.node_to_idx[from_node_str]
        to_idx = self.node_to_idx[to_node_str]
        
        if speed_kph is not None:
            # Day 24: 使用时变速度计算时间
            actual_speed = self.get_time_varying_speed(speed_kph)
            distance = self.distance_matrix[from_idx, to_idx]
            if np.isinf(distance):
                raise ValueError(f"No path from node {from_node} to {to_node}")
            return (distance / 1000) / actual_speed * 3600  # 转换为秒
        else:
            # 使用预计算的时间矩阵（不考虑时变，用于快速估算）
            travel_time = self.time_matrix[from_idx, to_idx]
            if np.isinf(travel_time):
                raise ValueError(f"No path from node {from_node} to {to_node}")
            # 应用天气影响
            return travel_time / self.weather_speed_multiplier
    
    def get_distance(self, from_node, to_node) -> float:
        """
        获取两点间距离
        
        Args:
            from_node: 起点节点ID或GPS坐标(lat, lng)
            to_node: 终点节点ID或GPS坐标(lat, lng)
        
        Returns:
            距离（米）
        """
        # GPS坐标模式：使用Haversine公式计算距离
        if self.use_gps_coords:
            # 从GPS坐标字典获取坐标
            if isinstance(from_node, tuple):
                from_coords = from_node
            elif from_node in self.gps_coords:
                from_coords = self.gps_coords[from_node]
            else:
                raise ValueError(f"GPS coords not found for {from_node}")
            
            if isinstance(to_node, tuple):
                to_coords = to_node
            elif to_node in self.gps_coords:
                to_coords = self.gps_coords[to_node]
            else:
                raise ValueError(f"GPS coords not found for {to_node}")
            
            return self._haversine_distance(from_coords, to_coords)
        
        # 路网节点模式：使用距离矩阵
        from_node_str = str(from_node)
        to_node_str = str(to_node)
        
        if from_node_str not in self.node_to_idx or to_node_str not in self.node_to_idx:
            raise ValueError(f"Node {from_node} or {to_node} not in node mapping")
        
        from_idx = self.node_to_idx[from_node_str]
        to_idx = self.node_to_idx[to_node_str]
        
        distance = self.distance_matrix[from_idx, to_idx]
        if np.isinf(distance):
            raise ValueError(f"No path from node {from_node} to {to_node}")
        
        return distance
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        使用Haversine公式计算两个GPS坐标点间的距离
        
        Args:
            coord1: (lat, lng)
            coord2: (lat, lng)
        
        Returns:
            距离（米）
        """
        R = 6371000  # 地球半径（米）
        
        lat1, lng1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lng2 = np.radians(coord2[0]), np.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def record_event(self, event_type: str, entity_id: int, details: Dict[str, Any] = None) -> None:
        """
        记录仿真事件
        
        Args:
            event_type: 事件类型
            entity_id: 实体ID
            details: 详细信息
        """
        event = SimulationEvent(
            timestamp=self.env.now,
            event_type=event_type,
            entity_id=entity_id,
            details=details or {}
        )
        self.events.append(event)
    
    def order_arrival_process(self):
        """
        订单到达过程（SimPy进程）
        按照订单的到达时间依次触发订单进入系统
        """
        logger.info("启动订单到达进程")
        
        # 按到达时间排序订单
        sorted_orders = sorted(self.orders.values(), key=lambda o: o.arrival_time)
        
        for order in sorted_orders:
            # 等待到订单到达时间
            yield self.env.timeout(order.arrival_time - self.env.now)
            
            # 订单进入待分配队列
            self.pending_orders.append(order.order_id)
            
            # 记录事件
            self.record_event(
                event_type='order_arrival',
                entity_id=order.order_id,
                details={
                    'merchant_node': order.merchant_node,
                    'customer_node': order.customer_node,
                    'earliest_pickup_time': order.earliest_pickup_time,
                    'latest_delivery_time': order.latest_delivery_time
                }
            )
            
            logger.debug(f"[{self.env.now:.1f}s] 订单 {order.order_id} 到达，进入待分配队列 (队列长度: {len(self.pending_orders)})")
    
    def dispatch_process(self):
        """
        调度触发进程（Day 3新增）
        定期触发调度器分配待处理订单
        """
        logger.info(f"调度进程启动，间隔: {self.dispatch_interval}秒")
        
        dispatch_count = 0
        while True:
            # 等待调度间隔
            yield self.env.timeout(self.dispatch_interval)
            
            dispatch_count += 1
            
            # 如果有待分配订单，触发调度
            if len(self.pending_orders) > 0:
                logger.info(f"[{self.env.now:.1f}s] 第{dispatch_count}次调度触发，待分配订单: {len(self.pending_orders)}")
                assigned = self.dispatcher.dispatch_pending_orders()
                logger.info(f"[{self.env.now:.1f}s] 本次调度分配了 {assigned} 个订单，剩余待分配: {len(self.pending_orders)}")
            else:
                logger.debug(f"[{self.env.now:.1f}s] 第{dispatch_count}次调度检查，无待分配订单")
    
    def courier_process(self, courier_id: int):
        """
        骑手过程（SimPy进程）
        Day 3: 执行订单任务
        
        Args:
            courier_id: 骑手ID
        """
        courier = self.couriers[courier_id]
        logger.debug(f"骑手 {courier_id} 开始工作，初始位置: 节点{courier.current_node}")
        
        while True:
            # 检查是否有任务
            if len(courier.current_route) > 0:
                # 【关键】保存当前任务的完整信息，用于后续精确移除
                current_task = courier.current_route[0]
                action, order_id, target_node = current_task
                order = self.orders[order_id]
                
                if action == 'pickup':
                    # 执行取货
                    yield from self._execute_pickup(courier, order, target_node)
                elif action == 'delivery':
                    # 执行配送
                    yield from self._execute_delivery(courier, order, target_node)
                
                # 【修复竞争条件】精确移除刚才执行的任务，而不是简单地pop(0)
                # 因为在执行任务期间（yield期间），RL Agent可能插入了新任务到路线开头
                if courier.current_route:
                    # 查找并移除刚才执行的具体任务
                    task_removed = False
                    for i, task in enumerate(courier.current_route):
                        if task[0] == action and task[1] == order_id and task[2] == target_node:
                            courier.current_route.pop(i)
                            task_removed = True
                            break
                    
                    if not task_removed:
                        # 如果找不到原任务（可能已被其他操作移除），记录警告
                        logger.warning(
                            f"[{self.env.now:.1f}s] 骑手{courier.courier_id}执行的任务"
                            f"({action},{order_id})在路线中未找到，可能已被移除"
                        )
                else:
                    logger.warning(
                        f"[{self.env.now:.1f}s] 骑手{courier.courier_id}的路线意外为空，"
                        f"可能在任务执行过程中被清空"
                    )
            else:
                # 如果没有任务，等待一段时间再检查
                courier.status = CourierStatus.IDLE
                yield self.env.timeout(1)
    
    def _execute_pickup(self, courier, order, target_node):
        """
        执行取货任务
        
        Args:
            courier: 骑手对象
            order: 订单对象
            target_node: 目标节点ID（商家位置）
        """
        # 检查订单状态，如果已经在取货或取货完成则跳过
        # 【Bug修复】也检查PENDING状态 - 当ALNS将订单移除并回退为PENDING时，
        # 骑手可能已经在移动中，到达后不应该尝试取货
        from .entities import OrderStatus
        if order.status in [OrderStatus.PENDING, OrderStatus.PICKING_UP, OrderStatus.PICKED_UP, OrderStatus.DELIVERING, OrderStatus.DELIVERED]:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}状态异常(状态:{order.status})，跳过取货"
            )
            return
        
        # 1. 前往取货点
        if courier.current_node != target_node:
            courier.status = CourierStatus.MOVING_TO_PICKUP
            travel_time = self.get_travel_time(courier.current_node, target_node, courier.speed_kph)
            distance = self.get_distance(courier.current_node, target_node)
            
            logger.debug(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}前往取货点(订单{order.order_id}), 距离{distance:.0f}m")
            yield self.env.timeout(travel_time)
            
            # 更新位置和统计
            if self.use_gps_coords:
                coords = self.gps_coords.get(target_node, (0, 0))
            else:
                node_data = self.graph.nodes[target_node]
                coords = (node_data['x'], node_data['y'])
            courier.update_position(target_node, coords)
            courier.add_distance(distance)
            courier.add_time(travel_time)
        
        # 2. 取货 - 再次检查状态（防止移动期间状态被改变）
        # 【Bug修复】也检查PENDING状态 - ALNS可能在骑手移动期间将订单移除
        if order.status in [OrderStatus.PENDING, OrderStatus.PICKING_UP, OrderStatus.PICKED_UP, OrderStatus.DELIVERING, OrderStatus.DELIVERED]:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}在骑手移动期间状态已改变(状态:{order.status})，跳过取货"
            )
            return
        
        courier.status = CourierStatus.PICKING_UP
        order.start_pickup(self.env.now)
        
        # Day 27: 记录骑手到达商家时间，计算等餐时间
        courier_arrival_time = self.env.now
        order.record_courier_arrival(courier_arrival_time)
        
        # 检查餐品是否已准备好
        # 如果订单有estimated_ready_time，使用它来模拟商家备餐
        wait_time = 0.0
        if hasattr(order, 'estimated_ready_time') and order.estimated_ready_time is not None:
            # 餐品准备好的时间
            food_ready_time = order.estimated_ready_time
            
            if courier_arrival_time < food_ready_time:
                # 骑手早到，需要等餐
                wait_time = food_ready_time - courier_arrival_time
                order.set_food_ready(food_ready_time)
                order.calculate_waiting_time(food_ready_time)
                
                logger.debug(
                    f"[{self.env.now:.1f}s] 骑手{courier.courier_id}在商家等餐 {wait_time:.1f}秒 (订单{order.order_id})"
                )
                
                # 记录等餐事件，供RL奖励计算使用
                self._merchant_wait_events.append({
                    'order_id': order.order_id,
                    'courier_id': courier.courier_id,
                    'merchant_id': order.merchant_node,
                    'wait_time': wait_time,
                    'courier_arrival_time': courier_arrival_time,
                    'food_ready_time': food_ready_time
                })
                
                # 等待餐品准备好
                yield self.env.timeout(wait_time)
            else:
                # 骑手晚到，餐已准备好
                order.set_food_ready(food_ready_time)
                order.calculate_waiting_time(food_ready_time)
        
        logger.debug(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}开始取货(订单{order.order_id})")
        yield self.env.timeout(order.pickup_duration)
        
        order.complete_pickup(self.env.now)
        self.record_event('pickup_complete', order.order_id, {
            'courier_id': courier.courier_id,
            'wait_time_at_merchant': wait_time
        })
        logger.info(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}完成取货(订单{order.order_id})")
    
    def _execute_delivery(self, courier, order, target_node):
        """
        执行配送任务
        
        Args:
            courier: 骑手对象
            order: 订单对象
            target_node: 目标节点ID（客户位置）
        """
        # 检查订单状态
        from .entities import OrderStatus
        if order.status == OrderStatus.DELIVERED:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}已配送完成，跳过配送"
            )
            return
        
        # 如果订单已经在配送中，跳过（避免重复执行）
        if order.status == OrderStatus.DELIVERING:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}已在配送中，跳过重复配送"
            )
            return
        
        # 如果订单还没取货，跳过配送（等待取货完成）
        if order.status != OrderStatus.PICKED_UP:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}尚未取货(状态:{order.status})，跳过配送"
            )
            return
        
        # 1. 前往配送点
        if courier.current_node != target_node:
            courier.status = CourierStatus.MOVING_TO_DELIVERY
            travel_time = self.get_travel_time(courier.current_node, target_node, courier.speed_kph)
            distance = self.get_distance(courier.current_node, target_node)
            
            logger.debug(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}前往配送点(订单{order.order_id}), 距离{distance:.0f}m")
            yield self.env.timeout(travel_time)
            
            # 更新位置和统计
            if self.use_gps_coords:
                coords = self.gps_coords.get(target_node, (0, 0))
            else:
                node_data = self.graph.nodes[target_node]
                coords = (node_data['x'], node_data['y'])
            courier.update_position(target_node, coords)
            courier.add_distance(distance)
            courier.add_time(travel_time)
        
        # 2. 送货（再次检查状态，防止并发问题）
        if order.status != OrderStatus.PICKED_UP:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}状态已变更(状态:{order.status})，跳过配送"
            )
            return
        
        # 检查订单是否属于当前骑手
        if order.order_id not in courier.assigned_orders:
            logger.warning(
                f"[{self.env.now:.1f}s] 订单{order.order_id}不属于骑手{courier.courier_id}，跳过配送"
            )
            return
        
        courier.status = CourierStatus.DELIVERING
        order.start_delivery(self.env.now)
        
        logger.debug(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}开始送货(订单{order.order_id})")
        yield self.env.timeout(order.dropoff_duration)
        
        order.complete_delivery(self.env.now)
        courier.remove_order(order.order_id)
        
        # 移动到完成列表（避免重复添加）
        if order.order_id in self.assigned_orders:
            self.assigned_orders.remove(order.order_id)
        if order.order_id not in self.completed_orders:
            self.completed_orders.append(order.order_id)
        self.stats['completed_orders'] += 1
        
        # 检查超时
        if order.is_timeout(self.env.now):
            self.stats['timeout_orders'] += 1
            logger.warning(f"[{self.env.now:.1f}s] 订单{order.order_id}超时送达")
        
        self.record_event('delivery_complete', order.order_id, {
            'courier_id': courier.courier_id,
            'delivery_time': self.env.now,
            'is_timeout': order.is_timeout(self.env.now)
        })
        logger.info(f"[{self.env.now:.1f}s] 骑手{courier.courier_id}完成配送(订单{order.order_id})")
    
    def initialize_processes(self):
        """
        初始化所有仿真进程（不运行）
        用于RL环境中需要手动控制仿真步进的场景
        """
        logger.info("初始化仿真进程...")
        
        # 启动订单到达进程
        self.env.process(self.order_arrival_process())
        
        # 启动调度进程（仅在enable_auto_dispatch=True时）
        # RL训练时应禁用，避免与Agent决策冲突
        if self.enable_auto_dispatch:
            self.env.process(self.dispatch_process())
            logger.info("自动调度进程已启动")
        else:
            logger.info("自动调度进程已禁用（RL模式）")
        
        # 启动所有骑手进程
        for courier_id in self.couriers:
            self.env.process(self.courier_process(courier_id))
        
        logger.info(f"仿真进程初始化完成 (订单: {len(self.orders)}, 骑手: {len(self.couriers)})")
    
    def run(self, until: Optional[float] = None):
        """
        运行仿真
        
        Args:
            until: 仿真终止时间，若为None则使用配置的simulation_duration
        """
        if until is None:
            until = self.simulation_duration
        
        logger.info(f"开始仿真，时长: {until}秒 ({until/3600:.1f}小时)")
        
        # 初始化进程
        self.initialize_processes()
        
        # 运行仿真
        self.env.run(until=until)
        
        logger.info("仿真完成")
        self._print_summary()
    
    def step(self, duration: float) -> Dict[str, Any]:
        """
        推进仿真指定时长（用于RL环境的逐步交互）
        
        Args:
            duration: 推进的时长（秒）
        
        Returns:
            步进后的状态信息
        """
        target_time = self.env.now + duration
        
        # 推进仿真到目标时间
        try:
            self.env.run(until=target_time)
        except simpy.core.EmptySchedule:
            # 所有进程都已完成，仿真自然结束
            logger.debug(f"仿真在 {self.env.now:.1f}s 自然结束")
        
        # Day 27: 收集本次step期间发生的等餐事件
        # 返回后清空，避免重复计算
        merchant_wait_events = self._merchant_wait_events.copy()
        self._merchant_wait_events.clear()
        
        # 返回当前状态信息
        return {
            'current_time': self.env.now,
            'pending_orders': len(self.pending_orders),
            'assigned_orders': len(self.assigned_orders),
            'completed_orders': len(self.completed_orders),
            'timeout_orders': self.stats.get('timeout_orders', 0),
            # Day 27: 商家等餐事件列表，用于RL奖励计算中的等餐惩罚
            # 格式: [{'order_id': int, 'courier_id': int, 'merchant_id': int, 
            #         'wait_time': float, 'courier_arrival_time': float, 'food_ready_time': float}, ...]
            'merchant_wait_events': merchant_wait_events,
            # TODO: 延迟派单合理性判断（未实现）
            # 'delay_justified': False,
        }
    
    def _print_summary(self):
        """打印仿真摘要"""
        logger.info("="*60)
        logger.info("仿真摘要")
        logger.info("="*60)
        logger.info(f"仿真时长: {self.env.now:.1f}秒 ({self.env.now/3600:.1f}小时)")
        logger.info(f"总订单数: {self.stats['total_orders']}")
        logger.info(f"待分配订单: {len(self.pending_orders)}")
        logger.info(f"已分配订单: {len(self.assigned_orders)}")
        logger.info(f"已完成订单: {len(self.completed_orders)}")
        logger.info(f"总事件数: {len(self.events)}")
        logger.info(f"总骑手数: {self.stats['total_couriers']}")
        
        # 统计订单到达率
        if len(self.events) > 0:
            arrival_events = [e for e in self.events if e.event_type == 'order_arrival']
            if len(arrival_events) > 1:
                time_span = arrival_events[-1].timestamp - arrival_events[0].timestamp
                if time_span > 0:
                    arrival_rate = len(arrival_events) / time_span
                    logger.info(f"平均到达率: {arrival_rate:.4f} 订单/秒 ({arrival_rate*3600:.1f} 订单/小时)")
        
        logger.info("="*60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取仿真统计信息
        
        Returns:
            统计字典
        """
        stats = {
            'simulation_time': self.env.now,
            'total_orders': self.stats['total_orders'],
            'pending_orders': len(self.pending_orders),
            'assigned_orders': len(self.assigned_orders),
            'completed_orders': len(self.completed_orders),
            'total_events': len(self.events),
            'total_couriers': self.stats['total_couriers']
        }
        
        # 添加订单状态统计
        status_counts = defaultdict(int)
        for order in self.orders.values():
            status_counts[order.status.value] += 1
        stats['order_status_counts'] = dict(status_counts)
        
        # 添加骑手状态统计
        courier_status_counts = defaultdict(int)
        for courier in self.couriers.values():
            courier_status_counts[courier.status.value] += 1
        stats['courier_status_counts'] = dict(courier_status_counts)
        
        return stats
    
    def save_events(self, output_file: Path) -> None:
        """
        保存事件日志
        
        Args:
            output_file: 输出文件路径
        """
        logger.info(f"保存事件日志到: {output_file}")
        
        events_data = [event.to_dict() for event in self.events]
        events_df = pd.DataFrame(events_data)
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.suffix == '.csv':
            events_df.to_csv(output_file, index=False, encoding='utf-8')
        elif output_file.suffix == '.json':
            events_df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_file.suffix}")
        
        logger.info(f"事件日志已保存，共 {len(events_data)} 条记录")
    
    def save_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        保存仿真结果
        
        Args:
            output_dir: 输出目录
        
        Returns:
            保存的文件路径字典
        """
        logger.info(f"保存仿真结果到: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存事件日志
        events_file = output_dir / "events.csv"
        self.save_events(events_file)
        saved_files['events'] = events_file
        
        # 保存统计信息
        stats = self.get_statistics()
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        saved_files['statistics'] = stats_file
        logger.info(f"统计信息已保存: {stats_file}")
        
        # 保存订单状态
        orders_data = [order.to_dict() for order in self.orders.values()]
        orders_df = pd.DataFrame(orders_data)
        orders_file = output_dir / "orders_result.csv"
        orders_df.to_csv(orders_file, index=False, encoding='utf-8')
        saved_files['orders'] = orders_file
        logger.info(f"订单状态已保存: {orders_file}")
        
        # 保存骑手状态
        couriers_data = [courier.to_dict() for courier in self.couriers.values()]
        couriers_df = pd.DataFrame(couriers_data)
        couriers_file = output_dir / "couriers_result.csv"
        couriers_df.to_csv(couriers_file, index=False, encoding='utf-8')
        saved_files['couriers'] = couriers_file
        logger.info(f"骑手状态已保存: {couriers_file}")
        
        logger.info("仿真结果保存完成")
        return saved_files

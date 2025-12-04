"""
仿真实体类定义
包含订单(Order)和骑手(Courier)的数据结构与状态机
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy import stats


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"  # 待分配
    ASSIGNED = "assigned"  # 已分配给骑手
    PICKING_UP = "picking_up"  # 骑手正在取货
    PICKED_UP = "picked_up"  # 已取货
    DELIVERING = "delivering"  # 配送中
    DELIVERED = "delivered"  # 已送达
    TIMEOUT = "timeout"  # 超时
    CANCELLED = "cancelled"  # 已取消


class CourierStatus(Enum):
    """骑手状态枚举"""
    IDLE = "idle"  # 空闲
    MOVING_TO_PICKUP = "moving_to_pickup"  # 前往取货点
    WAITING_FOR_FOOD = "waiting_for_food"  # 等待商家出餐
    PICKING_UP = "picking_up"  # 取货中
    MOVING_TO_DELIVERY = "moving_to_delivery"  # 前往配送点
    DELIVERING = "delivering"  # 配送中（交付订单）
    OFFLINE = "offline"  # 离线


class MerchantStatus(Enum):
    """商家状态枚举"""
    OPEN = "open"  # 营业中
    BUSY = "busy"  # 繁忙（队列较长）
    OVERLOADED = "overloaded"  # 超负荷
    CLOSED = "closed"  # 已打烊


@dataclass
class Merchant:
    """
    商家类 - 实现备餐不确定性建模
    
    特性：
    1. Gamma分布备餐时间：捕捉备餐时间的随机性和偏态分布
    2. M/M/c排队模型：模拟商家内部订单处理队列
    3. 动态产能：根据时段调整服务能力
    """
    merchant_id: int
    node_id: int  # 商家位置节点ID
    coords: Tuple[float, float]  # 商家坐标
    name: str = ""  # 商家名称
    
    # 备餐时间分布参数 (Gamma分布)
    # E[T] = alpha/beta, Var[T] = alpha/beta^2
    prep_time_alpha: float = 2.0  # 形状参数（控制分布偏态）
    prep_time_beta: float = 0.02  # 速率参数（1/scale）
    # 默认: E[T] = 2/0.02 = 100秒, Std = sqrt(2)/0.02 = 70.7秒
    
    # 服务能力参数 (M/M/c队列)
    service_rate: float = 0.01  # μ: 单位时间产能（订单/秒），默认6单/分钟
    num_servers: int = 2  # c: 并行制作能力（厨师/出餐口数量）
    
    # 运行时状态
    status: MerchantStatus = field(default=MerchantStatus.OPEN)
    current_queue: List[int] = field(default_factory=list)  # 当前排队订单ID列表
    orders_in_preparation: List[int] = field(default_factory=list)  # 正在制作的订单
    
    # 统计信息
    total_orders_received: int = 0  # 累计接单数
    total_orders_completed: int = 0  # 累计完成数
    total_prep_time: float = 0.0  # 累计备餐时间
    total_wait_time_caused: float = 0.0  # 累计造成的骑手等餐时间
    historical_avg_delay: float = 0.0  # 历史平均延迟
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            self.name = f"Merchant_{self.merchant_id}"
    
    def generate_prep_time(self) -> float:
        """
        生成随机备餐时间（Gamma分布）
        
        Returns:
            备餐时间（秒）
        """
        # 使用scipy的gamma分布
        # scipy的gamma参数: shape=alpha, scale=1/beta
        prep_time = stats.gamma.rvs(a=self.prep_time_alpha, scale=1.0/self.prep_time_beta)
        return max(prep_time, 30.0)  # 最少30秒
    
    def get_queue_length(self) -> int:
        """获取当前队列长度"""
        return len(self.current_queue) + len(self.orders_in_preparation)
    
    def get_utilization(self) -> float:
        """
        获取商家利用率 ρ = λ / (c * μ)
        
        Returns:
            利用率 (0-1+，可能超过1表示超负荷)
        """
        # 基于当前队列估算到达率
        if self.total_orders_received == 0:
            return 0.0
        
        # 简化计算：基于队列长度
        queue_len = self.get_queue_length()
        capacity = self.num_servers * self.service_rate
        
        if capacity <= 0:
            return 1.0
        
        return min(queue_len * self.service_rate / capacity, 2.0)
    
    def estimate_wait_time(self, current_time: float = 0) -> float:
        """
        预估新订单的等待时间（基于M/M/c队列模型）
        
        Args:
            current_time: 当前时间（用于动态调整）
        
        Returns:
            预估等待时间（秒）
        """
        queue_len = self.get_queue_length()
        
        if queue_len == 0:
            return 0.0
        
        # 简化的等待时间估算
        # W_q ≈ queue_length / (c * μ)
        capacity = self.num_servers * self.service_rate
        if capacity <= 0:
            return float('inf')
        
        base_wait = queue_len / capacity
        
        # 加上当前制作中订单的平均剩余时间
        avg_prep = self.prep_time_alpha / self.prep_time_beta
        
        return base_wait + avg_prep / 2
    
    def estimate_ready_time(self, order_arrival_time: float) -> float:
        """
        预估订单出餐时间
        
        Args:
            order_arrival_time: 订单到达时间
        
        Returns:
            预估出餐时间
        """
        wait_time = self.estimate_wait_time(order_arrival_time)
        prep_time = self.prep_time_alpha / self.prep_time_beta  # 期望备餐时间
        return order_arrival_time + wait_time + prep_time
    
    def add_order_to_queue(self, order_id: int) -> None:
        """添加订单到队列"""
        self.current_queue.append(order_id)
        self.total_orders_received += 1
        self._update_status()
    
    def start_preparation(self, order_id: int) -> float:
        """
        开始制作订单
        
        Args:
            order_id: 订单ID
        
        Returns:
            本次备餐时间（秒）
        """
        if order_id in self.current_queue:
            self.current_queue.remove(order_id)
        
        self.orders_in_preparation.append(order_id)
        prep_time = self.generate_prep_time()
        self._update_status()
        
        return prep_time
    
    def complete_preparation(self, order_id: int, prep_time: float) -> None:
        """
        完成订单制作
        
        Args:
            order_id: 订单ID
            prep_time: 实际备餐时间
        """
        if order_id in self.orders_in_preparation:
            self.orders_in_preparation.remove(order_id)
        
        self.total_orders_completed += 1
        self.total_prep_time += prep_time
        
        # 更新历史平均延迟
        if self.total_orders_completed > 0:
            self.historical_avg_delay = self.total_prep_time / self.total_orders_completed
        
        self._update_status()
    
    def record_courier_wait(self, wait_time: float) -> None:
        """记录骑手等餐时间"""
        self.total_wait_time_caused += wait_time
    
    def _update_status(self) -> None:
        """更新商家状态"""
        utilization = self.get_utilization()
        queue_len = self.get_queue_length()
        
        if utilization >= 1.0 or queue_len > self.num_servers * 3:
            self.status = MerchantStatus.OVERLOADED
        elif utilization >= 0.7 or queue_len > self.num_servers * 2:
            self.status = MerchantStatus.BUSY
        else:
            self.status = MerchantStatus.OPEN
    
    def get_state_features(self) -> Dict[str, float]:
        """
        获取商家状态特征（用于RL状态编码）
        
        Returns:
            特征字典
        """
        return {
            'queue_length': self.get_queue_length(),
            'utilization': self.get_utilization(),
            'estimated_wait_time': self.estimate_wait_time(),
            'historical_avg_delay': self.historical_avg_delay,
            'is_overloaded': 1.0 if self.status == MerchantStatus.OVERLOADED else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'merchant_id': self.merchant_id,
            'node_id': self.node_id,
            'coords': self.coords,
            'name': self.name,
            'status': self.status.value,
            'queue_length': self.get_queue_length(),
            'utilization': self.get_utilization(),
            'total_orders_received': self.total_orders_received,
            'total_orders_completed': self.total_orders_completed,
            'historical_avg_delay': self.historical_avg_delay,
            'total_wait_time_caused': self.total_wait_time_caused
        }


@dataclass
class Order:
    """
    订单类 - 扩展商家备餐不确定性支持
    
    新增特性：
    1. 商家关联：通过merchant_id关联到Merchant对象
    2. 备餐不确定性：记录预估和实际备餐时间
    3. 等餐时间追踪：记录骑手在商家的等待时间
    """
    order_id: int
    arrival_time: float  # 订单到达时间（秒）
    merchant_node: int  # 商家节点ID
    customer_node: int  # 客户节点ID
    merchant_coords: Tuple[float, float]  # 商家坐标
    customer_coords: Tuple[float, float]  # 客户坐标
    preparation_time: float  # 餐品准备时间（秒）- 初始预估值
    delivery_window: float  # 配送时间窗（秒）
    
    # 时间约束
    earliest_pickup_time: float  # 最早取货时间
    latest_delivery_time: float  # 最晚送达时间
    
    # === 商家备餐不确定性相关字段 ===
    merchant_id: Optional[int] = None  # 关联的商家ID
    
    # 备餐时间分布参数（可从商家继承或订单特定）
    prep_time_distribution: str = "gamma"  # 分布类型: gamma, lognormal, fixed
    prep_time_alpha: float = 2.0  # Gamma分布形状参数
    prep_time_beta: float = 0.02  # Gamma分布速率参数
    
    # 备餐状态追踪
    merchant_queue_position: int = 0  # 下单时在商家队列中的位置
    estimated_ready_time: Optional[float] = None  # 预估出餐时间
    actual_ready_time: Optional[float] = None  # 实际出餐时间
    food_ready: bool = False  # 餐品是否已准备好
    
    # 骑手等餐时间追踪
    courier_arrival_at_merchant: Optional[float] = None  # 骑手到达商家时间
    waiting_time_at_merchant: float = 0.0  # 骑手在商家等餐时间（秒）
    
    # 运行时状态
    status: OrderStatus = field(default=OrderStatus.PENDING)
    assigned_courier_id: Optional[int] = None  # 分配的骑手ID
    
    # 时间戳
    assigned_time: Optional[float] = None  # 分配时间
    pickup_start_time: Optional[float] = None  # 开始取货时间
    pickup_complete_time: Optional[float] = None  # 取货完成时间
    delivery_start_time: Optional[float] = None  # 开始配送时间
    delivery_complete_time: Optional[float] = None  # 配送完成时间
    
    # 服务时长配置
    pickup_duration: float = 120.0  # 取货耗时（秒）
    dropoff_duration: float = 120.0  # 送货耗时（秒）
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保时间窗合理
        if self.earliest_pickup_time < self.arrival_time:
            raise ValueError(f"Order {self.order_id}: earliest_pickup_time cannot be before arrival_time")
        
        if self.latest_delivery_time <= self.earliest_pickup_time:
            raise ValueError(f"Order {self.order_id}: latest_delivery_time must be after earliest_pickup_time")
    
    def assign_to_courier(self, courier_id: int, current_time: float) -> None:
        """
        分配给骑手
        
        注意：只有PENDING状态的订单可以被分配。
        已分配、已取货、已配送的订单不能被重新分配，防止状态回滚。
        """
        # 防止对已处理订单的重新分配（会导致状态回滚）
        if self.status in [OrderStatus.PICKED_UP, OrderStatus.DELIVERING, OrderStatus.DELIVERED]:
            import logging
            logging.getLogger(__name__).warning(
                f"订单{self.order_id}已在处理中(状态:{self.status})，拒绝重新分配给骑手{courier_id}"
            )
            return  # 静默拒绝，不抛出异常
        
        if self.status == OrderStatus.ASSIGNED:
            # 已分配的订单可以重新分配给其他骑手（正常场景）
            import logging
            logging.getLogger(__name__).debug(
                f"订单{self.order_id}从骑手{self.assigned_courier_id}重新分配给骑手{courier_id}"
            )
        elif self.status != OrderStatus.PENDING:
            raise ValueError(f"Order {self.order_id} cannot be assigned (status: {self.status})")
        
        self.assigned_courier_id = courier_id
        self.assigned_time = current_time
        self.status = OrderStatus.ASSIGNED
    
    def start_pickup(self, current_time: float) -> None:
        """开始取货"""
        if self.status != OrderStatus.ASSIGNED:
            raise ValueError(f"Order {self.order_id} cannot start pickup (status: {self.status})")
        
        self.pickup_start_time = current_time
        self.status = OrderStatus.PICKING_UP
    
    def complete_pickup(self, current_time: float) -> None:
        """完成取货"""
        if self.status != OrderStatus.PICKING_UP:
            raise ValueError(f"Order {self.order_id} cannot complete pickup (status: {self.status})")
        
        self.pickup_complete_time = current_time
        self.status = OrderStatus.PICKED_UP
    
    def start_delivery(self, current_time: float) -> None:
        """开始配送"""
        if self.status != OrderStatus.PICKED_UP:
            raise ValueError(f"Order {self.order_id} cannot start delivery (status: {self.status})")
        
        self.delivery_start_time = current_time
        self.status = OrderStatus.DELIVERING
    
    def complete_delivery(self, current_time: float) -> None:
        """完成配送"""
        if self.status != OrderStatus.DELIVERING:
            raise ValueError(f"Order {self.order_id} cannot complete delivery (status: {self.status})")
        
        self.delivery_complete_time = current_time
        self.status = OrderStatus.DELIVERED
    
    def is_timeout(self, current_time: float) -> bool:
        """检查是否超时"""
        if self.status == OrderStatus.DELIVERED:
            return self.delivery_complete_time > self.latest_delivery_time
        else:
            return current_time > self.latest_delivery_time
    
    def get_waiting_time(self, current_time: float) -> float:
        """获取等待时间"""
        if self.assigned_time is None:
            return current_time - self.arrival_time
        else:
            return self.assigned_time - self.arrival_time
    
    def get_total_service_time(self) -> Optional[float]:
        """获取总服务时长（从分配到送达）"""
        if self.delivery_complete_time and self.assigned_time:
            return self.delivery_complete_time - self.assigned_time
        return None
    
    def generate_actual_prep_time(self) -> float:
        """
        生成实际备餐时间（基于配置的分布）
        
        Returns:
            实际备餐时间（秒）
        """
        if self.prep_time_distribution == "fixed":
            return self.preparation_time
        elif self.prep_time_distribution == "gamma":
            # Gamma分布：shape=alpha, scale=1/beta
            prep_time = stats.gamma.rvs(a=self.prep_time_alpha, scale=1.0/self.prep_time_beta)
            return max(prep_time, 30.0)  # 最少30秒
        elif self.prep_time_distribution == "lognormal":
            # LogNormal分布
            mu = np.log(self.preparation_time) - 0.5 * (self.prep_time_alpha ** 2)
            prep_time = stats.lognorm.rvs(s=self.prep_time_alpha, scale=np.exp(mu))
            return max(prep_time, 30.0)
        else:
            return self.preparation_time
    
    def set_food_ready(self, ready_time: float) -> None:
        """
        标记餐品已准备好
        
        Args:
            ready_time: 出餐时间
        """
        self.actual_ready_time = ready_time
        self.food_ready = True
    
    def record_courier_arrival(self, arrival_time: float) -> None:
        """
        记录骑手到达商家时间
        
        Args:
            arrival_time: 到达时间
        """
        self.courier_arrival_at_merchant = arrival_time
        
        # 计算等餐时间（如果餐还没好）
        if self.food_ready and self.actual_ready_time is not None:
            # 骑手到达时餐已好，无需等待
            self.waiting_time_at_merchant = 0.0
        elif not self.food_ready:
            # 餐还没好，需要等待（实际等待时间在餐好时计算）
            pass
    
    def calculate_waiting_time(self, food_ready_time: float) -> float:
        """
        计算骑手实际等餐时间
        
        Args:
            food_ready_time: 餐品准备好的时间
        
        Returns:
            等餐时间（秒），如果骑手晚于出餐到达则为0
        """
        if self.courier_arrival_at_merchant is None:
            return 0.0
        
        # 等餐时间 = max(0, 出餐时间 - 骑手到达时间)
        wait_time = max(0.0, food_ready_time - self.courier_arrival_at_merchant)
        self.waiting_time_at_merchant = wait_time
        return wait_time
    
    def get_merchant_features(self) -> Dict[str, float]:
        """
        获取与商家相关的特征（用于RL状态编码）
        
        Returns:
            商家相关特征字典
        """
        # 归一化参数
        MAX_QUEUE = 20
        MAX_PREP_TIME = 600.0  # 10分钟
        MAX_DELAY = 300.0  # 5分钟
        
        return {
            'queue_position_normalized': min(self.merchant_queue_position / MAX_QUEUE, 1.0),
            'estimated_ready_time_normalized': min(
                (self.estimated_ready_time - self.arrival_time) / MAX_PREP_TIME, 1.0
            ) if self.estimated_ready_time else 0.5,
            'prep_time_expected': (self.prep_time_alpha / self.prep_time_beta) / MAX_PREP_TIME,
            'food_ready': 1.0 if self.food_ready else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'arrival_time': self.arrival_time,
            'merchant_node': self.merchant_node,
            'customer_node': self.customer_node,
            'merchant_id': self.merchant_id,
            'status': self.status.value,
            'assigned_courier_id': self.assigned_courier_id,
            'assigned_time': self.assigned_time,
            'pickup_complete_time': self.pickup_complete_time,
            'delivery_complete_time': self.delivery_complete_time,
            'is_timeout': self.is_timeout(self.delivery_complete_time or float('inf')),
            # 商家备餐不确定性相关
            'estimated_ready_time': self.estimated_ready_time,
            'actual_ready_time': self.actual_ready_time,
            'food_ready': self.food_ready,
            'waiting_time_at_merchant': self.waiting_time_at_merchant,
            'courier_arrival_at_merchant': self.courier_arrival_at_merchant
        }


@dataclass
class Courier:
    """
    骑手类 - 扩展接单行为概率模型
    
    新增特性：
    1. 接单概率模型：P_accept = σ(w · features)
    2. 骑手行为随机性：模拟真实骑手的拒单行为
    3. 疲劳度追踪：影响接单意愿
    """
    courier_id: int
    initial_node: int  # 初始位置节点ID
    initial_coords: Tuple[float, float]  # 初始坐标
    
    # 能力配置
    max_capacity: int = 5  # 最大同时携带订单数
    speed_kph: float = 15.0  # 速度（km/h）
    
    # === 骑手接单行为模型参数 ===
    # 接单概率模型权重: P_accept = σ(w · features)
    # features = [距离, 报酬, 当前负载, 疲劳度, 时段偏好]
    acceptance_weights: Dict[str, float] = field(default_factory=lambda: {
        'distance': -0.5,      # 距离越远，接单意愿越低
        'reward': 1.0,         # 报酬越高，接单意愿越高
        'current_load': -0.3,  # 当前负载越高，接单意愿越低
        'fatigue': -0.4,       # 疲劳度越高，接单意愿越低
        'peak_hour': 0.2,      # 高峰期接单意愿略高（收入高）
        'bias': 0.5            # 基础接单倾向
    })
    
    # 骑手个体差异参数
    base_acceptance_rate: float = 0.9  # 基础接单率（个体差异）
    fatigue_level: float = 0.0  # 当前疲劳度 (0-1)
    fatigue_recovery_rate: float = 0.001  # 疲劳恢复速率（每秒）
    fatigue_accumulation_rate: float = 0.0001  # 疲劳累积速率（每米）
    
    # 是否启用接单概率模型
    enable_acceptance_model: bool = True
    
    # 运行时状态
    status: CourierStatus = field(default=CourierStatus.IDLE)
    current_node: int = field(init=False)  # 当前节点ID
    current_coords: Tuple[float, float] = field(init=False)  # 当前坐标
    
    # 任务队列
    assigned_orders: List[int] = field(default_factory=list)  # 已分配的订单ID列表
    current_route: List[Tuple[str, int, int]] = field(default_factory=list)  # 当前路线 [(action, order_id, node_id), ...]
    # action: 'pickup' 或 'delivery'
    
    # 统计信息
    total_distance: float = 0.0  # 总行驶距离（米）
    total_time: float = 0.0  # 总工作时间（秒）
    completed_orders: int = 0  # 完成的订单数
    idle_time: float = 0.0  # 空闲时间（秒）
    rejected_orders: int = 0  # 拒绝的订单数
    last_activity_time: float = 0.0  # 上次活动时间（用于疲劳恢复）
    
    def __post_init__(self):
        """初始化后处理"""
        self.current_node = self.initial_node
        self.current_coords = self.initial_coords
        
        # 初始化个体差异（骑手之间有不同的接单倾向）
        if self.base_acceptance_rate == 0.9:
            # 添加随机个体差异 (0.7 - 1.0)
            self.base_acceptance_rate = 0.7 + np.random.random() * 0.3
    
    def is_available(self) -> bool:
        """检查是否可接新单（仅IDLE状态）"""
        return self.status == CourierStatus.IDLE and len(self.assigned_orders) < self.max_capacity
    
    def can_accept_new_order(self) -> bool:
        """检查是否可接新单（仅检查容量，不限制状态）- 用于动态插入"""
        return len(self.assigned_orders) < self.max_capacity
    
    def calculate_acceptance_probability(self, 
                                         order_distance: float,
                                         order_reward: float = 5.0,
                                         current_time: float = 0.0) -> float:
        """
        计算接单概率 P_accept = σ(w · features)
        
        实现研究大纲中的骑手接单概率模型：
        P_accept = sigmoid(w_distance * distance + w_reward * reward + 
                          w_load * load + w_fatigue * fatigue + 
                          w_peak * is_peak + bias)
        
        Args:
            order_distance: 订单距离（米）
            order_reward: 订单报酬（元）
            current_time: 当前时间（秒）
        
        Returns:
            接单概率 (0-1)
        """
        if not self.enable_acceptance_model:
            return 1.0  # 禁用模型时总是接单
        
        # 更新疲劳度（基于上次活动时间恢复）
        if current_time > self.last_activity_time:
            recovery_time = current_time - self.last_activity_time
            self.fatigue_level = max(0.0, self.fatigue_level - 
                                     self.fatigue_recovery_rate * recovery_time)
        
        # 构建特征向量
        # 距离归一化（5km为参考）
        distance_normalized = min(order_distance / 5000.0, 2.0)
        
        # 报酬归一化（10元为参考）
        reward_normalized = min(order_reward / 10.0, 2.0)
        
        # 当前负载归一化
        load_normalized = len(self.assigned_orders) / self.max_capacity
        
        # 疲劳度
        fatigue_normalized = self.fatigue_level
        
        # 判断是否高峰时段
        hour = (current_time / 3600) % 24
        is_peak = 1.0 if (11 <= hour <= 13 or 17 <= hour <= 20) else 0.0
        
        # 计算线性组合
        w = self.acceptance_weights
        z = (w['distance'] * distance_normalized +
             w['reward'] * reward_normalized +
             w['current_load'] * load_normalized +
             w['fatigue'] * fatigue_normalized +
             w['peak_hour'] * is_peak +
             w['bias'])
        
        # Sigmoid函数
        probability = 1.0 / (1.0 + np.exp(-z))
        
        # 应用基础接单率调整
        probability = probability * self.base_acceptance_rate
        
        return min(max(probability, 0.0), 1.0)
    
    def will_accept_order(self, 
                          order_distance: float,
                          order_reward: float = 5.0,
                          current_time: float = 0.0) -> bool:
        """
        判断骑手是否接受该订单（基于概率采样）
        
        Args:
            order_distance: 订单距离（米）
            order_reward: 订单报酬（元）
            current_time: 当前时间（秒）
        
        Returns:
            是否接单
        """
        # 容量检查
        if len(self.assigned_orders) >= self.max_capacity:
            return False
        
        # 计算接单概率
        probability = self.calculate_acceptance_probability(
            order_distance, order_reward, current_time
        )
        
        # 随机采样决定是否接单
        accept = np.random.random() < probability
        
        if not accept:
            self.rejected_orders += 1
        
        return accept
    
    def update_fatigue(self, distance_traveled: float, current_time: float) -> None:
        """
        更新骑手疲劳度
        
        Args:
            distance_traveled: 本次行驶距离（米）
            current_time: 当前时间
        """
        # 累积疲劳
        self.fatigue_level += self.fatigue_accumulation_rate * distance_traveled
        self.fatigue_level = min(self.fatigue_level, 1.0)  # 上限1.0
        
        # 更新活动时间
        self.last_activity_time = current_time
    
    def get_acceptance_features(self) -> Dict[str, float]:
        """
        获取接单相关特征（用于分析和可视化）
        
        Returns:
            特征字典
        """
        return {
            'base_acceptance_rate': self.base_acceptance_rate,
            'current_load': len(self.assigned_orders) / self.max_capacity,
            'fatigue_level': self.fatigue_level,
            'rejected_orders': self.rejected_orders,
            'acceptance_ratio': (self.completed_orders / 
                                max(self.completed_orders + self.rejected_orders, 1))
        }
    
    def get_current_route_info(self) -> Dict[str, Any]:
        """获取当前路线信息（用于VRP建模）"""
        return {
            'current_node': self.current_node,
            'current_route': self.current_route.copy(),
            'assigned_orders': self.assigned_orders.copy(),
            'num_tasks_remaining': len(self.current_route),
            'is_idle': self.status == CourierStatus.IDLE
        }
    
    def get_estimated_finish_time(self, env_time: float, env_ref) -> float:
        """估算当前路线的完成时间（粗略估计）
        
        Args:
            env_time: 当前仿真时间
            env_ref: 仿真环境引用（用于查询距离/时间）
        
        Returns:
            预估完成时间（秒）
        """
        if len(self.current_route) == 0:
            return env_time
        
        # 简化估计：基于剩余任务数和平均服务时间
        # 更精确的实现需要计算每段路径的实际时间
        estimated_time = env_time
        current_pos = self.current_node
        
        try:
            for action, order_id, target_node in self.current_route:
                # 行程时间
                if current_pos != target_node:
                    travel_time = env_ref.get_travel_time(current_pos, target_node, self.speed_kph)
                    estimated_time += travel_time
                
                # 服务时间
                service_time = 120.0  # 默认2分钟
                estimated_time += service_time
                
                current_pos = target_node
        except Exception:
            # 如果查询失败，返回简单估计
            estimated_time = env_time + len(self.current_route) * 300.0
        
        return estimated_time
    
    def assign_order(self, order_id: int) -> None:
        """分配订单"""
        if len(self.assigned_orders) >= self.max_capacity:
            raise ValueError(f"Courier {self.courier_id} is at full capacity")
        
        self.assigned_orders.append(order_id)
    
    def remove_order(self, order_id: int) -> None:
        """移除订单（完成后）"""
        if order_id in self.assigned_orders:
            self.assigned_orders.remove(order_id)
            self.completed_orders += 1
        else:
            # 订单可能已被移除（重复配送检测），记录警告但不抛出异常
            import logging
            logging.getLogger(__name__).warning(
                f"Order {order_id} not in courier {self.courier_id}'s assigned orders, may have been removed already"
            )
    
    def update_position(self, node_id: int, coords: Tuple[float, float]) -> None:
        """更新位置"""
        self.current_node = node_id
        self.current_coords = coords
    
    def add_distance(self, distance: float) -> None:
        """累加行驶距离"""
        self.total_distance += distance
    
    def add_time(self, duration: float) -> None:
        """累加工作时间"""
        self.total_time += duration
    
    def add_idle_time(self, duration: float) -> None:
        """累加空闲时间"""
        self.idle_time += duration
    
    def get_utilization(self) -> float:
        """获取利用率"""
        if self.total_time == 0:
            return 0.0
        return (self.total_time - self.idle_time) / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'courier_id': self.courier_id,
            'status': self.status.value,
            'current_node': self.current_node,
            'current_coords': self.current_coords,
            'assigned_orders': self.assigned_orders,
            'completed_orders': self.completed_orders,
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'idle_time': self.idle_time,
            'utilization': self.get_utilization(),
            # 骑手接单行为统计
            'rejected_orders': self.rejected_orders,
            'fatigue_level': self.fatigue_level,
            'base_acceptance_rate': self.base_acceptance_rate,
            'acceptance_ratio': (self.completed_orders / 
                                max(self.completed_orders + self.rejected_orders, 1))
        }


@dataclass
class SimulationEvent:
    """仿真事件记录"""
    timestamp: float
    event_type: str  # 'order_arrival', 'order_assigned', 'pickup_complete', 'delivery_complete'
    entity_id: int  # 订单ID或骑手ID
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'entity_id': self.entity_id,
            'details': self.details
        }

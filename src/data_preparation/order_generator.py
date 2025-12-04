"""
订单生成器模块
基于泊松过程和空间分布生成模拟订单流
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """订单数据类"""
    order_id: int
    arrival_time: float  # 到达时间（秒）
    merchant_node: int  # 商家节点ID
    customer_node: int  # 客户节点ID
    merchant_coords: Tuple[float, float]  # 商家坐标(x, y)
    customer_coords: Tuple[float, float]  # 客户坐标(x, y)
    preparation_time: float  # 准备时间（秒）
    delivery_window: float  # 配送时间窗（秒）
    earliest_pickup_time: float  # 最早取货时间
    latest_delivery_time: float  # 最晚送达时间
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class OrderGenerator:
    """订单生成器类"""
    
    def __init__(self, graph: nx.MultiDiGraph, config: Dict[str, Any], 
                 node_list: Optional[List] = None, random_seed: Optional[int] = None):
        """
        初始化订单生成器
        
        Args:
            graph: NetworkX图对象
            config: 订单生成配置字典
            node_list: 可用节点列表（如果为None则使用所有节点）
            random_seed: 随机种子
        """
        self.graph = graph
        self.config = config
        self.node_list = node_list if node_list else list(graph.nodes())
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"设置随机种子: {random_seed}")
        
        # 提取配置
        self.total_orders = config.get('total_orders', 1000)
        self.simulation_duration = config.get('simulation_duration', 86400)
        
        # 到达过程配置
        arrival_config = config.get('arrival_process', {})
        self.arrival_type = arrival_config.get('type', 'poisson')
        self.arrival_rate = arrival_config.get('rate', 0.0116)
        self.peak_hours = arrival_config.get('peak_hours', [[11, 13], [18, 20]])
        self.peak_multiplier = arrival_config.get('peak_multiplier', 3.0)
        
        # 空间分布配置
        spatial_config = config.get('spatial_distribution', {})
        self.merchant_config = spatial_config.get('merchant', {})
        self.customer_config = spatial_config.get('customer', {})
        
        # 服务时间配置
        service_config = config.get('service_time', {})
        self.preparation_time_range = service_config.get('preparation_time', [300, 900])
        self.delivery_window_range = service_config.get('delivery_window', [1800, 3600])
        self.pickup_duration = service_config.get('pickup_duration', 120)
        self.dropoff_duration = service_config.get('dropoff_duration', 120)
        
        # 初始化位置
        self.merchant_locations = None
        self.customer_locations = None
        self.node_kdtree = None
        
        self._initialize_locations()
    
    def _initialize_locations(self) -> None:
        """初始化商家和客户位置"""
        logger.info("初始化空间分布...")
        
        # 获取所有节点的坐标
        node_coords = []
        for node in self.node_list:
            data = self.graph.nodes[node]
            x, y = data.get('x', 0), data.get('y', 0)
            node_coords.append([x, y])
        
        node_coords = np.array(node_coords)
        
        # 构建KD树用于快速查找最近节点
        self.node_kdtree = KDTree(node_coords)
        
        # 计算坐标范围
        x_min, y_min = node_coords.min(axis=0)
        x_max, y_max = node_coords.max(axis=0)
        x_center, y_center = node_coords.mean(axis=0)
        
        # 生成商家位置
        merchant_type = self.merchant_config.get('type', 'clustered')
        num_clusters = self.merchant_config.get('num_clusters', 5)
        cluster_std = self.merchant_config.get('cluster_std', 300)
        
        if merchant_type == 'clustered':
            self.merchant_locations = self._generate_clustered_locations(
                num_clusters, 
                cluster_std / 111000,  # 转换为度（近似）
                x_center, y_center,
                x_min, x_max, y_min, y_max,
                int(self.total_orders * 0.3)  # 商家数量约为订单数的30%
            )
        elif merchant_type == 'uniform':
            self.merchant_locations = self._generate_uniform_locations(
                int(self.total_orders * 0.3),
                x_min, x_max, y_min, y_max
            )
        elif merchant_type == 'kde':
            # 新增：基于KDE的分布生成（从真实数据或历史订单学习）
            data_source = self.merchant_config.get('data_source', None)
            self.merchant_locations = self._generate_kde_locations(
                data_source,
                int(self.total_orders * 0.3),
                x_min, x_max, y_min, y_max,
                location_type='merchant'
            )
        else:
            logger.warning(f"未知的商家分布类型: {merchant_type}，使用聚类分布")
            self.merchant_locations = self._generate_clustered_locations(
                num_clusters, cluster_std / 111000,
                x_center, y_center, x_min, x_max, y_min, y_max,
                int(self.total_orders * 0.3)
            )
        
        # 生成客户位置（通常更分散）
        customer_type = self.customer_config.get('type', 'uniform')
        coverage = self.customer_config.get('coverage', 0.9)
        
        if customer_type == 'uniform':
            # 缩小范围以提高覆盖率
            margin = (1 - coverage) / 2
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            self.customer_locations = self._generate_uniform_locations(
                self.total_orders,
                x_min + margin * x_range,
                x_max - margin * x_range,
                y_min + margin * y_range,
                y_max - margin * y_range
            )
        elif customer_type == 'clustered':
            self.customer_locations = self._generate_clustered_locations(
                num_clusters * 2,  # 客户聚类更多
                cluster_std * 1.5 / 111000,
                x_center, y_center, x_min, x_max, y_min, y_max,
                self.total_orders
            )
        elif customer_type == 'kde':
            # 新增：基于KDE的分布生成
            data_source = self.customer_config.get('data_source', None)
            self.customer_locations = self._generate_kde_locations(
                data_source,
                self.total_orders,
                x_min, x_max, y_min, y_max,
                location_type='customer'
            )
        else:
            self.customer_locations = self._generate_uniform_locations(
                self.total_orders, x_min, x_max, y_min, y_max
            )
        
        logger.info(f"生成了 {len(self.merchant_locations)} 个商家位置")
        logger.info(f"生成了 {len(self.customer_locations)} 个客户位置")
    
    def _generate_clustered_locations(self, num_clusters: int, cluster_std: float,
                                     x_center: float, y_center: float,
                                     x_min: float, x_max: float,
                                     y_min: float, y_max: float,
                                     num_points: int) -> np.ndarray:
        """
        生成聚类分布的位置
        
        Args:
            num_clusters: 聚类数量
            cluster_std: 聚类标准差
            x_center, y_center: 中心坐标
            x_min, x_max, y_min, y_max: 坐标范围
            num_points: 点数量
        
        Returns:
            位置数组 (n, 2)
        """
        # 生成聚类中心
        cluster_centers = []
        for _ in range(num_clusters):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            cluster_centers.append([x, y])
        
        cluster_centers = np.array(cluster_centers)
        
        # 为每个点分配聚类
        cluster_assignments = np.random.choice(num_clusters, size=num_points)
        
        # 生成点
        locations = []
        for i in range(num_points):
            cluster_idx = cluster_assignments[i]
            center = cluster_centers[cluster_idx]
            
            # 在聚类中心周围生成点
            x = np.random.normal(center[0], cluster_std)
            y = np.random.normal(center[1], cluster_std)
            
            # 确保在范围内
            x = np.clip(x, x_min, x_max)
            y = np.clip(y, y_min, y_max)
            
            locations.append([x, y])
        
        return np.array(locations)
    
    def _generate_kde_locations(self, data_source: Optional[str],
                               num_points: int,
                               x_min: float, x_max: float,
                               y_min: float, y_max: float,
                               location_type: str = 'merchant') -> np.ndarray:
        """
        使用核密度估计(KDE)生成位置（基于真实数据学习分布）
        
        Args:
            data_source: 数据源文件路径（CSV格式，需包含lat/lon或x/y列）
                       如果为None，则使用合成数据模拟Olist分布
            num_points: 点数量
            x_min, x_max, y_min, y_max: 坐标范围
            location_type: 位置类型（'merchant'或'customer'）
        
        Returns:
            位置数组 (n, 2)
        """
        logger.info(f"使用KDE生成{location_type}位置分布...")
        
        # 加载或生成训练数据
        if data_source and Path(data_source).exists():
            logger.info(f"从真实数据加载: {data_source}")
            training_data = self._load_training_data(data_source)
        else:
            logger.info("数据源不存在，使用模拟Olist分布的合成数据")
            training_data = self._generate_synthetic_olist_data(
                x_min, x_max, y_min, y_max, location_type
            )
        
        if training_data is None or len(training_data) < 10:
            logger.warning("训练数据不足，回退到聚类分布")
            return self._generate_clustered_locations(
                5, 300/111000, 
                (x_min+x_max)/2, (y_min+y_max)/2,
                x_min, x_max, y_min, y_max, num_points
            )
        
        # 拟合KDE（使用Gaussian核）
        logger.info(f"拟合KDE模型（训练样本数: {len(training_data)}）...")
        
        try:
            # 转置为(2, n)格式以适配gaussian_kde
            kde = gaussian_kde(training_data.T, bw_method='scott')
            
            # 从KDE采样
            logger.info(f"从KDE分布采样 {num_points} 个点...")
            samples = kde.resample(num_points).T
            
            # 边界裁剪（确保在有效范围内）
            samples[:, 0] = np.clip(samples[:, 0], x_min, x_max)
            samples[:, 1] = np.clip(samples[:, 1], y_min, y_max)
            
            logger.info(f"KDE采样完成，生成 {len(samples)} 个{location_type}位置")
            return samples
            
        except Exception as e:
            logger.error(f"KDE拟合失败: {str(e)}，回退到聚类分布")
            return self._generate_clustered_locations(
                5, 300/111000,
                (x_min+x_max)/2, (y_min+y_max)/2,
                x_min, x_max, y_min, y_max, num_points
            )
    
    def _load_training_data(self, data_file: str) -> Optional[np.ndarray]:
        """
        从CSV文件加载训练数据
        
        支持的格式：
        - Olist格式：geolocation_lat, geolocation_lng
        - 通用格式：lat, lon 或 latitude, longitude
        - 笛卡尔坐标：x, y
        
        Args:
            data_file: CSV文件路径
        
        Returns:
            坐标数组 (n, 2) 或 None
        """
        try:
            df = pd.read_csv(data_file)
            
            # 尝试不同的列名组合
            coord_cols = [
                ('geolocation_lat', 'geolocation_lng'),  # Olist格式
                ('lat', 'lon'),
                ('latitude', 'longitude'),
                ('y', 'x'),  # 笛卡尔坐标（注意顺序）
                ('x', 'y')
            ]
            
            for lat_col, lon_col in coord_cols:
                if lat_col in df.columns and lon_col in df.columns:
                    coords = df[[lat_col, lon_col]].dropna().values
                    logger.info(f"成功加载 {len(coords)} 个坐标点（列: {lat_col}, {lon_col}）")
                    return coords
            
            logger.warning(f"文件 {data_file} 中未找到有效的坐标列")
            return None
            
        except Exception as e:
            logger.error(f"加载训练数据失败: {str(e)}")
            return None
    
    def _generate_synthetic_olist_data(self, x_min: float, x_max: float,
                                      y_min: float, y_max: float,
                                      location_type: str) -> np.ndarray:
        """
        生成模拟Olist分布特征的合成训练数据
        
        Olist数据特点（基于巴西电商数据集）：
        - 商家：高度聚集在商业中心，呈现3-5个主要聚类
        - 客户：相对分散但仍有聚集趋势，郊区密度较低
        
        Args:
            x_min, x_max, y_min, y_max: 坐标范围
            location_type: 'merchant' 或 'customer'
        
        Returns:
            合成坐标数组 (n, 2)
        """
        logger.info(f"生成模拟Olist分布的合成{location_type}数据...")
        
        if location_type == 'merchant':
            # 商家：3-5个强聚类中心
            num_centers = np.random.randint(3, 6)
            cluster_std = (x_max - x_min) * 0.03  # 约3%的区域范围
            samples_per_cluster = 200 // num_centers
            
            # 生成聚类中心（偏向中心区域）
            centers_x = np.random.normal(
                (x_min + x_max) / 2, 
                (x_max - x_min) * 0.15,
                num_centers
            )
            centers_y = np.random.normal(
                (y_min + y_max) / 2,
                (y_max - y_min) * 0.15,
                num_centers
            )
            
            # 在每个中心周围生成样本
            samples = []
            for cx, cy in zip(centers_x, centers_y):
                cluster_samples = np.random.normal(
                    [cx, cy],
                    [cluster_std, cluster_std],
                    (samples_per_cluster, 2)
                )
                samples.append(cluster_samples)
            
            training_data = np.vstack(samples)
            
        else:  # customer
            # 客户：多聚类 + 均匀背景噪声
            num_centers = np.random.randint(8, 12)
            cluster_std = (x_max - x_min) * 0.05
            
            # 聚类部分（70%）
            num_clustered = 140
            samples_per_cluster = num_clustered // num_centers
            
            centers_x = np.random.uniform(x_min, x_max, num_centers)
            centers_y = np.random.uniform(y_min, y_max, num_centers)
            
            clustered_samples = []
            for cx, cy in zip(centers_x, centers_y):
                cluster_samples = np.random.normal(
                    [cx, cy],
                    [cluster_std, cluster_std],
                    (samples_per_cluster, 2)
                )
                clustered_samples.append(cluster_samples)
            
            # 均匀背景（30%）
            num_uniform = 60
            uniform_samples = np.column_stack([
                np.random.uniform(x_min, x_max, num_uniform),
                np.random.uniform(y_min, y_max, num_uniform)
            ])
            
            training_data = np.vstack(clustered_samples + [uniform_samples])
        
        # 边界裁剪
        training_data[:, 0] = np.clip(training_data[:, 0], x_min, x_max)
        training_data[:, 1] = np.clip(training_data[:, 1], y_min, y_max)
        
        logger.info(f"生成 {len(training_data)} 个合成训练样本")
        return training_data
    
    def _generate_uniform_locations(self, num_points: int,
                                   x_min: float, x_max: float,
                                   y_min: float, y_max: float) -> np.ndarray:
        """
        生成均匀分布的位置
        
        Args:
            num_points: 点数量
            x_min, x_max, y_min, y_max: 坐标范围
        
        Returns:
            位置数组 (n, 2)
        """
        x = np.random.uniform(x_min, x_max, size=num_points)
        y = np.random.uniform(y_min, y_max, size=num_points)
        return np.column_stack([x, y])
    
    def _coords_to_node(self, coords: np.ndarray) -> int:
        """
        将坐标映射到最近的路网节点
        
        Args:
            coords: 坐标 [x, y]
        
        Returns:
            节点ID
        """
        _, idx = self.node_kdtree.query(coords)
        return self.node_list[idx]
    
    def generate_arrival_times(self) -> np.ndarray:
        """
        生成订单到达时间
        
        Returns:
            到达时间数组（秒）
        """
        logger.info(f"生成 {self.total_orders} 个订单的到达时间...")
        
        if self.arrival_type == 'poisson':
            # 齐次泊松过程
            inter_arrival_times = np.random.exponential(
                1 / self.arrival_rate, 
                size=self.total_orders
            )
            arrival_times = np.cumsum(inter_arrival_times)
            
            # 确保在仿真时长内
            arrival_times = arrival_times[arrival_times < self.simulation_duration]
            
            logger.info(f"齐次泊松过程，平均到达率: {self.arrival_rate:.4f} 订单/秒")
        
        elif self.arrival_type == 'non_homogeneous':
            # 非齐次泊松过程
            arrival_times = self._generate_non_homogeneous_poisson()
            logger.info(f"非齐次泊松过程，高峰时段: {self.peak_hours}")
        
        else:
            logger.warning(f"未知的到达过程类型: {self.arrival_type}，使用齐次泊松")
            inter_arrival_times = np.random.exponential(
                1 / self.arrival_rate, 
                size=self.total_orders
            )
            arrival_times = np.cumsum(inter_arrival_times)
        
        # 排序
        arrival_times = np.sort(arrival_times)
        
        logger.info(f"实际生成 {len(arrival_times)} 个订单")
        logger.info(f"时间跨度: {arrival_times[0]:.2f}s 到 {arrival_times[-1]:.2f}s")
        
        return arrival_times
    
    def _generate_non_homogeneous_poisson(self) -> np.ndarray:
        """
        生成非齐次泊松过程的到达时间
        
        Returns:
            到达时间数组
        """
        # 时间分段（以秒为单位）
        time_slots = np.arange(0, self.simulation_duration, 3600)  # 每小时一个时段
        arrival_times = []
        
        for i in range(len(time_slots) - 1):
            slot_start = time_slots[i]
            slot_end = time_slots[i + 1]
            slot_duration = slot_end - slot_start
            
            # 判断是否为高峰时段
            hour = int(slot_start / 3600)
            is_peak = any(
                start <= hour < end 
                for start, end in self.peak_hours
            )
            
            # 计算该时段的到达率
            if is_peak:
                rate = self.arrival_rate * self.peak_multiplier
            else:
                rate = self.arrival_rate
            
            # 生成该时段的订单数
            expected_orders = rate * slot_duration
            num_orders = np.random.poisson(expected_orders)
            
            # 在该时段内均匀分布订单
            if num_orders > 0:
                times = np.random.uniform(slot_start, slot_end, size=num_orders)
                arrival_times.extend(times)
        
        return np.array(sorted(arrival_times))
    
    def generate_orders(self) -> List[Order]:
        """
        生成完整的订单列表
        
        Returns:
            订单列表
        """
        logger.info("开始生成订单...")
        
        # 生成到达时间
        arrival_times = self.generate_arrival_times()
        actual_num_orders = len(arrival_times)
        
        # 为每个订单分配商家和客户位置
        merchant_indices = np.random.choice(
            len(self.merchant_locations), 
            size=actual_num_orders
        )
        customer_indices = np.arange(actual_num_orders)  # 每个订单唯一的客户
        
        orders = []
        
        logger.info("映射坐标到路网节点...")
        for i in range(actual_num_orders):
            # 商家和客户坐标
            merchant_coords = self.merchant_locations[merchant_indices[i]]
            customer_coords = self.customer_locations[customer_indices[i]]
            
            # 映射到路网节点
            merchant_node = self._coords_to_node(merchant_coords)
            customer_node = self._coords_to_node(customer_coords)
            
            # 生成服务时间参数
            arrival_time = arrival_times[i]
            preparation_time = np.random.uniform(*self.preparation_time_range)
            delivery_window = np.random.uniform(*self.delivery_window_range)
            
            # 计算时间窗
            earliest_pickup_time = arrival_time + preparation_time
            latest_delivery_time = earliest_pickup_time + delivery_window
            
            # 创建订单对象
            order = Order(
                order_id=i + 1,
                arrival_time=arrival_time,
                merchant_node=merchant_node,
                customer_node=customer_node,
                merchant_coords=(float(merchant_coords[0]), float(merchant_coords[1])),
                customer_coords=(float(customer_coords[0]), float(customer_coords[1])),
                preparation_time=preparation_time,
                delivery_window=delivery_window,
                earliest_pickup_time=earliest_pickup_time,
                latest_delivery_time=latest_delivery_time
            )
            
            orders.append(order)
        
        logger.info(f"订单生成完成，共 {len(orders)} 个订单")
        
        return orders
    
    def save_orders(self, orders: List[Order], output_dir: Path) -> Dict[str, Path]:
        """
        保存订单数据
        
        Args:
            orders: 订单列表
            output_dir: 输出目录
        
        Returns:
            保存的文件路径字典
        """
        logger.info("保存订单数据...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 转换为DataFrame
        orders_data = [order.to_dict() for order in orders]
        orders_df = pd.DataFrame(orders_data)
        
        # 保存订单CSV
        orders_file = output_dir / "orders.csv"
        orders_df.to_csv(orders_file, index=False, encoding='utf-8')
        saved_files['orders'] = orders_file
        logger.info(f"订单文件已保存: {orders_file}")
        
        # 保存商家位置
        unique_merchants = orders_df[['merchant_node', 'merchant_coords']].drop_duplicates()
        merchants_df = pd.DataFrame({
            'merchant_id': range(1, len(unique_merchants) + 1),
            'node_id': unique_merchants['merchant_node'].values,
            'x': [coords[0] for coords in unique_merchants['merchant_coords'].values],
            'y': [coords[1] for coords in unique_merchants['merchant_coords'].values]
        })
        
        merchants_file = output_dir / "merchants.csv"
        merchants_df.to_csv(merchants_file, index=False, encoding='utf-8')
        saved_files['merchants'] = merchants_file
        logger.info(f"商家文件已保存: {merchants_file}")
        
        # 保存客户位置
        customers_df = pd.DataFrame({
            'customer_id': orders_df['order_id'],
            'node_id': orders_df['customer_node'],
            'x': [coords[0] for coords in orders_df['customer_coords'].values],
            'y': [coords[1] for coords in orders_df['customer_coords'].values]
        })
        
        customers_file = output_dir / "customers.csv"
        customers_df.to_csv(customers_file, index=False, encoding='utf-8')
        saved_files['customers'] = customers_file
        logger.info(f"客户文件已保存: {customers_file}")
        
        # 保存统计信息
        stats = {
            'total_orders': len(orders),
            'simulation_duration': self.simulation_duration,
            'arrival_type': self.arrival_type,
            'num_merchants': len(unique_merchants),
            'avg_preparation_time': orders_df['preparation_time'].mean(),
            'avg_delivery_window': orders_df['delivery_window'].mean(),
            'first_arrival': float(orders_df['arrival_time'].min()),
            'last_arrival': float(orders_df['arrival_time'].max())
        }
        
        stats_file = output_dir / "order_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        saved_files['statistics'] = stats_file
        logger.info(f"统计信息已保存: {stats_file}")
        
        logger.info("订单数据保存完成")
        return saved_files
    
    def get_statistics(self, orders: List[Order]) -> Dict[str, Any]:
        """
        获取订单统计信息
        
        Args:
            orders: 订单列表
        
        Returns:
            统计信息字典
        """
        orders_df = pd.DataFrame([order.to_dict() for order in orders])
        
        stats = {
            'total_orders': len(orders),
            'avg_preparation_time_min': orders_df['preparation_time'].mean() / 60,
            'avg_delivery_window_min': orders_df['delivery_window'].mean() / 60,
            'first_arrival_time': orders_df['arrival_time'].min(),
            'last_arrival_time': orders_df['arrival_time'].max(),
            'time_span_hours': (orders_df['arrival_time'].max() - orders_df['arrival_time'].min()) / 3600
        }
        
        logger.info("订单统计信息:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return stats


def generate_orders(graph: nx.MultiDiGraph,
                   config: Dict[str, Any],
                   node_list: Optional[List] = None,
                   output_dir: Optional[Path] = None,
                   random_seed: Optional[int] = None) -> Tuple[List[Order], Dict[str, Path]]:
    """
    生成订单（便捷函数）
    
    Args:
        graph: NetworkX图对象
        config: 配置字典
        node_list: 可用节点列表
        output_dir: 输出目录
        random_seed: 随机种子
    
    Returns:
        (订单列表, 保存的文件路径字典)
    """
    generator = OrderGenerator(graph, config, node_list, random_seed)
    
    # 生成订单
    orders = generator.generate_orders()
    
    # 获取统计信息
    generator.get_statistics(orders)
    
    # 保存订单
    saved_files = {}
    if output_dir:
        saved_files = generator.save_orders(orders, output_dir)
    
    return orders, saved_files


if __name__ == "__main__":
    # 测试订单生成
    from src.utils.config import get_config
    from src.data_preparation.osm_network import extract_osm_network
    from src.data_preparation.distance_matrix import compute_distance_matrices
    
    config = get_config()
    network_config = config.get_network_config()
    matrix_config = config.get_distance_matrix_config()
    order_config = config.get_order_generation_config()
    random_seed = config.get_random_seed()
    
    output_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    print("=== 加载路网 ===")
    graph, _ = extract_osm_network(network_config, output_dir)
    
    print("\n=== 加载距离矩阵 ===")
    _, _, mapping = compute_distance_matrices(graph, matrix_config, output_dir)
    
    print("\n=== 生成订单 ===")
    orders, files = generate_orders(
        graph,
        order_config,
        node_list=mapping['node_list'],
        output_dir=orders_dir,
        random_seed=random_seed
    )
    
    print(f"\n生成订单数: {len(orders)}")
    print(f"\n保存的文件:")
    for key, path in files.items():
        print(f"  {key}: {path}")

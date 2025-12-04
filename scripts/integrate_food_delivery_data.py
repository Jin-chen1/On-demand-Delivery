"""
Food Delivery Dataset 集成脚本
将Kaggle真实外卖配送数据转换为项目仿真格式

数据来源: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

特点：
- 保留真实订单的时间模式（到达时间分布）
- 保留真实配送时间数据
- 将印度城市GPS坐标映射到项目路网节点（上海）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from scipy.spatial import KDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_dataset(df: pd.DataFrame) -> dict:
    """分析数据集基本统计信息"""
    logger.info("=" * 60)
    logger.info("Food Delivery Dataset 分析")
    logger.info("=" * 60)
    
    stats = {
        'total_orders': len(df),
        'unique_riders': df['Delivery_person_ID'].nunique(),
        'cities': df['City'].unique().tolist(),
        'order_types': df['Type_of_order'].unique().tolist(),
        'vehicle_types': df['Type_of_vehicle'].unique().tolist(),
    }
    
    logger.info(f"总订单数: {stats['total_orders']}")
    logger.info(f"骑手数: {stats['unique_riders']}")
    logger.info(f"城市: {stats['cities']}")
    logger.info(f"订单类型: {stats['order_types']}")
    logger.info(f"车辆类型: {stats['vehicle_types']}")
    
    # 清洗配送时间
    df['delivery_time_min'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
    
    logger.info(f"\n配送时间统计:")
    logger.info(f"  最短: {df['delivery_time_min'].min():.0f} 分钟")
    logger.info(f"  最长: {df['delivery_time_min'].max():.0f} 分钟")
    logger.info(f"  平均: {df['delivery_time_min'].mean():.1f} 分钟")
    logger.info(f"  中位数: {df['delivery_time_min'].median():.1f} 分钟")
    
    stats['avg_delivery_time'] = df['delivery_time_min'].mean()
    stats['median_delivery_time'] = df['delivery_time_min'].median()
    
    # GPS坐标范围
    logger.info(f"\n地理范围:")
    logger.info(f"  餐厅经度: {df['Restaurant_longitude'].min():.4f} ~ {df['Restaurant_longitude'].max():.4f}")
    logger.info(f"  餐厅纬度: {df['Restaurant_latitude'].min():.4f} ~ {df['Restaurant_latitude'].max():.4f}")
    
    return stats


def load_road_network(processed_dir: Path):
    """
    加载项目路网节点信息
    
    Returns:
        node_list: 节点ID列表
        node_coords: 节点坐标字典 {node_id: (lng, lat)}
    """
    import networkx as nx
    
    graph_file = processed_dir / 'road_network.graphml'
    mapping_file = processed_dir / 'node_id_mapping.json'
    
    if not graph_file.exists():
        raise FileNotFoundError(f"路网文件不存在: {graph_file}")
    
    # 加载路网
    graph = nx.read_graphml(graph_file)
    logger.info(f"加载路网: {len(graph.nodes())} 节点")
    
    # 加载节点映射
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        node_list = [int(k) for k in mapping['node_to_idx'].keys()]
    else:
        node_list = [int(n) for n in graph.nodes()]
    
    # 提取节点坐标
    node_coords = {}
    for node_id in node_list:
        str_node = str(node_id)
        if str_node in graph.nodes:
            data = graph.nodes[str_node]
            node_coords[node_id] = (float(data.get('x', 0)), float(data.get('y', 0)))
    
    logger.info(f"有效节点数: {len(node_coords)}")
    return node_list, node_coords


def map_to_network_nodes(merchant_coords: np.ndarray, 
                          customer_coords: np.ndarray,
                          node_list: list,
                          node_coords: dict,
                          random_seed: int = 42) -> tuple:
    """
    将原始GPS坐标映射到路网节点
    
    策略：
    1. 随机选择商家节点
    2. 根据原始配送距离，在合适范围内选择客户节点
    3. 保持配送距离的相对大小顺序
    
    Args:
        merchant_coords: 商家坐标 [(lat, lng), ...]
        customer_coords: 客户坐标 [(lat, lng), ...]
        node_list: 路网节点列表
        node_coords: 节点坐标 {node_id: (lng, lat)}
    
    Returns:
        merchant_nodes, customer_nodes, merchant_node_coords, customer_node_coords
    """
    np.random.seed(random_seed)
    
    # 构建KD树用于最近邻查询
    coords_array = np.array([node_coords[n] for n in node_list])  # (lng, lat)
    kdtree = KDTree(coords_array)
    
    # 获取路网边界
    lngs = coords_array[:, 0]
    lats = coords_array[:, 1]
    network_center = (np.mean(lngs), np.mean(lats))
    network_range = (lngs.max() - lngs.min(), lats.max() - lats.min())
    network_radius = min(network_range) / 2
    
    logger.info(f"路网中心: ({network_center[0]:.4f}, {network_center[1]:.4f})")
    logger.info(f"路网范围: {network_range[0]:.4f} x {network_range[1]:.4f}")
    
    # 计算原始订单的配送距离（用于保持相对距离模式）
    orig_merchant_lngs = merchant_coords[:, 1]
    orig_merchant_lats = merchant_coords[:, 0]
    orig_customer_lngs = customer_coords[:, 1]
    orig_customer_lats = customer_coords[:, 0]
    
    orig_distances = np.sqrt(
        (orig_merchant_lngs - orig_customer_lngs)**2 + 
        (orig_merchant_lats - orig_customer_lats)**2
    )
    
    # 将原始距离归一化到路网尺度
    # 原始距离中位数映射为路网半径的50%
    orig_median_dist = np.median(orig_distances[orig_distances > 0])
    scale_factor = network_radius * 0.5 / max(orig_median_dist, 0.001)
    scaled_distances = orig_distances * scale_factor
    # 限制在路网范围内
    scaled_distances = np.clip(scaled_distances, network_radius * 0.1, network_radius * 0.8)
    
    # 映射商家和客户坐标
    n_orders = len(merchant_coords)
    merchant_nodes = []
    customer_nodes = []
    merchant_node_coords = []
    customer_node_coords = []
    
    # 创建一个商家节点池（模拟30个商家）
    n_merchants = min(30, len(node_list))
    merchant_pool_indices = np.random.choice(len(node_list), size=n_merchants, replace=False)
    merchant_pool = [node_list[i] for i in merchant_pool_indices]
    
    for i in range(n_orders):
        # 随机选择一个商家节点
        m_node = merchant_pool[i % n_merchants]
        m_coord = node_coords[m_node]
        merchant_nodes.append(m_node)
        merchant_node_coords.append(m_coord)
        
        # 根据配送距离选择客户节点
        target_dist = scaled_distances[i]
        
        # 随机角度
        angle = np.random.uniform(0, 2 * np.pi)
        target_lng = m_coord[0] + target_dist * np.cos(angle)
        target_lat = m_coord[1] + target_dist * np.sin(angle)
        
        # 找到最近的路网节点
        _, c_idx = kdtree.query([target_lng, target_lat])
        c_node = node_list[c_idx]
        
        # 确保客户节点与商家节点不同
        if c_node == m_node:
            # 找次近的节点
            _, indices = kdtree.query([target_lng, target_lat], k=3)
            for idx in indices:
                if node_list[idx] != m_node:
                    c_node = node_list[idx]
                    break
        
        customer_nodes.append(c_node)
        customer_node_coords.append(node_coords[c_node])
    
    return merchant_nodes, customer_nodes, merchant_node_coords, customer_node_coords


def convert_to_simulation_format(df: pd.DataFrame, 
                                  node_list: list = None,
                                  node_coords: dict = None,
                                  city_filter: str = None,
                                  n_orders: int = None,
                                  simulation_hours: float = 2.0,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    将Food Delivery数据转换为仿真格式
    
    Args:
        df: 原始数据
        city_filter: 城市过滤（可选）
        n_orders: 采样订单数（可选）
        simulation_hours: 仿真时长（小时）
        random_seed: 随机种子
    
    Returns:
        仿真格式的订单DataFrame
    """
    np.random.seed(random_seed)
    
    # 清洗数据
    df = df.copy()
    df['delivery_time_min'] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
    
    # 过滤无效数据
    df = df.dropna(subset=['Restaurant_latitude', 'Restaurant_longitude', 
                           'Delivery_location_latitude', 'Delivery_location_longitude',
                           'delivery_time_min'])
    
    # 过滤城市
    if city_filter:
        df = df[df['City'].str.contains(city_filter, case=False, na=False)]
        logger.info(f"过滤城市 '{city_filter}': {len(df)} 条记录")
    
    # 采样
    if n_orders and n_orders < len(df):
        df = df.sample(n=n_orders, random_state=random_seed)
        logger.info(f"采样 {n_orders} 条订单")
    
    # 生成仿真时间（将订单均匀分布在仿真时段内）
    simulation_duration = simulation_hours * 3600  # 秒
    n = len(df)
    
    # 使用泊松过程模拟订单到达
    inter_arrival_times = np.random.exponential(simulation_duration / n, size=n)
    arrival_times = np.cumsum(inter_arrival_times)
    # 归一化到仿真时长内
    arrival_times = arrival_times / arrival_times.max() * (simulation_duration * 0.9)
    
    # 根据实际配送时间估算准备时间和时间窗
    # 准备时间: 5-15分钟
    preparation_times = np.random.uniform(300, 900, size=n)
    
    # 配送时间窗 = 实际配送时间 * 1.2 + 准备时间（给一些buffer）
    actual_delivery_times = df['delivery_time_min'].values * 60  # 转秒
    delivery_windows = actual_delivery_times * 1.2 + preparation_times
    
    # 如果提供了路网，映射到路网节点
    if node_list is not None and node_coords is not None:
        merchant_coords_arr = df[['Restaurant_latitude', 'Restaurant_longitude']].values
        customer_coords_arr = df[['Delivery_location_latitude', 'Delivery_location_longitude']].values
        
        merchant_nodes, customer_nodes, merchant_node_coords, customer_node_coords = map_to_network_nodes(
            merchant_coords_arr, customer_coords_arr,
            node_list, node_coords, random_seed
        )
        
        # 构建仿真订单（包含路网节点）
        orders = pd.DataFrame({
            'order_id': range(1, n + 1),
            'arrival_time': arrival_times,
            'merchant_node': merchant_nodes,
            'customer_node': customer_nodes,
            'merchant_coords': [str(c) for c in merchant_node_coords],
            'customer_coords': [str(c) for c in customer_node_coords],
            'preparation_time': preparation_times,
            'delivery_window': delivery_windows,
            'earliest_pickup_time': arrival_times + preparation_times,
            'latest_delivery_time': arrival_times + delivery_windows,
            # 保留原始信息
            'actual_delivery_time_min': df['delivery_time_min'].values,
        })
    else:
        # 不映射到路网，保留原始GPS坐标
        orders = pd.DataFrame({
            'order_id': range(1, n + 1),
            'arrival_time': arrival_times,
            'merchant_lat': df['Restaurant_latitude'].values,
            'merchant_lng': df['Restaurant_longitude'].values,
            'customer_lat': df['Delivery_location_latitude'].values,
            'customer_lng': df['Delivery_location_longitude'].values,
            'preparation_time': preparation_times,
            'delivery_window': delivery_windows,
            'earliest_pickup_time': arrival_times + preparation_times,
            'latest_delivery_time': arrival_times + delivery_windows,
            # 保留原始信息
            'actual_delivery_time_min': df['delivery_time_min'].values,
            'order_type': df['Type_of_order'].values,
            'traffic_density': df['Road_traffic_density'].values,
            'weather': df['Weatherconditions'].values,
            'city': df['City'].values,
        })
    
    # 按到达时间排序
    orders = orders.sort_values('arrival_time').reset_index(drop=True)
    orders['order_id'] = range(1, len(orders) + 1)
    
    return orders


def save_orders(orders: pd.DataFrame, output_path: Path) -> None:
    """保存订单到CSV"""
    # 根据是否有路网节点选择列
    if 'merchant_node' in orders.columns:
        core_columns = [
            'order_id', 'arrival_time', 
            'merchant_node', 'customer_node',
            'merchant_coords', 'customer_coords',
            'preparation_time', 'delivery_window',
            'earliest_pickup_time', 'latest_delivery_time'
        ]
    else:
        core_columns = [
            'order_id', 'arrival_time', 
            'merchant_lat', 'merchant_lng',
            'customer_lat', 'customer_lng',
            'preparation_time', 'delivery_window',
            'earliest_pickup_time', 'latest_delivery_time'
        ]
    
    orders[core_columns].to_csv(output_path, index=False)
    logger.info(f"订单保存到: {output_path}")


def main():
    """主函数"""
    # 路径配置
    raw_dir = Path('data/raw')
    orders_dir = Path('data/orders')
    orders_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    train_file = raw_dir / 'train.csv'
    if not train_file.exists():
        logger.error(f"找不到数据文件: {train_file}")
        return
    
    logger.info(f"加载数据: {train_file}")
    df = pd.read_csv(train_file)
    
    # 分析数据集
    stats = analyze_dataset(df)
    
    # 直接使用原始GPS坐标（不映射到路网）
    logger.info("将使用原始GPS坐标（印度城市真实位置）")
    
    # 生成不同规模的仿真订单文件
    configs = [
        {'name': 'real_delivery_100', 'n_orders': 100, 'hours': 1.0},
        {'name': 'real_delivery_500', 'n_orders': 500, 'hours': 2.0},
        {'name': 'real_delivery_1000', 'n_orders': 1000, 'hours': 4.0},
    ]
    
    for config in configs:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"生成 {config['name']} 订单集...")
        
        orders = convert_to_simulation_format(
            df,
            node_list=None,  # 不使用路网映射
            node_coords=None,
            n_orders=config['n_orders'],
            simulation_hours=config['hours'],
            random_seed=42
        )
        
        output_file = orders_dir / f"{config['name']}.csv"
        save_orders(orders, output_file)
        
        logger.info(f"  订单数: {len(orders)}")
        logger.info(f"  仿真时长: {config['hours']} 小时")
        logger.info(f"  时间范围: {orders['arrival_time'].min():.0f}s - {orders['arrival_time'].max():.0f}s")
        
        # 统计城市分布
        if 'city' in orders.columns:
            city_counts = orders['city'].value_counts()
            logger.info(f"  城市分布: {dict(city_counts)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("真实数据集成完成!")
    logger.info(f"数据来源: Kaggle Food Delivery Dataset (45,593 真实订单)")
    logger.info(f"输出目录: {orders_dir}")
    logger.info("使用原始GPS坐标（印度城市）")
    logger.info("=" * 60)
    
    return orders_dir


if __name__ == "__main__":
    main()

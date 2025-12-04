"""
Olist数据集成脚本
从真实巴西电商数据中提取地理位置，生成仿真订单
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_olist_geolocation(raw_dir: Path) -> pd.DataFrame:
    """加载Olist地理位置数据"""
    geo_file = raw_dir / "olist_geolocation_dataset.csv"
    
    if not geo_file.exists():
        raise FileNotFoundError(f"Olist地理位置文件不存在: {geo_file}")
    
    df = pd.read_csv(geo_file)
    logger.info(f"加载 {len(df)} 条地理位置记录")
    
    return df


def filter_sao_paulo_region(df: pd.DataFrame, 
                            center_lat: float = -23.55,
                            center_lng: float = -46.63,
                            radius_km: float = 15.0) -> pd.DataFrame:
    """
    筛选圣保罗市区域的数据点
    
    Args:
        df: 原始地理位置数据
        center_lat: 中心纬度（圣保罗市中心）
        center_lng: 中心经度
        radius_km: 半径（公里）
    
    Returns:
        筛选后的数据
    """
    # 简单的距离过滤（使用度数近似）
    # 1度纬度 ≈ 111km, 1度经度 ≈ 111km * cos(lat)
    lat_range = radius_km / 111.0
    lng_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    filtered = df[
        (df['geolocation_lat'] >= center_lat - lat_range) &
        (df['geolocation_lat'] <= center_lat + lat_range) &
        (df['geolocation_lng'] >= center_lng - lng_range) &
        (df['geolocation_lng'] <= center_lng + lng_range)
    ].copy()
    
    # 去除异常值
    filtered = filtered[
        (filtered['geolocation_lat'] > -90) & 
        (filtered['geolocation_lat'] < 90) &
        (filtered['geolocation_lng'] > -180) & 
        (filtered['geolocation_lng'] < 180)
    ]
    
    logger.info(f"筛选圣保罗区域数据: {len(filtered)} 条 (半径 {radius_km}km)")
    
    return filtered


def sample_locations(df: pd.DataFrame, 
                    n_merchants: int = 50,
                    n_customers: int = 200,
                    random_seed: int = 42) -> tuple:
    """
    从筛选后的数据中采样商家和客户位置
    
    Returns:
        (merchants_df, customers_df)
    """
    np.random.seed(random_seed)
    
    # 按邮编聚合，每个邮编取一个代表位置
    unique_locations = df.groupby('geolocation_zip_code_prefix').agg({
        'geolocation_lat': 'mean',
        'geolocation_lng': 'mean'
    }).reset_index()
    
    logger.info(f"唯一位置点: {len(unique_locations)}")
    
    # 随机采样商家位置（商家倾向于聚集在商业区）
    n_merchants = min(n_merchants, len(unique_locations))
    merchant_indices = np.random.choice(len(unique_locations), n_merchants, replace=False)
    merchants = unique_locations.iloc[merchant_indices].copy()
    merchants['merchant_id'] = range(1, n_merchants + 1)
    merchants = merchants.rename(columns={
        'geolocation_lat': 'lat',
        'geolocation_lng': 'lng'
    })
    
    # 随机采样客户位置（客户分布更广泛）
    n_customers = min(n_customers, len(unique_locations))
    customer_indices = np.random.choice(len(unique_locations), n_customers, replace=False)
    customers = unique_locations.iloc[customer_indices].copy()
    customers['customer_id'] = range(1, n_customers + 1)
    customers = customers.rename(columns={
        'geolocation_lat': 'lat',
        'geolocation_lng': 'lng'
    })
    
    logger.info(f"采样商家: {n_merchants}, 客户: {n_customers}")
    
    return merchants, customers


def generate_orders_with_network(node_list: list,
                                  node_coords: dict,
                                  n_orders: int = 100,
                                  n_merchants: int = 30,
                                  duration_hours: float = 1.0,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    基于项目路网生成订单（参考Olist的订单分布模式）
    
    Args:
        node_list: 路网节点列表
        node_coords: 节点坐标字典 {node_id: (x, y)}
        n_orders: 订单数量
        n_merchants: 商家数量
        duration_hours: 仿真时长（小时）
        random_seed: 随机种子
    
    Returns:
        订单DataFrame
    """
    np.random.seed(random_seed)
    
    duration_seconds = duration_hours * 3600
    
    # 随机选择商家节点（模拟Olist的商家聚集模式）
    merchant_nodes = np.random.choice(node_list, n_merchants, replace=False)
    
    orders = []
    for i in range(n_orders):
        # 随机选择商家和客户节点
        merchant_node = int(np.random.choice(merchant_nodes))
        customer_node = int(np.random.choice(node_list))
        
        # 避免商家和客户在同一节点
        while customer_node == merchant_node:
            customer_node = int(np.random.choice(node_list))
        
        # 生成到达时间（泊松过程）
        arrival_time = np.random.uniform(0, duration_seconds)
        
        # 生成服务时间参数（参考Olist的配送时间分布）
        preparation_time = np.random.uniform(180, 480)  # 3-8分钟备餐
        delivery_window = np.random.uniform(1500, 2700)  # 25-45分钟送达窗口
        
        # 获取坐标
        merchant_coords = node_coords.get(merchant_node, (0, 0))
        customer_coords = node_coords.get(customer_node, (0, 0))
        
        order = {
            'order_id': i + 1,
            'arrival_time': arrival_time,
            'merchant_node': merchant_node,
            'customer_node': customer_node,
            'merchant_coords': str(merchant_coords),
            'customer_coords': str(customer_coords),
            'preparation_time': preparation_time,
            'delivery_window': delivery_window,
            'earliest_pickup_time': arrival_time + preparation_time,
            'latest_delivery_time': arrival_time + preparation_time + delivery_window
        }
        orders.append(order)
    
    orders_df = pd.DataFrame(orders)
    orders_df = orders_df.sort_values('arrival_time').reset_index(drop=True)
    # 重新编号order_id
    orders_df['order_id'] = range(1, len(orders_df) + 1)
    
    logger.info(f"生成 {len(orders_df)} 个订单")
    logger.info(f"到达时间范围: {orders_df.arrival_time.min():.1f}s - {orders_df.arrival_time.max():.1f}s")
    
    return orders_df


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Olist数据集成（映射到项目路网）")
    logger.info("=" * 60)
    
    # 路径设置
    config = get_config()
    processed_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    orders_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载项目路网
    logger.info("加载项目路网...")
    network_config = config.get_network_config()
    graph, _ = extract_osm_network(network_config, processed_dir)
    logger.info(f"路网节点数: {len(graph.nodes)}")
    
    # 2. 加载节点映射
    mapping_file = processed_dir / "node_id_mapping.json"
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    node_list = [int(k) for k in mapping['node_to_idx'].keys()]
    
    # 3. 获取节点坐标
    node_coords = {}
    for node_id in node_list:
        if node_id in graph.nodes:
            node_data = graph.nodes[node_id]
            node_coords[node_id] = (node_data.get('x', 0), node_data.get('y', 0))
    
    logger.info(f"可用节点: {len(node_list)}")
    
    # 4. 生成订单数据（使用项目路网节点，参考Olist分布模式）
    orders_df = generate_orders_with_network(
        node_list=node_list,
        node_coords=node_coords,
        n_orders=100,
        n_merchants=30,
        duration_hours=1.0,
        random_seed=42
    )
    
    # 5. 保存订单数据
    orders_file = orders_dir / "orders_olist_1h.csv"
    orders_df.to_csv(orders_file, index=False)
    logger.info(f"订单数据保存到: {orders_file}")
    
    # 6. 显示统计信息
    logger.info("\n" + "=" * 40)
    logger.info("数据统计:")
    logger.info(f"  订单数: {len(orders_df)}")
    logger.info(f"  商家节点数: 30")
    logger.info(f"  仿真时长: 1小时")
    logger.info(f"  路网: {network_config['location']['city']}")
    
    return orders_file


if __name__ == "__main__":
    main()

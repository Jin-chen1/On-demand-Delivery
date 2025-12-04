"""
基于路网节点分层均匀采样生成合成订单数据

学术依据：
1. Solomon (1987) VRPTW基准问题中的Random分布类型
2. 网格分层采样确保空间均匀覆盖
3. 路网节点采样保证商家位置可达性

生成方法：
1. 将路网划分为K×K网格
2. 在每个非空网格中从路网节点均匀采样商家
3. 筛选degree≥2的交叉口节点（模拟真实选址）
4. 客户位置从全路网节点均匀采样
5. 生成商家-客户配对订单
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_road_network(network_path: str) -> nx.MultiDiGraph:
    """加载路网数据"""
    print(f"Loading road network from {network_path}...")
    G = nx.read_graphml(network_path)
    
    # 转换节点属性为数值类型
    for node in G.nodes():
        G.nodes[node]['x'] = float(G.nodes[node].get('x', 0))
        G.nodes[node]['y'] = float(G.nodes[node].get('y', 0))
        G.nodes[node]['street_count'] = int(G.nodes[node].get('street_count', 1))
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def get_network_bounds(G: nx.MultiDiGraph) -> dict:
    """获取路网地理边界"""
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lngs = [G.nodes[n]['x'] for n in G.nodes()]
    
    bounds = {
        'lat_min': min(lats),
        'lat_max': max(lats),
        'lng_min': min(lngs),
        'lng_max': max(lngs)
    }
    
    # 计算覆盖范围（km）
    lat_range_km = (bounds['lat_max'] - bounds['lat_min']) * 111.0
    lng_range_km = (bounds['lng_max'] - bounds['lng_min']) * 111.0 * np.cos(np.radians(np.mean(lats)))
    
    print(f"  Network bounds:")
    print(f"    Lat: [{bounds['lat_min']:.4f}, {bounds['lat_max']:.4f}]")
    print(f"    Lng: [{bounds['lng_min']:.4f}, {bounds['lng_max']:.4f}]")
    print(f"    Coverage: {lat_range_km:.2f}km x {lng_range_km:.2f}km")
    
    return bounds


def filter_intersection_nodes(G: nx.MultiDiGraph, min_degree: int = 2) -> list:
    """
    筛选交叉口节点（degree >= min_degree）
    
    学术依据：商家通常位于交通便利位置（十字路口、主干道）
    """
    intersection_nodes = []
    for node in G.nodes():
        # 使用street_count或度数判断
        street_count = G.nodes[node].get('street_count', 0)
        if street_count >= min_degree:
            intersection_nodes.append(node)
    
    # 如果street_count不可靠，使用图的度数
    if len(intersection_nodes) < G.number_of_nodes() * 0.1:
        intersection_nodes = [n for n in G.nodes() if G.degree(n) >= min_degree * 2]
    
    print(f"  Intersection nodes (degree >= {min_degree}): {len(intersection_nodes)}")
    return intersection_nodes


def create_grid_partition(G: nx.MultiDiGraph, bounds: dict, 
                          grid_size: int = 5) -> dict:
    """
    将路网划分为grid_size × grid_size网格
    
    返回：每个网格中的节点列表
    """
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / grid_size
    lng_step = (bounds['lng_max'] - bounds['lng_min']) / grid_size
    
    grid_nodes = defaultdict(list)
    
    for node in G.nodes():
        lat = G.nodes[node]['y']
        lng = G.nodes[node]['x']
        
        # 计算网格索引
        i = min(int((lat - bounds['lat_min']) / lat_step), grid_size - 1)
        j = min(int((lng - bounds['lng_min']) / lng_step), grid_size - 1)
        
        grid_nodes[(i, j)].append(node)
    
    # 统计非空网格
    non_empty_grids = sum(1 for k, v in grid_nodes.items() if len(v) > 0)
    print(f"  Grid partition: {grid_size}x{grid_size}")
    print(f"  Non-empty grids: {non_empty_grids}/{grid_size*grid_size}")
    
    return dict(grid_nodes)


def stratified_sample_merchants(G: nx.MultiDiGraph, 
                                 intersection_nodes: list,
                                 grid_nodes: dict,
                                 num_merchants: int,
                                 random_seed: int = 42) -> list:
    """
    分层均匀采样商家位置
    
    方法：
    1. 计算每个网格应采样的商家数（按节点密度加权或均匀）
    2. 从交叉口节点中采样
    """
    np.random.seed(random_seed)
    
    # 筛选网格中的交叉口节点
    intersection_set = set(intersection_nodes)
    grid_intersections = {}
    for (i, j), nodes in grid_nodes.items():
        intersections = [n for n in nodes if n in intersection_set]
        if intersections:
            grid_intersections[(i, j)] = intersections
    
    non_empty_grids = list(grid_intersections.keys())
    num_grids = len(non_empty_grids)
    
    if num_grids == 0:
        raise ValueError("No grids with intersection nodes found")
    
    # 计算每个网格的采样数量（均匀分配）
    base_per_grid = num_merchants // num_grids
    remainder = num_merchants % num_grids
    
    # 随机选择余数网格
    extra_grids = set(np.random.choice(num_grids, remainder, replace=False))
    
    sampled_merchants = []
    sampling_plan = {}
    
    for idx, (i, j) in enumerate(non_empty_grids):
        n_sample = base_per_grid + (1 if idx in extra_grids else 0)
        available = grid_intersections[(i, j)]
        
        # 如果可用节点不足，全部采样
        actual_sample = min(n_sample, len(available))
        sampled = np.random.choice(available, actual_sample, replace=False).tolist()
        sampled_merchants.extend(sampled)
        sampling_plan[(i, j)] = actual_sample
    
    print(f"  Sampled {len(sampled_merchants)} merchants from {num_grids} grids")
    
    return sampled_merchants


def sample_customers(G: nx.MultiDiGraph, 
                     all_nodes: list,
                     num_customers: int,
                     random_seed: int = 42) -> list:
    """
    均匀采样客户位置（从全路网节点）
    """
    np.random.seed(random_seed + 1000)  # 不同的种子
    sampled = np.random.choice(all_nodes, num_customers, replace=True).tolist()
    return sampled


def intensity_function(t: float, base_rate: float = 1.0) -> float:
    """
    非齐次泊松过程的强度函数 λ(t)
    
    模拟外卖订单的时间分布特征：
    - 午高峰 (11:00-13:00): 强度 × 3
    - 晚高峰 (17:00-20:00): 强度 × 4
    - 其他时段 (8:00-22:00): 基础强度
    - 营业外时段: 强度 = 0
    
    Args:
        t: 时间（秒，从0点开始）
        base_rate: 基础订单到达率（订单/秒）
    
    Returns:
        该时刻的强度值 λ(t)
    """
    hour = (t / 3600) % 24  # 转换为小时
    
    if hour < 8 or hour >= 22:
        # 非营业时间
        return 0.0
    elif 11 <= hour < 13:
        # 午高峰
        return base_rate * 3.0
    elif 17 <= hour < 20:
        # 晚高峰
        return base_rate * 4.0
    else:
        # 普通时段
        return base_rate


def generate_nhpp_arrivals(target_orders: int, 
                           start_hour: float = 8.0,
                           end_hour: float = 22.0,
                           random_seed: int = 42) -> list:
    """
    使用薄化法（Thinning Method）生成非齐次泊松过程的到达时间
    
    参考：Lewis, P.A.W. and Shedler, G.S. (1979) 
    "Simulation of nonhomogeneous Poisson processes by thinning"
    
    Args:
        target_orders: 目标订单数（用于计算基础强度）
        start_hour: 开始时间（小时）
        end_hour: 结束时间（小时）
        random_seed: 随机种子
    
    Returns:
        到达时间列表（秒）
    """
    np.random.seed(random_seed)
    
    # 计算时间范围（秒）
    t_start = start_hour * 3600
    t_end = end_hour * 3600
    duration = t_end - t_start
    
    # 计算基础强度以达到目标订单数
    # 积分 λ(t) dt 应约等于 target_orders
    # 营业时间14小时：普通时段9小时(×1) + 午高峰2小时(×3) + 晚高峰3小时(×4)
    # 等效时长 = 9 + 2*3 + 3*4 = 9 + 6 + 12 = 27 小时等效基础时长
    equivalent_hours = 9 + 2 * 3 + 3 * 4  # = 27
    base_rate = target_orders / (equivalent_hours * 3600)
    
    # 最大强度（晚高峰时的4倍基础强度）
    lambda_max = base_rate * 4.0
    
    # 薄化法生成到达时间
    arrival_times = []
    t = t_start
    
    while t < t_end:
        # 生成指数分布的时间间隔（使用最大强度）
        u1 = np.random.random()
        inter_arrival = -np.log(u1) / lambda_max
        t = t + inter_arrival
        
        if t >= t_end:
            break
        
        # 薄化：以概率 λ(t)/λ_max 接受该到达
        u2 = np.random.random()
        lambda_t = intensity_function(t, base_rate)
        
        if u2 <= lambda_t / lambda_max:
            arrival_times.append(t)
    
    return arrival_times


def generate_orders(G: nx.MultiDiGraph,
                    merchant_nodes: list,
                    customer_nodes: list,
                    num_orders: int,
                    simulation_hours: int = 12,
                    random_seed: int = 42) -> pd.DataFrame:
    """
    生成订单数据（使用非齐次泊松过程生成到达时间）
    
    参数：
        merchant_nodes: 商家节点列表
        customer_nodes: 客户节点池
        num_orders: 目标订单数量（实际数量可能略有浮动）
        simulation_hours: 仿真时长（小时）
        random_seed: 随机种子
    
    到达时间分布特征：
        - 午高峰 (11:00-13:00): 强度 × 3
        - 晚高峰 (17:00-20:00): 强度 × 4
        - 其他时段: 基础强度
    """
    np.random.seed(random_seed)
    
    orders = []
    
    # 使用非齐次泊松过程生成到达时间
    # 薄化法（Thinning Method）- Lewis & Shedler (1979)
    arrival_times = generate_nhpp_arrivals(
        target_orders=num_orders,
        start_hour=8.0,
        end_hour=22.0,
        random_seed=random_seed
    )
    
    # 如果生成的订单数与目标差距较大，进行调整
    actual_count = len(arrival_times)
    if actual_count < num_orders * 0.9:
        # 订单数不足，重新生成（增加基础强度）
        np.random.seed(random_seed + 1)
        arrival_times = generate_nhpp_arrivals(
            target_orders=int(num_orders * 1.2),
            start_hour=8.0,
            end_hour=22.0,
            random_seed=random_seed + 1
        )
    
    # 截断到目标数量
    if len(arrival_times) > num_orders:
        arrival_times = arrival_times[:num_orders]
    
    for i, arrival_time in enumerate(arrival_times):
        # 随机选择商家
        merchant_node = np.random.choice(merchant_nodes)
        merchant_lat = G.nodes[merchant_node]['y']
        merchant_lng = G.nodes[merchant_node]['x']
        
        # 随机选择客户（从预采样池）
        customer_node = customer_nodes[i % len(customer_nodes)]
        customer_lat = G.nodes[customer_node]['y']
        customer_lng = G.nodes[customer_node]['x']
        
        # 计算直线距离
        distance_km = np.sqrt(
            ((merchant_lat - customer_lat) * 111.0) ** 2 +
            ((merchant_lng - customer_lng) * 111.0 * np.cos(np.radians(merchant_lat))) ** 2
        )
        
        # 准备时间（5-15分钟）
        prep_time = np.random.uniform(300, 900)
        
        # 时间窗（30-60分钟）
        delivery_window = np.random.uniform(1800, 3600)
        deadline = arrival_time + delivery_window
        
        # 最早取餐时间 = 到达时间 + 准备时间
        pickup_time = arrival_time + prep_time
        
        orders.append({
            'order_id': i + 1,
            'arrival_time': arrival_time,
            'merchant_node': merchant_node,
            'customer_node': customer_node,
            'merchant_lat': merchant_lat,
            'merchant_lng': merchant_lng,
            'customer_lat': customer_lat,
            'customer_lng': customer_lng,
            'distance_km': distance_km,
            'preparation_time': prep_time,
            'deadline': deadline,
            'delivery_window': delivery_window,
            'earliest_pickup_time': pickup_time,
            'latest_delivery_time': deadline
        })
    
    df = pd.DataFrame(orders)
    return df


def visualize_distribution(G: nx.MultiDiGraph,
                           merchant_nodes: list,
                           customer_nodes: list,
                           grid_nodes: dict,
                           bounds: dict,
                           grid_size: int,
                           output_path: str):
    """
    可视化商家和客户分布
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：商家分布 + 网格
    ax1 = axes[0]
    
    # 绘制所有节点（灰色背景）
    all_lats = [G.nodes[n]['y'] for n in G.nodes()]
    all_lngs = [G.nodes[n]['x'] for n in G.nodes()]
    ax1.scatter(all_lngs, all_lats, c='lightgray', s=1, alpha=0.3, label='Road nodes')
    
    # 绘制商家（红色）
    m_lats = [G.nodes[n]['y'] for n in merchant_nodes]
    m_lngs = [G.nodes[n]['x'] for n in merchant_nodes]
    ax1.scatter(m_lngs, m_lats, c='red', s=50, marker='s', 
                edgecolors='black', linewidths=0.5, label=f'Merchants ({len(merchant_nodes)})')
    
    # 绘制网格线
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / grid_size
    lng_step = (bounds['lng_max'] - bounds['lng_min']) / grid_size
    
    for i in range(grid_size + 1):
        lat = bounds['lat_min'] + i * lat_step
        ax1.axhline(y=lat, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
    for j in range(grid_size + 1):
        lng = bounds['lng_min'] + j * lng_step
        ax1.axvline(x=lng, color='blue', linestyle='--', alpha=0.3, linewidth=0.5)
    
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'Merchant Distribution (Stratified Grid Sampling)\n{grid_size}x{grid_size} Grid', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    
    # 右图：客户分布
    ax2 = axes[1]
    
    ax2.scatter(all_lngs, all_lats, c='lightgray', s=1, alpha=0.3, label='Road nodes')
    
    # 绘制客户（蓝色）
    c_lats = [G.nodes[n]['y'] for n in customer_nodes[:500]]  # 仅绘制前500个
    c_lngs = [G.nodes[n]['x'] for n in customer_nodes[:500]]
    ax2.scatter(c_lngs, c_lats, c='blue', s=20, alpha=0.5, 
                label=f'Customers (sample of {min(500, len(customer_nodes))})')
    
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.set_title('Customer Distribution (Uniform Random Sampling)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: {output_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Uniform Grid Sampling Order Generation")
    print("Based on Solomon (1987) Random Distribution Methodology")
    print("=" * 70)
    
    # 配置参数
    NETWORK_PATH = project_root / "data" / "processed" / "shanghai" / "road_network.graphml"
    OUTPUT_DIR = project_root / "data" / "orders"
    FIGURE_DIR = project_root / "docs" / "figures"
    
    GRID_SIZE = 5  # 5x5网格
    MIN_DEGREE = 2  # 交叉口最小度数
    RANDOM_SEED = 42
    
    # 订单规模配置
    ORDER_CONFIGS = [
        {'num_orders': 100, 'num_merchants': 25},
        {'num_orders': 300, 'num_merchants': 40},
        {'num_orders': 500, 'num_merchants': 50},
        {'num_orders': 1000, 'num_merchants': 100},
    ]
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载路网
    G = load_road_network(str(NETWORK_PATH))
    bounds = get_network_bounds(G)
    
    # 筛选交叉口节点
    intersection_nodes = filter_intersection_nodes(G, MIN_DEGREE)
    
    # 创建网格划分
    grid_nodes = create_grid_partition(G, bounds, GRID_SIZE)
    
    # 获取所有节点列表
    all_nodes = list(G.nodes())
    
    # 为每个规模生成订单
    for config in ORDER_CONFIGS:
        num_orders = config['num_orders']
        num_merchants = config['num_merchants']
        
        print(f"\n{'=' * 70}")
        print(f"Generating {num_orders} orders with {num_merchants} merchants")
        print("=" * 70)
        
        # 分层采样商家
        merchant_nodes = stratified_sample_merchants(
            G, intersection_nodes, grid_nodes,
            num_merchants, RANDOM_SEED
        )
        
        # 均匀采样客户
        customer_nodes = sample_customers(
            G, all_nodes, num_orders * 2, RANDOM_SEED
        )
        
        # 生成订单
        orders_df = generate_orders(
            G, merchant_nodes, customer_nodes,
            num_orders, simulation_hours=12,
            random_seed=RANDOM_SEED
        )
        
        # 保存订单
        output_file = OUTPUT_DIR / f"uniform_grid_{num_orders}.csv"
        orders_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # 统计信息
        print(f"\n  Order Statistics:")
        print(f"    Total orders: {len(orders_df)}")
        print(f"    Unique merchants: {orders_df['merchant_node'].nunique()}")
        print(f"    Merchant lat range: [{orders_df['merchant_lat'].min():.5f}, {orders_df['merchant_lat'].max():.5f}]")
        print(f"    Merchant lng range: [{orders_df['merchant_lng'].min():.5f}, {orders_df['merchant_lng'].max():.5f}]")
        print(f"    Avg distance: {orders_df['distance_km'].mean():.2f} km")
        print(f"    Avg delivery window: {orders_df['delivery_window'].mean()/60:.1f} min")
    
    # 可视化（使用最大规模的数据）
    print(f"\n{'=' * 70}")
    print("Generating visualization...")
    print("=" * 70)
    
    # 使用100商家的配置
    merchant_nodes = stratified_sample_merchants(
        G, intersection_nodes, grid_nodes, 100, RANDOM_SEED
    )
    customer_nodes = sample_customers(G, all_nodes, 500, RANDOM_SEED)
    
    visualize_distribution(
        G, merchant_nodes, customer_nodes,
        grid_nodes, bounds, GRID_SIZE,
        str(FIGURE_DIR / "uniform_grid_distribution.png")
    )
    
    print(f"\n{'=' * 70}")
    print("Done! Generated files:")
    for config in ORDER_CONFIGS:
        print(f"  - uniform_grid_{config['num_orders']}.csv")
    print(f"  - uniform_grid_distribution.png")
    print("=" * 70)
    
    # 输出论文引用建议
    print("\n" + "=" * 70)
    print("ACADEMIC CITATION NOTES")
    print("=" * 70)
    print("""
For paper writing, you may describe this methodology as:

"Following the spatial distribution paradigm established by Solomon (1987) 
for VRPTW benchmark instances, we generate synthetic merchant locations 
using a stratified grid sampling approach. The service area is partitioned 
into a 5×5 grid, and merchant nodes are uniformly sampled from road network 
intersections (nodes with degree ≥ 2) within each grid cell. This ensures:
(a) spatial uniformity across the service region,
(b) realistic accessibility of merchant locations, and
(c) reproducibility of experimental results.

Customer locations are uniformly sampled from all road network nodes, 
following the Random (R-type) distribution pattern in Solomon's taxonomy."

Reference:
Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling 
problems with time window constraints. Operations Research, 35(2), 254-265.
""")


if __name__ == "__main__":
    main()

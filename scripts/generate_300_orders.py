"""
生成300单订单数据
基于 generate_uniform_grid_orders.py 的逻辑
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 70)
    print("Generating 300 Orders with Uniform Grid Sampling")
    print("=" * 70)
    
    # 配置
    NETWORK_PATH = project_root / "data" / "processed" / "shanghai" / "road_network.graphml"
    OUTPUT_DIR = project_root / "data" / "orders"
    GRID_SIZE = 5
    MIN_DEGREE = 2
    RANDOM_SEED = 42
    
    NUM_ORDERS = 300
    NUM_MERCHANTS = 40
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载路网
    print(f"Loading road network from {NETWORK_PATH}...")
    G = nx.read_graphml(str(NETWORK_PATH))
    
    # 转换节点属性
    for node in G.nodes():
        G.nodes[node]['x'] = float(G.nodes[node].get('x', 0))
        G.nodes[node]['y'] = float(G.nodes[node].get('y', 0))
        G.nodes[node]['street_count'] = int(G.nodes[node].get('street_count', 1))
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # 获取边界
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lngs = [G.nodes[n]['x'] for n in G.nodes()]
    bounds = {
        'lat_min': min(lats), 'lat_max': max(lats),
        'lng_min': min(lngs), 'lng_max': max(lngs)
    }
    
    # 筛选交叉口节点
    intersection_nodes = [n for n in G.nodes() if G.nodes[n].get('street_count', 0) >= MIN_DEGREE]
    if len(intersection_nodes) < G.number_of_nodes() * 0.1:
        intersection_nodes = [n for n in G.nodes() if G.degree(n) >= MIN_DEGREE * 2]
    print(f"  Intersection nodes: {len(intersection_nodes)}")
    
    # 网格划分
    lat_step = (bounds['lat_max'] - bounds['lat_min']) / GRID_SIZE
    lng_step = (bounds['lng_max'] - bounds['lng_min']) / GRID_SIZE
    grid_nodes = defaultdict(list)
    for node in G.nodes():
        lat = G.nodes[node]['y']
        lng = G.nodes[node]['x']
        i = min(int((lat - bounds['lat_min']) / lat_step), GRID_SIZE - 1)
        j = min(int((lng - bounds['lng_min']) / lng_step), GRID_SIZE - 1)
        grid_nodes[(i, j)].append(node)
    
    # 分层采样商家
    np.random.seed(RANDOM_SEED)
    intersection_set = set(intersection_nodes)
    grid_intersections = {}
    for (i, j), nodes in grid_nodes.items():
        intersections = [n for n in nodes if n in intersection_set]
        if intersections:
            grid_intersections[(i, j)] = intersections
    
    non_empty_grids = list(grid_intersections.keys())
    num_grids = len(non_empty_grids)
    base_per_grid = NUM_MERCHANTS // num_grids
    remainder = NUM_MERCHANTS % num_grids
    extra_grids = set(np.random.choice(num_grids, remainder, replace=False))
    
    sampled_merchants = []
    for idx, (i, j) in enumerate(non_empty_grids):
        n_sample = base_per_grid + (1 if idx in extra_grids else 0)
        available = grid_intersections[(i, j)]
        actual_sample = min(n_sample, len(available))
        sampled = np.random.choice(available, actual_sample, replace=False).tolist()
        sampled_merchants.extend(sampled)
    
    print(f"  Sampled {len(sampled_merchants)} merchants from {num_grids} grids")
    
    # 采样客户
    np.random.seed(RANDOM_SEED + 1000)
    all_nodes = list(G.nodes())
    customer_nodes = np.random.choice(all_nodes, NUM_ORDERS * 2, replace=True).tolist()
    
    # 生成订单
    np.random.seed(RANDOM_SEED)
    peak_hours = [(11, 13), (17, 20)]
    peak_ratio = 0.6
    
    arrival_times = []
    for _ in range(NUM_ORDERS):
        if np.random.random() < peak_ratio:
            peak = peak_hours[np.random.randint(0, len(peak_hours))]
            hour = np.random.uniform(peak[0], peak[1])
        else:
            hour = np.random.uniform(8, 22)
        arrival_times.append(hour * 3600)
    arrival_times.sort()
    
    orders = []
    for i, arrival_time in enumerate(arrival_times):
        merchant_node = np.random.choice(sampled_merchants)
        merchant_lat = G.nodes[merchant_node]['y']
        merchant_lng = G.nodes[merchant_node]['x']
        
        customer_node = customer_nodes[i % len(customer_nodes)]
        customer_lat = G.nodes[customer_node]['y']
        customer_lng = G.nodes[customer_node]['x']
        
        distance_km = np.sqrt(
            ((merchant_lat - customer_lat) * 111.0) ** 2 +
            ((merchant_lng - customer_lng) * 111.0 * np.cos(np.radians(merchant_lat))) ** 2
        )
        
        prep_time = np.random.uniform(300, 900)
        delivery_window = np.random.uniform(1800, 3600)
        deadline = arrival_time + delivery_window
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
    output_file = OUTPUT_DIR / "uniform_grid_300.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"Saved: {output_file}")
    print(f"\nOrder Statistics:")
    print(f"  Total orders: {len(df)}")
    print(f"  Unique merchants: {df['merchant_node'].nunique()}")
    print(f"  Avg distance: {df['distance_km'].mean():.2f} km")
    print(f"  Avg delivery window: {df['delivery_window'].mean()/60:.1f} min")
    print(f"  Arrival time range: {df['arrival_time'].min():.0f}s - {df['arrival_time'].max():.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""下载印度班加罗尔(Bangalore)的OSM路网数据"""
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import json
from pathlib import Path

# 配置OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# 首先检查订单数据的坐标范围
print("检查班加罗尔区域订单...")
df = pd.read_csv('D:/0On-demand Delivery/data/orders/real_delivery_100.csv')

# 班加罗尔/金奈区域 (11-14°N, 77-81°E)
mask = (df['merchant_lat'] >= 11) & (df['merchant_lat'] <= 14) & \
       (df['merchant_lng'] >= 77) & (df['merchant_lng'] <= 81) & \
       (df['merchant_lat'] > 1)
df_bangalore = df[mask].copy()
df_bangalore['order_id'] = range(1, len(df_bangalore) + 1)

print(f"班加罗尔区域订单数: {len(df_bangalore)}")
print(f"商家坐标: lat {df_bangalore['merchant_lat'].min():.2f}-{df_bangalore['merchant_lat'].max():.2f}")
print(f"         lng {df_bangalore['merchant_lng'].min():.2f}-{df_bangalore['merchant_lng'].max():.2f}")
print(f"客户坐标: lat {df_bangalore['customer_lat'].min():.2f}-{df_bangalore['customer_lat'].max():.2f}")
print(f"         lng {df_bangalore['customer_lng'].min():.2f}-{df_bangalore['customer_lng'].max():.2f}")

# 计算边界（扩展一点以确保覆盖）
north = max(df_bangalore['merchant_lat'].max(), df_bangalore['customer_lat'].max()) + 0.02
south = min(df_bangalore['merchant_lat'].min(), df_bangalore['customer_lat'].min()) - 0.02
east = max(df_bangalore['merchant_lng'].max(), df_bangalore['customer_lng'].max()) + 0.02
west = min(df_bangalore['merchant_lng'].min(), df_bangalore['customer_lng'].min()) - 0.02

print(f"\n下载路网边界: N={north:.2f}, S={south:.2f}, E={east:.2f}, W={west:.2f}")
print(f"范围: {(north-south)*111:.1f}km x {(east-west)*111:.1f}km")

# 下载道路网络
print("\n开始下载...")
bbox = (west, south, east, north)
G = ox.graph_from_bbox(
    bbox=bbox,
    network_type='drive',
    simplify=True
)

print(f"\n路网下载成功!")
print(f"  节点数: {G.number_of_nodes()}")
print(f"  边数: {G.number_of_edges()}")

# 保存完整路网
output_dir = Path("D:/0On-demand Delivery/data/processed/bangalore")
output_dir.mkdir(parents=True, exist_ok=True)
ox.save_graphml(G, output_dir / "road_network.graphml")
print(f"已保存完整路网: {output_dir / 'road_network.graphml'}")

# 保存班加罗尔订单
orders_path = 'D:/0On-demand Delivery/data/orders/real_delivery_bangalore.csv'
df_bangalore.to_csv(orders_path, index=False)
print(f"已保存订单: {orders_path}")

# 创建订单子图
print("\n创建订单覆盖子图...")

# 获取所有订单位置
locations = []
for _, row in df_bangalore.iterrows():
    locations.append((row['merchant_lat'], row['merchant_lng']))
    locations.append((row['customer_lat'], row['customer_lng']))

# 找到最近节点
node_coords = {n: (G.nodes[n]['y'], G.nodes[n]['x']) for n in G.nodes()}
node_list = list(node_coords.keys())
node_lats = np.array([node_coords[n][0] for n in node_list])
node_lngs = np.array([node_coords[n][1] for n in node_list])

required_nodes = set()
for lat, lng in locations:
    distances = np.sqrt((node_lats - lat)**2 + (node_lngs - lng)**2)
    nearest_idx = np.argmin(distances)
    required_nodes.add(node_list[nearest_idx])

print(f"必需节点: {len(required_nodes)}")

# 计算节点间最短路径
G_undirected = G.to_undirected()
path_nodes = set(required_nodes)
required_list = list(required_nodes)

for i, source in enumerate(required_list):
    for target in required_list[i+1:]:
        try:
            path = nx.shortest_path(G_undirected, source, target, weight='length')
            path_nodes.update(path)
        except nx.NetworkXNoPath:
            pass

print(f"路径覆盖节点: {len(path_nodes)}")

# 创建子图
G_sub = G.subgraph(path_nodes).copy()
if not nx.is_weakly_connected(G_sub):
    largest_wcc = max(nx.weakly_connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_wcc).copy()

print(f"子图: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

# 保存子图
ox.save_graphml(G_sub, output_dir / "road_network_orders.graphml")

# 计算距离矩阵
print("\n计算距离矩阵...")
nodes = list(G_sub.nodes())
n = len(nodes)
print(f"矩阵大小: {n} x {n}")

G_sub_undirected = G_sub.to_undirected()
node_to_idx = {node: idx for idx, node in enumerate(nodes)}
dist_matrix = np.full((n, n), np.inf)
np.fill_diagonal(dist_matrix, 0)

for i, source in enumerate(nodes):
    if i % 100 == 0:
        print(f"  进度: {i}/{n}")
    try:
        lengths = nx.single_source_dijkstra_path_length(G_sub_undirected, source, weight='length')
        for target, length in lengths.items():
            if target in node_to_idx:
                j = node_to_idx[target]
                dist_matrix[i, j] = length
    except:
        pass

connected = np.isfinite(dist_matrix).sum() / (n * n) * 100
print(f"连通性: {connected:.1f}%")

# 保存
np.save(output_dir / "distance_matrix_orders.npy", dist_matrix)

speed_mps = 15 * 1000 / 3600
time_matrix = dist_matrix / speed_mps
np.save(output_dir / "time_matrix_orders.npy", time_matrix)

mapping = {str(node): idx for idx, node in enumerate(nodes)}
with open(output_dir / "node_mapping_orders.json", 'w') as f:
    json.dump(mapping, f)

# 统计
valid_dists = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
print(f"\n距离统计:")
print(f"  平均: {np.mean(valid_dists)/1000:.1f} km")
print(f"  最大: {np.max(valid_dists)/1000:.1f} km")

print(f"\n完成! 班加罗尔路网已准备好")
print(f"  订单数: {len(df_bangalore)}")
print(f"  路网节点: {G_sub.number_of_nodes()}")

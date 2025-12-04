"""基于订单位置创建最小路网子图"""
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import json
from pathlib import Path

# 读取苏拉特完整路网
print("读取苏拉特完整路网...")
G_full = ox.load_graphml("D:/0On-demand Delivery/data/processed/surat/road_network.graphml")
print(f"完整路网: {G_full.number_of_nodes()} 节点")

# 读取订单
print("\n读取订单...")
df = pd.read_csv('D:/0On-demand Delivery/data/orders/real_delivery_single_city.csv')
print(f"订单数: {len(df)}")

# 获取所有需要的位置坐标
locations = []
for _, row in df.iterrows():
    locations.append((row['merchant_lat'], row['merchant_lng']))
    locations.append((row['customer_lat'], row['customer_lng']))

print(f"需要覆盖的位置: {len(locations)}")

# 找到每个位置最近的路网节点
print("\n查找最近节点...")
node_coords = {n: (G_full.nodes[n]['y'], G_full.nodes[n]['x']) for n in G_full.nodes()}
node_list = list(node_coords.keys())
node_lats = np.array([node_coords[n][0] for n in node_list])
node_lngs = np.array([node_coords[n][1] for n in node_list])

required_nodes = set()
for lat, lng in locations:
    distances = np.sqrt((node_lats - lat)**2 + (node_lngs - lng)**2)
    nearest_idx = np.argmin(distances)
    required_nodes.add(node_list[nearest_idx])

print(f"必需节点: {len(required_nodes)}")

# 转换为无向图以确保双向连通
print("\n转换为无向图...")
G_undirected = G_full.to_undirected()

# 为每对节点找到最短路径上的所有节点
print("计算节点间最短路径...")
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
G_sub = G_full.subgraph(path_nodes).copy()

# 确保连通
if not nx.is_weakly_connected(G_sub):
    largest_wcc = max(nx.weakly_connected_components(G_sub), key=len)
    G_sub = G_sub.subgraph(largest_wcc).copy()

print(f"\n子图: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

# 保存子图
output_dir = Path("D:/0On-demand Delivery/data/processed/surat")
ox.save_graphml(G_sub, output_dir / "road_network_orders.graphml")
print(f"已保存: road_network_orders.graphml")

# 计算距离矩阵（使用无向图）
print("\n计算距离矩阵...")
nodes = list(G_sub.nodes())
n = len(nodes)
print(f"矩阵大小: {n} x {n}")

# 转换子图为无向图以确保双向连通
G_sub_undirected = G_sub.to_undirected()

node_to_idx = {node: idx for idx, node in enumerate(nodes)}
dist_matrix = np.full((n, n), np.inf)
np.fill_diagonal(dist_matrix, 0)

for i, source in enumerate(nodes):
    if i % 50 == 0:
        print(f"  进度: {i}/{n}")
    try:
        lengths = nx.single_source_dijkstra_path_length(G_sub_undirected, source, weight='length')
        for target, length in lengths.items():
            if target in node_to_idx:
                j = node_to_idx[target]
                dist_matrix[i, j] = length
    except:
        pass

# 检查连通性
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

# 验证
lats = [G_sub.nodes[n]['y'] for n in G_sub.nodes()]
lngs = [G_sub.nodes[n]['x'] for n in G_sub.nodes()]
print(f"\n子图坐标范围:")
print(f"  lat: {min(lats):.4f} - {max(lats):.4f}")
print(f"  lng: {min(lngs):.4f} - {max(lngs):.4f}")

print(f"\n完成!")

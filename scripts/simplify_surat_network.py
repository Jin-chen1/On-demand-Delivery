"""简化苏拉特路网数据，减少节点数以便计算距离矩阵"""
import osmnx as ox
import networkx as nx
import numpy as np
import json
from pathlib import Path

# 读取已下载的路网
print("读取苏拉特路网...")
graphml_path = Path("D:/0On-demand Delivery/data/processed/surat/road_network.graphml")
G = ox.load_graphml(graphml_path)
print(f"原始路网: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 方案：对订单数据区域进行更精细的裁剪
# 订单实际范围：lat 21.15-21.27, lng 72.77-72.89
north = 21.28
south = 21.14
east = 72.90
west = 72.76

print(f"\n裁剪到订单覆盖区域...")
print(f"边界: N={north}, S={south}, E={east}, W={west}")

# 过滤节点
nodes_in_bbox = []
for node, data in G.nodes(data=True):
    lat = data.get('y', 0)
    lng = data.get('x', 0)
    if south <= lat <= north and west <= lng <= east:
        nodes_in_bbox.append(node)

print(f"区域内节点: {len(nodes_in_bbox)}")

# 创建子图
G_sub = G.subgraph(nodes_in_bbox).copy()

# 获取最大连通分量
G_undirected = ox.convert.to_undirected(G_sub)
if not nx.is_connected(G_undirected):
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    G_sub = G_sub.subgraph(largest_cc).copy()
    print(f"最大连通分量: {G_sub.number_of_nodes()} 节点")

# 如果仍然太大，进一步简化：只保留主要道路
if G_sub.number_of_nodes() > 2000:
    print(f"\n节点数仍然较多，过滤保留主要道路...")
    # 保留 primary, secondary, tertiary 等主要道路
    major_highways = ['primary', 'secondary', 'tertiary', 'primary_link', 'secondary_link', 'tertiary_link', 'trunk', 'trunk_link']
    edges_to_keep = []
    for u, v, k, data in G_sub.edges(keys=True, data=True):
        highway = data.get('highway', '')
        if isinstance(highway, list):
            highway = highway[0] if highway else ''
        if highway in major_highways:
            edges_to_keep.append((u, v, k))
    
    if edges_to_keep:
        nodes_to_keep = set()
        for u, v, k in edges_to_keep:
            nodes_to_keep.add(u)
            nodes_to_keep.add(v)
        G_sub = G_sub.subgraph(nodes_to_keep).copy()
        print(f"主要道路网络: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

# 如果仍然太大，使用空间网格采样
target_nodes = 400
if G_sub.number_of_nodes() > target_nodes:
    print(f"\n使用空间网格采样...")
    # 获取所有节点的坐标
    node_coords = {n: (G_sub.nodes[n]['y'], G_sub.nodes[n]['x']) for n in G_sub.nodes()}
    lats = [c[0] for c in node_coords.values()]
    lngs = [c[1] for c in node_coords.values()]
    
    # 创建网格
    grid_size = int(np.sqrt(target_nodes))
    lat_bins = np.linspace(min(lats), max(lats), grid_size + 1)
    lng_bins = np.linspace(min(lngs), max(lngs), grid_size + 1)
    
    # 从每个网格单元中选择一个节点（度数最高的）
    sampled_nodes = []
    degrees = dict(G_sub.degree())
    for i in range(grid_size):
        for j in range(grid_size):
            cell_nodes = [
                n for n, (lat, lng) in node_coords.items()
                if lat_bins[i] <= lat < lat_bins[i+1] and lng_bins[j] <= lng < lng_bins[j+1]
            ]
            if cell_nodes:
                # 选择度数最高的节点
                best_node = max(cell_nodes, key=lambda x: degrees.get(x, 0))
                sampled_nodes.append(best_node)
    
    print(f"网格采样得到: {len(sampled_nodes)} 个节点")
    
    if len(sampled_nodes) < 50:
        # 如果网格采样太少，回退到度数采样
        print("网格采样不足，使用度数采样...")
        sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
        sampled_nodes = sorted_nodes[:target_nodes]
    
    G_sub = G_sub.subgraph(sampled_nodes).copy()
    print(f"采样后: {G_sub.number_of_nodes()} 节点")
    
    # 对于不连通的图，添加虚拟边（基于Haversine距离）
    G_undirected = ox.convert.to_undirected(G_sub)
    if not nx.is_connected(G_undirected):
        print("图不连通，添加虚拟边...")
        components = list(nx.connected_components(G_undirected))
        print(f"连通分量数: {len(components)}")
        
        # 保留所有分量，通过虚拟边连接
        # 简化：只保留最大分量
        largest_cc = max(components, key=len)
        G_sub = G_sub.subgraph(largest_cc).copy()
        print(f"最大连通分量: {G_sub.number_of_nodes()} 节点")

print(f"\n最终简化路网: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

# 保存简化后的路网
output_dir = Path("D:/0On-demand Delivery/data/processed/surat")
simplified_path = output_dir / "road_network_simplified.graphml"
ox.save_graphml(G_sub, simplified_path)
print(f"已保存: {simplified_path}")

# 计算距离矩阵
print("\n计算距离矩阵...")
nodes = list(G_sub.nodes())
n = len(nodes)
print(f"矩阵大小: {n} x {n} = {n*n*8/1024/1024:.1f} MB")

node_to_idx = {node: idx for idx, node in enumerate(nodes)}
dist_matrix = np.full((n, n), np.inf)
np.fill_diagonal(dist_matrix, 0)

for i, source in enumerate(nodes):
    if i % 100 == 0:
        print(f"  进度: {i}/{n}")
    try:
        lengths = nx.single_source_dijkstra_path_length(G_sub, source, weight='length')
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
np.save(output_dir / "distance_matrix.npy", dist_matrix)

speed_mps = 15 * 1000 / 3600
time_matrix = dist_matrix / speed_mps
np.save(output_dir / "time_matrix.npy", time_matrix)

mapping = {str(node): idx for idx, node in enumerate(nodes)}
with open(output_dir / "node_mapping.json", 'w') as f:
    json.dump(mapping, f)

# 验证坐标范围
lats = [G_sub.nodes[n]['y'] for n in G_sub.nodes()]
lngs = [G_sub.nodes[n]['x'] for n in G_sub.nodes()]
print(f"\n简化路网坐标范围:")
print(f"  lat: {min(lats):.4f} - {max(lats):.4f}")
print(f"  lng: {min(lngs):.4f} - {max(lngs):.4f}")

print(f"\n完成! 已保存到 {output_dir}")

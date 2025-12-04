"""下载印度苏拉特(Surat)的OSM路网数据"""
import osmnx as ox
import networkx as nx
import numpy as np
from pathlib import Path

# 配置OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# 订单数据的坐标范围
# 商家: lat 21.15-21.19, lng 72.77-72.81
# 客户: lat 21.18-21.27, lng 72.81-72.89

# 扩展边界以确保覆盖所有订单
north = 21.30  # 最北
south = 21.10  # 最南
east = 72.95   # 最东
west = 72.70   # 最西

print(f"下载苏拉特路网数据...")
print(f"边界: N={north}, S={south}, E={east}, W={west}")
print(f"范围: {(north-south)*111:.1f}km x {(east-west)*111:.1f}km")

# 下载道路网络 (drive模式 = 可驾驶道路)
# bbox格式: (north, south, east, west) 或 (left, bottom, right, top)
try:
    # 新版OSMnx使用bbox元组 (west, south, east, north)
    bbox = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type='drive',
        simplify=True
    )
    
    print(f"\n路网下载成功!")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    
    # 转换为无向图以便于路径规划
    G_undirected = ox.convert.to_undirected(G)
    
    # 获取最大连通分量
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"  最大连通分量节点数: {G.number_of_nodes()}")
    
    # 保存路网
    output_dir = Path("D:/0On-demand Delivery/data/processed/surat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存GraphML格式
    graphml_path = output_dir / "road_network.graphml"
    ox.save_graphml(G, graphml_path)
    print(f"\n已保存路网: {graphml_path}")
    
    # 计算并保存距离矩阵
    print("\n计算距离矩阵...")
    nodes = list(G.nodes())
    n = len(nodes)
    
    # 创建节点映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 初始化距离矩阵
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    
    # 计算最短路径距离（使用边长度）
    print(f"计算 {n} 个节点间的最短路径...")
    for i, source in enumerate(nodes):
        if i % 100 == 0:
            print(f"  进度: {i}/{n}")
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='length')
            for target, length in lengths.items():
                j = node_to_idx[target]
                dist_matrix[i, j] = length
        except:
            pass
    
    # 保存距离矩阵
    np.save(output_dir / "distance_matrix.npy", dist_matrix)
    
    # 计算时间矩阵（假设平均速度15km/h）
    speed_mps = 15 * 1000 / 3600  # 米/秒
    time_matrix = dist_matrix / speed_mps
    np.save(output_dir / "time_matrix.npy", time_matrix)
    
    # 保存节点映射
    import json
    mapping = {str(node): idx for idx, node in enumerate(nodes)}
    with open(output_dir / "node_mapping.json", 'w') as f:
        json.dump(mapping, f)
    
    print(f"\n已保存:")
    print(f"  - distance_matrix.npy: {dist_matrix.shape}")
    print(f"  - time_matrix.npy: {time_matrix.shape}")
    print(f"  - node_mapping.json: {len(mapping)} 个节点")
    
    # 验证坐标范围
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lngs = [G.nodes[n]['x'] for n in G.nodes()]
    print(f"\n路网坐标范围:")
    print(f"  lat: {min(lats):.4f} - {max(lats):.4f}")
    print(f"  lng: {min(lngs):.4f} - {max(lngs):.4f}")
    
except Exception as e:
    print(f"下载失败: {e}")
    raise

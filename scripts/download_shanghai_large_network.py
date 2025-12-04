"""
下载上海更大范围的OSM路网数据
覆盖多中心订单数据的所有商圈
"""
import osmnx as ox
import networkx as nx
import numpy as np
import json
from pathlib import Path
import time

# 配置OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# 订单数据的坐标范围（从generate_diverse_orders.py的输出）
# Merchant lat range: [30.8894, 31.4107]
# Merchant lng range: [121.1284, 121.8647]

# 扩展边界以确保覆盖所有订单
north = 31.45   # 最北
south = 30.85   # 最南
east = 121.90   # 最东
west = 121.10   # 最西

# 输出目录
OUTPUT_DIR = Path("D:/0On-demand Delivery/data/processed/shanghai_large")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_road_network():
    """下载道路网络"""
    print("=" * 60)
    print("Downloading Shanghai Large-Scale Road Network")
    print("=" * 60)
    
    print(f"\nBounds: N={north}, S={south}, E={east}, W={west}")
    print(f"Coverage: {(north-south)*111:.1f}km × {(east-west)*95:.1f}km")
    
    # 下载道路网络
    print("\n1. Downloading OSM road network...")
    print("   (This may take a few minutes...)")
    
    start_time = time.time()
    
    # bbox格式: (west, south, east, north)
    bbox = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type='drive',  # 可驾驶道路
        simplify=True
    )
    
    download_time = time.time() - start_time
    print(f"   Downloaded in {download_time:.1f} seconds")
    print(f"   Nodes: {G.number_of_nodes():,}")
    print(f"   Edges: {G.number_of_edges():,}")
    
    # 转换为无向图以便于路径规划
    print("\n2. Converting to undirected graph...")
    G_undirected = ox.convert.to_undirected(G)
    
    # 获取最大连通分量
    print("3. Extracting largest connected component...")
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"   Largest component: {G.number_of_nodes():,} nodes")
    
    # 保留完整路网，不进行采样
    # 采样会导致连通性问题
    print(f"\n4. Keeping full graph (no sampling)...")
    print(f"   Full graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    # 保存GraphML
    print("\n5. Saving road network...")
    graphml_path = OUTPUT_DIR / "road_network.graphml"
    ox.save_graphml(G, graphml_path)
    print(f"   Saved: {graphml_path}")
    
    return G


def compute_matrices(G):
    """计算距离和时间矩阵（采样版本，避免内存溢出）"""
    all_nodes = list(G.nodes())
    total_nodes = len(all_nodes)
    
    # 采样节点用于距离矩阵计算
    MAX_SAMPLE = 2000
    if total_nodes > MAX_SAMPLE:
        print(f"\n6. Sampling {MAX_SAMPLE} nodes for distance matrix...")
        print(f"   (Full graph has {total_nodes} nodes, too large for full matrix)")
        
        # 空间均匀采样
        node_coords = np.array([(G.nodes[n]['y'], G.nodes[n]['x']) for n in all_nodes])
        
        # 使用网格采样确保空间均匀
        np.random.seed(42)
        indices = np.random.choice(total_nodes, size=MAX_SAMPLE, replace=False)
        sampled_nodes = [all_nodes[i] for i in indices]
    else:
        sampled_nodes = all_nodes
    
    n = len(sampled_nodes)
    print(f"   Using {n} nodes for distance matrix")
    
    # 创建节点映射
    node_to_idx = {node: idx for idx, node in enumerate(sampled_nodes)}
    idx_to_node = {idx: str(node) for idx, node in enumerate(sampled_nodes)}
    node_list = [str(node) for node in sampled_nodes]
    
    # 初始化距离矩阵
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    
    # 计算最短路径距离
    print(f"\n7. Computing shortest paths...")
    start_time = time.time()
    
    for i, source in enumerate(sampled_nodes):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (n - i - 1) if i > 0 else 0
            print(f"   Progress: {i}/{n} ({i*100//n}%), ETA: {eta:.0f}s")
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='length')
            for target, length in lengths.items():
                if target in node_to_idx:
                    j = node_to_idx[target]
                    dist_matrix[i, j] = length
        except:
            pass
    
    compute_time = time.time() - start_time
    print(f"   Computed in {compute_time:.1f} seconds")
    
    # 保存距离矩阵
    np.save(OUTPUT_DIR / "distance_matrix.npy", dist_matrix)
    print(f"   Saved: distance_matrix.npy ({dist_matrix.shape})")
    
    # 计算时间矩阵（假设平均速度15km/h）
    print("\n8. Computing time matrix...")
    speed_mps = 15 * 1000 / 3600  # 米/秒
    time_matrix = dist_matrix / speed_mps
    np.save(OUTPUT_DIR / "time_matrix.npy", time_matrix)
    print(f"   Saved: time_matrix.npy")
    
    # 保存节点映射
    print("\n9. Saving node mapping...")
    mapping = {
        'node_to_idx': {str(node): idx for idx, node in enumerate(sampled_nodes)},
        'idx_to_node': idx_to_node,
        'node_list': node_list
    }
    with open(OUTPUT_DIR / "node_id_mapping.json", 'w') as f:
        json.dump(mapping, f)
    print(f"   Saved: node_id_mapping.json ({len(sampled_nodes)} nodes)")
    
    return dist_matrix, time_matrix


def verify_coverage(G):
    """验证路网覆盖范围"""
    print("\n10. Verifying coverage...")
    
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lngs = [G.nodes[n]['x'] for n in G.nodes()]
    
    print(f"   Road network bounds:")
    print(f"     lat: [{min(lats):.4f}, {max(lats):.4f}]")
    print(f"     lng: [{min(lngs):.4f}, {max(lngs):.4f}]")
    print(f"   Coverage: {(max(lats)-min(lats))*111:.1f}km x {(max(lngs)-min(lngs))*95:.1f}km")
    
    # 检查订单覆盖
    orders_bounds = {
        'lat_min': 30.8894, 'lat_max': 31.4107,
        'lng_min': 121.1284, 'lng_max': 121.8647
    }
    
    coverage_ok = (
        min(lats) <= orders_bounds['lat_min'] and
        max(lats) >= orders_bounds['lat_max'] and
        min(lngs) <= orders_bounds['lng_min'] and
        max(lngs) >= orders_bounds['lng_max']
    )
    
    if coverage_ok:
        print("   [OK] Road network covers all order locations")
    else:
        print("   [WARNING] Road network may not cover all order locations")


def main():
    try:
        # 下载路网
        G = download_road_network()
        
        # 计算矩阵
        compute_matrices(G)
        
        # 验证覆盖
        verify_coverage(G)
        
        print("\n" + "=" * 60)
        print("Done! Shanghai large-scale road network downloaded.")
        print(f"Output directory: {OUTPUT_DIR}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

"""
重新计算上海路网的完整距离矩阵（不采样）
确保所有订单节点都在矩阵中
"""

import sys
import numpy as np
import networkx as nx
import json
import pickle
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 路径配置
SHANGHAI_DIR = project_root / "data" / "processed" / "shanghai"
ROAD_NETWORK = SHANGHAI_DIR / "road_network.graphml"


def load_graph():
    """加载路网"""
    print(f"Loading road network from {ROAD_NETWORK}...")
    G = nx.read_graphml(str(ROAD_NETWORK))
    
    # 转换边属性为数值类型
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            try:
                data['length'] = float(data['length'])
            except (ValueError, TypeError):
                data['length'] = 100.0  # 默认100米
        else:
            data['length'] = 100.0
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def calculate_full_distance_matrix(G):
    """计算完整距离矩阵（所有节点）"""
    nodes = list(G.nodes())
    n = len(nodes)
    
    print(f"\nCalculating full distance matrix ({n}x{n})...")
    print(f"  This will take a few minutes...")
    
    # 创建节点映射
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    
    # 初始化矩阵
    distance_matrix = np.full((n, n), np.inf)
    time_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(distance_matrix, 0)
    np.fill_diagonal(time_matrix, 0)
    
    # 默认速度 15 km/h
    default_speed = 15.0
    
    # 使用Dijkstra计算最短路径
    for i, source in enumerate(tqdm(nodes, desc="Computing distances")):
        try:
            # 计算从源节点到所有节点的最短路径（距离）
            lengths = nx.single_source_dijkstra_path_length(
                G, source, weight='length'
            )
            
            for target, length in lengths.items():
                if target in node_to_idx:
                    j = node_to_idx[target]
                    distance_matrix[i, j] = length
                    # 时间 = 距离(m) / 速度(km/h) * 3.6
                    time_matrix[i, j] = (length / 1000) / default_speed * 3600
                    
        except Exception as e:
            print(f"  Warning: Error at node {source}: {e}")
            continue
    
    # ================== 【关键修复】替换无穷大值 ==================
    # 问题：如果路网不连通（有孤立节点），两点间距离会保持 np.inf
    # 后果：OR-Tools 尝试将 inf 转为整数时会导致 OverflowError 或 C++ 层崩溃
    # 解决：将 inf 替换为一个非常大的有限数字（惩罚值）
    print("\n  Post-processing: Replacing infinite distances with penalty...")
    inf_count = np.isinf(distance_matrix).sum()
    if inf_count > 0:
        print(f"  Found {inf_count} unreachable pairs! Replacing with 1,000,000m.")
        # 将无穷大替换为 1,000,000m (1000km)，作为不可达的高额惩罚
        distance_matrix[np.isinf(distance_matrix)] = 1000000.0
        # 时间也做相应替换：假设极慢速度，设为24小时
        time_matrix[np.isinf(time_matrix)] = 24 * 3600.0  # 86400秒
        print(f"  Replaced {inf_count} infinite values with penalty values.")
    else:
        print(f"  No infinite values found. Road network is fully connected.")
    # =============================================================
    
    return distance_matrix, time_matrix, node_to_idx, idx_to_node, nodes


def save_matrices(distance_matrix, time_matrix, node_to_idx, idx_to_node, nodes):
    """保存矩阵和映射"""
    print(f"\nSaving matrices to {SHANGHAI_DIR}...")
    
    # 保存距离矩阵
    np.save(SHANGHAI_DIR / "distance_matrix.npy", distance_matrix)
    print(f"  Saved: distance_matrix.npy ({distance_matrix.shape})")
    
    # 保存时间矩阵
    np.save(SHANGHAI_DIR / "time_matrix.npy", time_matrix)
    print(f"  Saved: time_matrix.npy ({time_matrix.shape})")
    
    # 保存节点映射
    mapping_data = {
        'node_to_idx': {str(k): v for k, v in node_to_idx.items()},
        'idx_to_node': {str(k): v for k, v in idx_to_node.items()},
        'num_nodes': len(nodes)
    }
    with open(SHANGHAI_DIR / "node_id_mapping.json", 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"  Saved: node_id_mapping.json ({len(nodes)} nodes)")
    
    # 保存节点列表
    with open(SHANGHAI_DIR / "selected_nodes.pkl", 'wb') as f:
        pickle.dump(nodes, f)
    print(f"  Saved: selected_nodes.pkl")


def verify_coverage(node_to_idx):
    """验证订单节点覆盖率"""
    import pandas as pd
    
    orders_file = project_root / "data" / "orders" / "uniform_grid_100.csv"
    if not orders_file.exists():
        print("\n  Order file not found, skipping verification.")
        return
    
    df = pd.read_csv(orders_file)
    order_nodes = set(df['merchant_node'].astype(str)) | set(df['customer_node'].astype(str))
    matrix_nodes = set(node_to_idx.keys())
    
    covered = order_nodes & matrix_nodes
    missing = order_nodes - matrix_nodes
    
    print(f"\n  Order node coverage verification:")
    print(f"    Order nodes: {len(order_nodes)}")
    print(f"    Covered: {len(covered)} ({len(covered)/len(order_nodes)*100:.1f}%)")
    print(f"    Missing: {len(missing)}")
    
    if missing:
        print(f"    Warning: {len(missing)} nodes still missing!")
    else:
        print(f"    [OK] All order nodes are in the distance matrix!")


def main():
    print("=" * 70)
    print("Recalculating Full Shanghai Distance Matrix")
    print("=" * 70)
    
    # 加载路网
    G = load_graph()
    
    # 计算完整矩阵
    dist_matrix, time_matrix, node_to_idx, idx_to_node, nodes = calculate_full_distance_matrix(G)
    
    # 统计
    finite_dist = dist_matrix[np.isfinite(dist_matrix)]
    print(f"\n  Matrix statistics:")
    print(f"    Shape: {dist_matrix.shape}")
    print(f"    Finite distances: {len(finite_dist)}")
    print(f"    Mean distance: {finite_dist.mean():.0f} m")
    print(f"    Max distance: {finite_dist.max():.0f} m")
    
    # 保存
    save_matrices(dist_matrix, time_matrix, node_to_idx, idx_to_node, nodes)
    
    # 验证覆盖率
    verify_coverage(node_to_idx)
    
    print("\n" + "=" * 70)
    print("Done! Full distance matrix calculated.")
    print("=" * 70)


if __name__ == "__main__":
    main()

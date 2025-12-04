"""
为上海小范围路网生成多中心分布的订单数据
解决当前 lade_shanghai_matched_*.csv 的单中心问题

策略：
1. 从路网文件读取实际地理边界
2. 筛选边界内的原始LaDe订单
3. K-Means聚类识别小范围内的微型商圈
4. 分层采样确保空间多样性
5. 重新分配到达时间
"""
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt

# 配置
INPUT_FILE = Path("D:/0On-demand Delivery/data/lade/delivery/delivery_sh.csv")
ROAD_NETWORK_FILE = Path("D:/0On-demand Delivery/data/processed/shanghai/road_network.graphml")
OUTPUT_DIR = Path("D:/0On-demand Delivery/data/orders")
FIGURE_DIR = Path("D:/0On-demand Delivery/docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 目标订单数量
TARGET_ORDERS = [100, 500, 1000]

# 聚类数量（小范围适合较少聚类）
NUM_CLUSTERS = 5


def load_road_network_bounds():
    """从路网文件获取地理边界"""
    print("Loading road network to get bounds...")
    G = nx.read_graphml(ROAD_NETWORK_FILE)
    
    lats = [float(G.nodes[n]['y']) for n in G.nodes()]
    lngs = [float(G.nodes[n]['x']) for n in G.nodes()]
    
    bounds = {
        'lat_min': min(lats),
        'lat_max': max(lats),
        'lng_min': min(lngs),
        'lng_max': max(lngs)
    }
    
    print(f"  Road network bounds:")
    print(f"    lat: [{bounds['lat_min']:.4f}, {bounds['lat_max']:.4f}]")
    print(f"    lng: [{bounds['lng_min']:.4f}, {bounds['lng_max']:.4f}]")
    print(f"  Coverage: {(bounds['lat_max']-bounds['lat_min'])*111:.2f}km x "
          f"{(bounds['lng_max']-bounds['lng_min'])*95:.2f}km")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    return bounds, G


def haversine_distance(lat1, lng1, lat2, lng2):
    """计算两点间的Haversine距离（米）"""
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def parse_time(time_str):
    """解析时间字符串为秒数"""
    try:
        parts = time_str.split(' ')
        time_part = parts[1] if len(parts) > 1 else parts[0]
        h, m, s = map(int, time_part.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return None


def load_and_filter_data(bounds):
    """加载并过滤数据到路网边界内"""
    print("\nLoading LaDe Shanghai data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Total records: {len(df):,}")
    
    # 扩展边界10%以确保边缘订单可用
    lat_margin = (bounds['lat_max'] - bounds['lat_min']) * 0.1
    lng_margin = (bounds['lng_max'] - bounds['lng_min']) * 0.1
    
    expanded_bounds = {
        'lat_min': bounds['lat_min'] - lat_margin,
        'lat_max': bounds['lat_max'] + lat_margin,
        'lng_min': bounds['lng_min'] - lng_margin,
        'lng_max': bounds['lng_max'] + lng_margin
    }
    
    # 筛选商家在边界内的订单
    mask_merchant = (
        (df['accept_gps_lat'] >= expanded_bounds['lat_min']) & 
        (df['accept_gps_lat'] <= expanded_bounds['lat_max']) & 
        (df['accept_gps_lng'] >= expanded_bounds['lng_min']) & 
        (df['accept_gps_lng'] <= expanded_bounds['lng_max'])
    )
    
    # 筛选客户也在边界内的订单（确保配送可达）
    mask_customer = (
        (df['lat'] >= expanded_bounds['lat_min']) & 
        (df['lat'] <= expanded_bounds['lat_max']) & 
        (df['lng'] >= expanded_bounds['lng_min']) & 
        (df['lng'] <= expanded_bounds['lng_max'])
    )
    
    df = df[mask_merchant & mask_customer].copy()
    print(f"  Orders in bounds (merchant & customer): {len(df):,}")
    
    if len(df) < 100:
        raise ValueError(f"Too few orders in bounds: {len(df)}. Need at least 100.")
    
    # 解析时间
    df['arrival_seconds'] = df['accept_time'].apply(parse_time)
    df['delivery_seconds'] = df['delivery_time'].apply(parse_time)
    df = df.dropna(subset=['arrival_seconds', 'delivery_seconds'])
    
    # 计算实际配送时间
    df['actual_delivery_time'] = df['delivery_seconds'] - df['arrival_seconds']
    df.loc[df['actual_delivery_time'] < 0, 'actual_delivery_time'] += 86400
    
    # 过滤合理配送时间（5分钟到2小时）
    time_mask = (df['actual_delivery_time'] >= 300) & (df['actual_delivery_time'] <= 7200)
    df = df[time_mask]
    
    # 计算配送距离
    df['distance'] = haversine_distance(
        df['accept_gps_lat'].values, df['accept_gps_lng'].values,
        df['lat'].values, df['lng'].values
    )
    
    # 过滤合理距离（200米到5公里，小范围路网适合短距离）
    distance_mask = (df['distance'] >= 200) & (df['distance'] <= 5000)
    df = df[distance_mask]
    print(f"  After time/distance filter: {len(df):,}")
    
    return df


def cluster_merchants(df, n_clusters=NUM_CLUSTERS):
    """对商家位置进行聚类"""
    print(f"\nClustering merchants into {n_clusters} micro-zones...")
    
    merchants = df[['accept_gps_lat', 'accept_gps_lng']].values
    
    # 如果数据太少，减少聚类数
    actual_clusters = min(n_clusters, len(df) // 20)
    actual_clusters = max(actual_clusters, 2)  # 至少2个聚类
    
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(merchants)
    df['cluster'] = clusters
    
    print(f"  Using {actual_clusters} clusters")
    cluster_stats = df.groupby('cluster').agg({
        'order_id': 'count',
        'accept_gps_lat': 'mean',
        'accept_gps_lng': 'mean'
    }).rename(columns={'order_id': 'count'})
    
    print("  Cluster distribution:")
    for cluster_id, row in cluster_stats.iterrows():
        print(f"    Zone {cluster_id}: {int(row['count']):,} orders, "
              f"center ({row['accept_gps_lat']:.5f}, {row['accept_gps_lng']:.5f})")
    
    return df, kmeans


def stratified_sample(df, n_orders, kmeans):
    """分层采样，确保每个商圈都有代表"""
    print(f"\nStratified sampling {n_orders} orders...")
    
    n_clusters = kmeans.n_clusters
    cluster_sizes = df.groupby('cluster').size()
    total_size = len(df)
    
    # 最小每个簇采样数量
    min_per_cluster = max(3, n_orders // (n_clusters * 3))
    remaining = n_orders - min_per_cluster * n_clusters
    
    samples_per_cluster = {}
    for cluster_id in range(n_clusters):
        cluster_count = cluster_sizes.get(cluster_id, 0)
        if cluster_count == 0:
            samples_per_cluster[cluster_id] = 0
            continue
        
        proportion = cluster_count / total_size
        extra = int(remaining * proportion)
        samples_per_cluster[cluster_id] = min(min_per_cluster + extra, cluster_count)
    
    # 调整总数
    total_samples = sum(samples_per_cluster.values())
    if total_samples < n_orders:
        diff = n_orders - total_samples
        largest_cluster = cluster_sizes.idxmax()
        samples_per_cluster[largest_cluster] += diff
    
    print("  Sampling plan:")
    for cluster_id, count in samples_per_cluster.items():
        print(f"    Zone {cluster_id}: {count} orders")
    
    # 执行采样
    sampled_dfs = []
    for cluster_id, n_samples in samples_per_cluster.items():
        if n_samples <= 0:
            continue
        cluster_df = df[df['cluster'] == cluster_id]
        if len(cluster_df) >= n_samples:
            sampled = cluster_df.sample(n=n_samples, random_state=42 + cluster_id)
        else:
            sampled = cluster_df
        sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # 如果还不够，从整体随机采样补充
    if len(result) < n_orders:
        remaining_needed = n_orders - len(result)
        remaining_df = df[~df.index.isin(result.index)]
        if len(remaining_df) >= remaining_needed:
            extra = remaining_df.sample(n=remaining_needed, random_state=99)
            result = pd.concat([result, extra], ignore_index=True)
    
    print(f"  Total sampled: {len(result)}")
    return result


def create_order_dataset(sampled_df, simulation_duration=7200):
    """创建最终订单数据集"""
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df['new_order_id'] = range(1, len(sampled_df) + 1)
    
    n_orders = len(sampled_df)
    
    # 使用泊松过程模拟订单到达
    np.random.seed(42)
    inter_arrival_times = np.random.exponential(
        scale=simulation_duration / (n_orders * 1.2),
        size=n_orders
    )
    arrival_times = np.cumsum(inter_arrival_times)
    arrival_times = arrival_times / arrival_times.max() * (simulation_duration * 0.9)
    
    sampled_df['adjusted_arrival'] = arrival_times
    
    orders = pd.DataFrame({
        'order_id': sampled_df['new_order_id'],
        'arrival_time': sampled_df['adjusted_arrival'].round(0).astype(int),
        'merchant_lat': sampled_df['accept_gps_lat'],
        'merchant_lng': sampled_df['accept_gps_lng'],
        'customer_lat': sampled_df['lat'],
        'customer_lng': sampled_df['lng'],
        'preparation_time': 300,
        'delivery_window': 3600,
        'earliest_pickup_time': sampled_df['adjusted_arrival'].round(0).astype(int) + 300,
        'latest_delivery_time': sampled_df['adjusted_arrival'].round(0).astype(int) + 3600,
        'actual_delivery_time': sampled_df['actual_delivery_time'],
        'distance': sampled_df['distance'],
        'cluster': sampled_df['cluster']
    })
    
    return orders


def visualize_comparison(old_orders, new_orders, bounds, output_path):
    """对比可视化：旧数据 vs 新数据"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：旧数据（单中心）
    axes[0].scatter(
        old_orders['merchant_lng'], old_orders['merchant_lat'],
        c='red', s=50, marker='^', alpha=0.8, edgecolors='white'
    )
    axes[0].scatter(
        old_orders['customer_lng'], old_orders['customer_lat'],
        c='blue', s=30, marker='o', alpha=0.6
    )
    
    # 绘制路网边界
    rect = plt.Rectangle(
        (bounds['lng_min'], bounds['lat_min']),
        bounds['lng_max'] - bounds['lng_min'],
        bounds['lat_max'] - bounds['lat_min'],
        fill=False, edgecolor='green', linewidth=2, linestyle='--'
    )
    axes[0].add_patch(rect)
    
    axes[0].set_xlabel('Longitude', fontsize=11)
    axes[0].set_ylabel('Latitude', fontsize=11)
    axes[0].set_title('Original Data (Single-Center)\nlade_shanghai_matched_100.csv',
                      fontsize=12, fontweight='bold', color='red')
    axes[0].grid(True, alpha=0.3)
    
    # 右图：新数据（多中心）
    scatter = axes[1].scatter(
        new_orders['merchant_lng'], new_orders['merchant_lat'],
        c=new_orders['cluster'], cmap='tab10', s=50, marker='^', 
        alpha=0.8, edgecolors='white'
    )
    axes[1].scatter(
        new_orders['customer_lng'], new_orders['customer_lat'],
        c='blue', s=30, marker='o', alpha=0.4
    )
    
    rect2 = plt.Rectangle(
        (bounds['lng_min'], bounds['lat_min']),
        bounds['lng_max'] - bounds['lng_min'],
        bounds['lat_max'] - bounds['lat_min'],
        fill=False, edgecolor='green', linewidth=2, linestyle='--'
    )
    axes[1].add_patch(rect2)
    
    axes[1].set_xlabel('Longitude', fontsize=11)
    axes[1].set_ylabel('Latitude', fontsize=11)
    axes[1].set_title(f'Optimized Data (Multi-Center)\n{new_orders["cluster"].nunique()} micro-zones',
                      fontsize=12, fontweight='bold', color='green')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Zone ID')
    
    plt.suptitle('Small-Area Order Distribution Optimization\n'
                 '(Shanghai Experimental Area, ~5km x 5km)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("Generating Optimized Orders for Small-Area Road Network")
    print("=" * 70)
    
    # 加载路网边界
    bounds, G = load_road_network_bounds()
    
    # 加载并筛选数据
    df = load_and_filter_data(bounds)
    
    # 聚类
    df, kmeans = cluster_merchants(df, n_clusters=NUM_CLUSTERS)
    
    # 加载旧数据用于对比
    old_orders = pd.read_csv(OUTPUT_DIR / "lade_shanghai_matched_100.csv")
    
    # 为每个目标规模生成数据集
    for n_orders in TARGET_ORDERS:
        if len(df) < n_orders:
            print(f"\n[WARNING] Not enough data for {n_orders} orders. "
                  f"Max available: {len(df)}. Skipping.")
            continue
            
        print(f"\n{'='*70}")
        print(f"Generating {n_orders} orders dataset")
        print("=" * 70)
        
        # 分层采样
        sampled = stratified_sample(df, n_orders, kmeans)
        
        # 创建订单数据集
        orders = create_order_dataset(sampled)
        
        # 保存（使用新命名以区分）
        output_file = OUTPUT_DIR / f"lade_shanghai_small_diverse_{n_orders}.csv"
        orders.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # 统计
        print(f"\n  Statistics:")
        print(f"    Orders: {len(orders)}")
        print(f"    Zones covered: {orders['cluster'].nunique()}")
        print(f"    Merchant lat range: [{orders['merchant_lat'].min():.5f}, {orders['merchant_lat'].max():.5f}]")
        print(f"    Merchant lng range: [{orders['merchant_lng'].min():.5f}, {orders['merchant_lng'].max():.5f}]")
        print(f"    Avg distance: {orders['distance'].mean()/1000:.2f} km")
        
        # 只为100单生成对比图
        if n_orders == 100:
            fig_path = FIGURE_DIR / "small_area_order_optimization.png"
            visualize_comparison(old_orders, orders, bounds, fig_path)
    
    # 打印改进统计
    print("\n" + "=" * 70)
    print("Comparison: Original vs Optimized (100 orders)")
    print("=" * 70)
    
    new_orders = pd.read_csv(OUTPUT_DIR / "lade_shanghai_small_diverse_100.csv")
    
    old_lat_spread = old_orders['merchant_lat'].max() - old_orders['merchant_lat'].min()
    new_lat_spread = new_orders['merchant_lat'].max() - new_orders['merchant_lat'].min()
    old_lng_spread = old_orders['merchant_lng'].max() - old_orders['merchant_lng'].min()
    new_lng_spread = new_orders['merchant_lng'].max() - new_orders['merchant_lng'].min()
    
    print(f"\n| Metric | Original | Optimized | Improvement |")
    print(f"|--------|----------|-----------|-------------|")
    print(f"| Lat spread (m) | {old_lat_spread*111000:.0f} | {new_lat_spread*111000:.0f} | {new_lat_spread/old_lat_spread:.1f}x |")
    print(f"| Lng spread (m) | {old_lng_spread*95000:.0f} | {new_lng_spread*95000:.0f} | {new_lng_spread/old_lng_spread:.1f}x |")
    print(f"| Unique merchants | {len(old_orders.groupby(['merchant_lat', 'merchant_lng']))} | {len(new_orders.groupby(['merchant_lat', 'merchant_lng']))} | - |")
    print(f"| Zones | 1 | {new_orders['cluster'].nunique()} | {new_orders['cluster'].nunique()}x |")
    
    print("\n" + "=" * 70)
    print("Done! Optimized order datasets generated.")
    print("Files: lade_shanghai_small_diverse_*.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()

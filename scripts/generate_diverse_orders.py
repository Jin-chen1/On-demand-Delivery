"""
生成多中心分布的订单数据集

模式切换：
- MODE = 'lade': 从LaDe原始数据生成多中心订单（K-Means聚类）
- MODE = 'uniform': 使用预生成的均匀网格订单数据（Solomon Random分布）
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 模式选择：'uniform' 使用均匀网格数据，'lade' 使用LaDe原始数据
# ============================================================
MODE = 'uniform'  # 切换模式

# 配置
INPUT_FILE = Path("D:/0On-demand Delivery/data/lade/delivery/delivery_sh.csv")
UNIFORM_DIR = Path("D:/0On-demand Delivery/data/orders")  # 均匀网格数据目录
OUTPUT_DIR = Path("D:/0On-demand Delivery/data/orders")
FIGURE_DIR = Path("D:/0On-demand Delivery/docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 目标订单数量
TARGET_ORDERS = [100, 500, 1000]

# 聚类数量（商圈数）- 仅LaDe模式使用
NUM_CLUSTERS = 10


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


def load_and_filter_data():
    """加载并过滤数据"""
    print("Loading LaDe Shanghai data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Total records: {len(df):,}")
    
    # 基本GPS过滤（上海范围）
    valid_mask = (
        (df['lng'] > 121.0) & (df['lng'] < 122.0) &
        (df['lat'] > 30.5) & (df['lat'] < 32.0) &
        (df['accept_gps_lng'] > 121.0) & (df['accept_gps_lng'] < 122.0) &
        (df['accept_gps_lat'] > 30.5) & (df['accept_gps_lat'] < 32.0)
    )
    df = df[valid_mask].copy()
    print(f"  After GPS filter: {len(df):,}")
    
    # 解析时间
    df['arrival_seconds'] = df['accept_time'].apply(parse_time)
    df['delivery_seconds'] = df['delivery_time'].apply(parse_time)
    df = df.dropna(subset=['arrival_seconds', 'delivery_seconds'])
    
    # 计算配送时间
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
    
    # 过滤合理距离（200米到15公里）
    distance_mask = (df['distance'] >= 200) & (df['distance'] <= 15000)
    df = df[distance_mask]
    print(f"  After time/distance filter: {len(df):,}")
    
    return df


def cluster_merchants(df, n_clusters=NUM_CLUSTERS):
    """对商家位置进行聚类"""
    print(f"\nClustering merchants into {n_clusters} zones...")
    
    # 提取商家坐标
    merchants = df[['accept_gps_lat', 'accept_gps_lng']].values
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(merchants)
    df['cluster'] = clusters
    
    # 统计每个簇的订单数
    cluster_stats = df.groupby('cluster').agg({
        'order_id': 'count',
        'accept_gps_lat': 'mean',
        'accept_gps_lng': 'mean'
    }).rename(columns={'order_id': 'count'})
    
    print("Cluster statistics:")
    for cluster_id, row in cluster_stats.iterrows():
        print(f"  Zone {cluster_id}: {int(row['count']):,} orders, "
              f"center ({row['accept_gps_lat']:.4f}, {row['accept_gps_lng']:.4f})")
    
    return df, kmeans


def stratified_sample(df, n_orders, kmeans):
    """分层采样，确保每个商圈都有代表"""
    print(f"\nStratified sampling {n_orders} orders...")
    
    n_clusters = kmeans.n_clusters
    
    # 计算每个簇应该采样的数量（按簇大小比例，但保证每个簇至少有一定数量）
    cluster_sizes = df.groupby('cluster').size()
    total_size = len(df)
    
    # 最小每个簇采样数量
    min_per_cluster = max(5, n_orders // (n_clusters * 3))
    
    # 按比例分配剩余配额
    remaining = n_orders - min_per_cluster * n_clusters
    
    samples_per_cluster = {}
    for cluster_id in range(n_clusters):
        cluster_count = cluster_sizes.get(cluster_id, 0)
        if cluster_count == 0:
            samples_per_cluster[cluster_id] = 0
            continue
        
        # 基础配额 + 按比例分配
        proportion = cluster_count / total_size
        extra = int(remaining * proportion)
        samples_per_cluster[cluster_id] = min(min_per_cluster + extra, cluster_count)
    
    # 调整总数
    total_samples = sum(samples_per_cluster.values())
    if total_samples < n_orders:
        # 从最大的簇补充
        diff = n_orders - total_samples
        largest_cluster = cluster_sizes.idxmax()
        samples_per_cluster[largest_cluster] += diff
    
    print("Sampling plan:")
    for cluster_id, count in samples_per_cluster.items():
        print(f"  Zone {cluster_id}: {count} orders")
    
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
    # 重新编号
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df['new_order_id'] = range(1, len(sampled_df) + 1)
    
    # 重新分配到达时间（在simulation_duration内均匀分布）
    n_orders = len(sampled_df)
    
    # 使用泊松过程模拟订单到达
    np.random.seed(42)
    inter_arrival_times = np.random.exponential(
        scale=simulation_duration / (n_orders * 1.2),  # 稍密一点
        size=n_orders
    )
    arrival_times = np.cumsum(inter_arrival_times)
    
    # 归一化到simulation_duration范围内
    arrival_times = arrival_times / arrival_times.max() * (simulation_duration * 0.9)
    
    sampled_df['adjusted_arrival'] = arrival_times
    
    # 创建输出DataFrame
    orders = pd.DataFrame({
        'order_id': sampled_df['new_order_id'],
        'arrival_time': sampled_df['adjusted_arrival'].round(0).astype(int),
        'merchant_lat': sampled_df['accept_gps_lat'],
        'merchant_lng': sampled_df['accept_gps_lng'],
        'customer_lat': sampled_df['lat'],
        'customer_lng': sampled_df['lng'],
        'preparation_time': 300,  # 5分钟备餐
        'delivery_window': 3600,  # 1小时配送窗口
        'earliest_pickup_time': sampled_df['adjusted_arrival'].round(0).astype(int) + 300,
        'latest_delivery_time': sampled_df['adjusted_arrival'].round(0).astype(int) + 3600,
        'actual_delivery_time': sampled_df['actual_delivery_time'],
        'distance': sampled_df['distance'],
        'cluster': sampled_df['cluster']  # 保留聚类信息用于分析
    })
    
    return orders


def visualize_distribution(orders, output_path, title_suffix=""):
    """可视化订单分布"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 左图：商家分布（按聚类着色）
    scatter1 = axes[0].scatter(
        orders['merchant_lng'], 
        orders['merchant_lat'],
        c=orders['cluster'],
        cmap='tab10',
        s=50,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )
    axes[0].set_xlabel('Longitude', fontsize=11)
    axes[0].set_ylabel('Latitude', fontsize=11)
    axes[0].set_title(f'Merchant Distribution by Zone\n({len(orders)} orders, {orders["cluster"].nunique()} zones)',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Zone ID')
    
    # 右图：商家和客户连线
    for i in range(min(50, len(orders))):  # 只画前50条
        row = orders.iloc[i]
        axes[1].plot(
            [row['merchant_lng'], row['customer_lng']],
            [row['merchant_lat'], row['customer_lat']],
            color='purple', linewidth=0.5, alpha=0.3
        )
    
    axes[1].scatter(
        orders['merchant_lng'], orders['merchant_lat'],
        c='red', s=40, marker='^', label='Merchants', alpha=0.8
    )
    axes[1].scatter(
        orders['customer_lng'], orders['customer_lat'],
        c='blue', s=20, marker='o', label='Customers', alpha=0.6
    )
    axes[1].set_xlabel('Longitude', fontsize=11)
    axes[1].set_ylabel('Latitude', fontsize=11)
    axes[1].set_title(f'Order Distribution (Merchants & Customers){title_suffix}',
                      fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def load_uniform_grid_data(n_orders):
    """加载预生成的均匀网格订单数据"""
    input_file = UNIFORM_DIR / f"uniform_grid_{n_orders}.csv"
    print(f"Loading uniform grid data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} orders")
    return df


def process_uniform_data(df):
    """处理均匀网格数据，添加聚类标签用于可视化"""
    # 使用K-Means对均匀数据进行后标注（仅用于可视化）
    coords = df[['merchant_lat', 'merchant_lng']].values
    
    # 根据数据量选择聚类数
    n_clusters = min(10, len(df) // 10 + 1)
    n_clusters = max(3, n_clusters)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(coords)
    
    # 添加缺失的列（如果需要）
    if 'distance' not in df.columns and 'distance_km' in df.columns:
        df['distance'] = df['distance_km'] * 1000
    
    return df


def main():
    print("=" * 70)
    print(f"Generating Order Dataset (MODE: {MODE})")
    print("=" * 70)
    
    if MODE == 'uniform':
        # 使用均匀网格数据
        print("\nUsing pre-generated uniform grid sampling data...")
        print("Based on Solomon (1987) Random distribution methodology")
        
        for n_orders in TARGET_ORDERS:
            print(f"\n{'='*70}")
            print(f"Processing {n_orders} orders dataset")
            print("=" * 70)
            
            # 加载均匀网格数据
            orders = load_uniform_grid_data(n_orders)
            orders = process_uniform_data(orders)
            
            # 重命名列以匹配项目格式
            if 'distance_km' in orders.columns and 'distance' not in orders.columns:
                orders['distance'] = orders['distance_km'] * 1000
            
            # 保存为diverse格式（覆盖旧文件）
            output_file = OUTPUT_DIR / f"lade_shanghai_diverse_{n_orders}.csv"
            orders.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
            
            # 统计
            print(f"\n  Statistics:")
            print(f"    Orders: {len(orders)}")
            print(f"    Zones (post-labeled): {orders['cluster'].nunique()}")
            print(f"    Merchant lat range: [{orders['merchant_lat'].min():.4f}, {orders['merchant_lat'].max():.4f}]")
            print(f"    Merchant lng range: [{orders['merchant_lng'].min():.4f}, {orders['merchant_lng'].max():.4f}]")
            if 'distance' in orders.columns:
                print(f"    Avg distance: {orders['distance'].mean()/1000:.2f} km")
            elif 'distance_km' in orders.columns:
                print(f"    Avg distance: {orders['distance_km'].mean():.2f} km")
            
            # 可视化
            fig_path = FIGURE_DIR / f"diverse_orders_{n_orders}_distribution.png"
            visualize_distribution(orders, fig_path, f"\n({n_orders} orders, Uniform Grid)")
    
    else:
        # 使用LaDe原始数据（原有逻辑）
        df = load_and_filter_data()
        df, kmeans = cluster_merchants(df, n_clusters=NUM_CLUSTERS)
        
        for n_orders in TARGET_ORDERS:
            print(f"\n{'='*70}")
            print(f"Generating {n_orders} orders dataset")
            print("=" * 70)
            
            sampled = stratified_sample(df, n_orders, kmeans)
            orders = create_order_dataset(sampled)
            
            output_file = OUTPUT_DIR / f"lade_shanghai_diverse_{n_orders}.csv"
            orders.to_csv(output_file, index=False)
            print(f"  Saved: {output_file}")
            
            print(f"\n  Statistics:")
            print(f"    Orders: {len(orders)}")
            print(f"    Zones covered: {orders['cluster'].nunique()}")
            print(f"    Merchant lat range: [{orders['merchant_lat'].min():.4f}, {orders['merchant_lat'].max():.4f}]")
            print(f"    Merchant lng range: [{orders['merchant_lng'].min():.4f}, {orders['merchant_lng'].max():.4f}]")
            print(f"    Avg distance: {orders['distance'].mean()/1000:.2f} km")
            
            fig_path = FIGURE_DIR / f"diverse_orders_{n_orders}_distribution.png"
            visualize_distribution(orders, fig_path, f"\n({n_orders} orders)")
    
    # 数据对比
    print("\n" + "=" * 70)
    print("Data Comparison Summary")
    print("=" * 70)
    
    comparison_data = []
    for n_orders in TARGET_ORDERS:
        # 加载各版本数据
        files = {
            'Original (LaDe)': f"lade_shanghai_matched_{n_orders}.csv",
            'Uniform Grid': f"uniform_grid_{n_orders}.csv",
            'Current Diverse': f"lade_shanghai_diverse_{n_orders}.csv"
        }
        
        print(f"\n{n_orders} Orders:")
        for name, filename in files.items():
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                lat_spread = df['merchant_lat'].max() - df['merchant_lat'].min()
                lng_spread = df['merchant_lng'].max() - df['merchant_lng'].min()
                print(f"  {name:20s}: lat_spread={lat_spread*111:.1f}km, lng_spread={lng_spread*111:.1f}km")
    
    print("\n" + "=" * 70)
    print("Done! Datasets updated with uniform grid distribution.")
    print("=" * 70)


if __name__ == "__main__":
    main()

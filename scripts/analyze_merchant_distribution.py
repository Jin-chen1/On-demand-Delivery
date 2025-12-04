"""
分析LaDe上海数据的商家空间分布
诊断当前单中心问题的原因，为生成多中心订单集提供依据
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
INPUT_FILE = Path("D:/0On-demand Delivery/data/lade/delivery/delivery_sh.csv")
OUTPUT_DIR = Path("D:/0On-demand Delivery/docs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 当前路网范围
BOUNDS = {
    'lat_min': 31.20,
    'lat_max': 31.25,
    'lng_min': 121.42,
    'lng_max': 121.47
}


def analyze_distribution():
    """分析商家分布"""
    print("=" * 60)
    print("LaDe Shanghai Merchant Distribution Analysis")
    print("=" * 60)
    
    # 加载数据
    print(f"\n1. Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Total records: {len(df):,}")
    
    # 检查全局坐标范围
    print(f"\n2. Global coordinate range:")
    print(f"   Merchant lat: [{df['accept_gps_lat'].min():.4f}, {df['accept_gps_lat'].max():.4f}]")
    print(f"   Merchant lng: [{df['accept_gps_lng'].min():.4f}, {df['accept_gps_lng'].max():.4f}]")
    print(f"   Customer lat: [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    print(f"   Customer lng: [{df['lng'].min():.4f}, {df['lng'].max():.4f}]")
    
    # 筛选在路网范围内的订单
    print(f"\n3. Filtering orders within current map bounds...")
    print(f"   Bounds: lat [{BOUNDS['lat_min']}, {BOUNDS['lat_max']}], "
          f"lng [{BOUNDS['lng_min']}, {BOUNDS['lng_max']}]")
    
    mask = (
        (df['accept_gps_lat'] >= BOUNDS['lat_min']) & 
        (df['accept_gps_lat'] <= BOUNDS['lat_max']) & 
        (df['accept_gps_lng'] >= BOUNDS['lng_min']) & 
        (df['accept_gps_lng'] <= BOUNDS['lng_max'])
    )
    
    df_in_bounds = df[mask].copy()
    print(f"   Orders in bounds: {len(df_in_bounds):,} ({len(df_in_bounds)/len(df)*100:.2f}%)")
    
    if len(df_in_bounds) < 100:
        print("\n   WARNING: Too few orders in current map bounds!")
        print("   Expanding bounds to cover more area...")
        
        # 扩大范围
        expanded_bounds = {
            'lat_min': 31.15,
            'lat_max': 31.30,
            'lng_min': 121.35,
            'lng_max': 121.55
        }
        
        mask_expanded = (
            (df['accept_gps_lat'] >= expanded_bounds['lat_min']) & 
            (df['accept_gps_lat'] <= expanded_bounds['lat_max']) & 
            (df['accept_gps_lng'] >= expanded_bounds['lng_min']) & 
            (df['accept_gps_lng'] <= expanded_bounds['lng_max'])
        )
        
        df_in_bounds = df[mask_expanded].copy()
        print(f"   Orders in expanded bounds: {len(df_in_bounds):,}")
        BOUNDS.update(expanded_bounds)
    
    if len(df_in_bounds) < 100:
        print("   ERROR: Still too few orders. Check data file.")
        return None
    
    # 分析商家位置的唯一性
    print(f"\n4. Analyzing merchant location diversity...")
    unique_merchants = df_in_bounds.groupby(
        ['accept_gps_lat', 'accept_gps_lng']
    ).size().reset_index(name='order_count')
    
    print(f"   Unique merchant locations: {len(unique_merchants):,}")
    print(f"   Top 10 busiest merchants:")
    top_merchants = unique_merchants.nlargest(10, 'order_count')
    for i, row in top_merchants.iterrows():
        print(f"     ({row['accept_gps_lat']:.5f}, {row['accept_gps_lng']:.5f}): "
              f"{row['order_count']} orders")
    
    # K-Means聚类分析
    print(f"\n5. K-Means clustering (k=8)...")
    merchants_coords = df_in_bounds[['accept_gps_lat', 'accept_gps_lng']].values
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(merchants_coords)
    df_in_bounds['cluster'] = clusters
    
    print("   Cluster distribution:")
    cluster_counts = df_in_bounds['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        center = kmeans.cluster_centers_[cluster_id]
        print(f"     Cluster {cluster_id}: {count:,} orders, "
              f"center at ({center[0]:.5f}, {center[1]:.5f})")
    
    # 分析"前100单"的问题
    print(f"\n6. Analyzing 'first 100 orders' issue...")
    df_first_100 = df_in_bounds.head(100)
    unique_in_first_100 = df_first_100.groupby(
        ['accept_gps_lat', 'accept_gps_lng']
    ).size().reset_index(name='count')
    
    print(f"   Unique merchant locations in first 100: {len(unique_in_first_100)}")
    
    # 可视化
    print(f"\n7. Generating visualizations...")
    
    # 图1: 全部商家分布（带聚类）
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    scatter1 = axes[0].scatter(
        df_in_bounds['accept_gps_lng'], 
        df_in_bounds['accept_gps_lat'],
        c=df_in_bounds['cluster'], 
        cmap='tab10', 
        s=2, 
        alpha=0.3
    )
    # 标记聚类中心
    for i, center in enumerate(kmeans.cluster_centers_):
        axes[0].scatter(center[1], center[0], c='red', s=100, marker='*', 
                       edgecolors='white', linewidths=1, zorder=5)
        axes[0].annotate(f'C{i}', (center[1], center[0]), fontsize=8, 
                        color='red', fontweight='bold')
    
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title(f'All Merchants with K-Means Clusters (n={len(df_in_bounds):,})')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # 图2: 前100单 vs 随机100单
    df_random_100 = df_in_bounds.sample(n=100, random_state=42)
    
    axes[1].scatter(
        df_in_bounds['accept_gps_lng'], 
        df_in_bounds['accept_gps_lat'],
        c='lightgray', s=1, alpha=0.2, label='All'
    )
    axes[1].scatter(
        df_first_100['accept_gps_lng'], 
        df_first_100['accept_gps_lat'],
        c='red', s=30, marker='^', label='First 100 (sequential)'
    )
    axes[1].scatter(
        df_random_100['accept_gps_lng'], 
        df_random_100['accept_gps_lat'],
        c='blue', s=30, marker='o', alpha=0.7, label='Random 100'
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('First 100 vs Random 100 Sampling')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "merchant_distribution_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")
    
    # 返回分析结果
    return {
        'df': df_in_bounds,
        'kmeans': kmeans,
        'bounds': BOUNDS,
        'unique_merchants': len(unique_merchants),
        'total_orders': len(df_in_bounds)
    }


def main():
    result = analyze_distribution()
    
    if result:
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        print(f"Total orders in bounds: {result['total_orders']:,}")
        print(f"Unique merchant locations: {result['unique_merchants']:,}")
        print(f"Number of clusters: {result['kmeans'].n_clusters}")
        print("\nRECOMMENDATION:")
        print("  Use stratified sampling from each cluster to create")
        print("  a diverse multi-center order dataset.")
        print("=" * 60)


if __name__ == "__main__":
    main()

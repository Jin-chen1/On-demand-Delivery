"""
绘制均匀网格订单分布图
背景为实际路网，显示商家和客户分布，带热力图
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.ndimage import gaussian_filter

# 配置
PROJECT_ROOT = Path("D:/0On-demand Delivery")
NETWORK_PATH = PROJECT_ROOT / "data/processed/shanghai/road_network.graphml"
ORDER_FILE = PROJECT_ROOT / "data/orders/uniform_grid_100.csv"
OUTPUT_PATH = PROJECT_ROOT / "docs/figures/uniform_grid_road_network.png"


def load_road_network(path):
    """加载路网"""
    print(f"Loading road network from {path}...")
    G = nx.read_graphml(path)
    
    # 转换坐标
    for node in G.nodes():
        G.nodes[node]['x'] = float(G.nodes[node].get('x', 0))
        G.nodes[node]['y'] = float(G.nodes[node].get('y', 0))
    
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def create_heatmap(lats, lngs, bounds, resolution=100):
    """创建热力图数据"""
    # 创建网格
    x_bins = np.linspace(bounds['lng_min'], bounds['lng_max'], resolution)
    y_bins = np.linspace(bounds['lat_min'], bounds['lat_max'], resolution)
    
    # 统计密度
    heatmap, _, _ = np.histogram2d(lngs, lats, bins=[x_bins, y_bins])
    
    # 高斯平滑
    heatmap = gaussian_filter(heatmap, sigma=3)
    
    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap, x_bins, y_bins


def plot_distribution(G, orders_df, output_path):
    """绘制分布图"""
    print("Creating visualization...")
    
    # 获取路网边界
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lngs = [G.nodes[n]['x'] for n in G.nodes()]
    
    bounds = {
        'lat_min': min(lats),
        'lat_max': max(lats),
        'lng_min': min(lngs),
        'lng_max': max(lngs)
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 绘制路网边（灰色背景）
    print("  Drawing road network...")
    for u, v, data in G.edges(data=True):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        ax.plot([x1, x2], [y1, y2], color='#888888', linewidth=0.3, alpha=0.5, zorder=1)
    
    # 2. 绘制热力图（商家密度）
    print("  Creating heatmap...")
    merchant_lats = orders_df['merchant_lat'].values
    merchant_lngs = orders_df['merchant_lng'].values
    
    heatmap, x_bins, y_bins = create_heatmap(merchant_lats, merchant_lngs, bounds, resolution=80)
    
    # 自定义colormap（白色到橙色）
    colors = ['#FFFFFF00', '#FFF5EB', '#FED8B1', '#FDB863', '#E66100']
    cmap = LinearSegmentedColormap.from_list('orange_heat', colors)
    
    # 绘制热力图
    extent = [bounds['lng_min'], bounds['lng_max'], bounds['lat_min'], bounds['lat_max']]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, 
                   alpha=0.7, aspect='auto', zorder=2)
    
    # 3. 绘制客户点（蓝色圆点）
    print("  Plotting customers...")
    ax.scatter(orders_df['customer_lng'], orders_df['customer_lat'],
               c='#1E90FF', s=30, marker='o', alpha=0.8, 
               edgecolors='white', linewidths=0.3,
               label=f'Customers (n={len(orders_df)})', zorder=4)
    
    # 4. 绘制商家点（红色三角）
    print("  Plotting merchants...")
    # 获取唯一商家
    unique_merchants = orders_df.drop_duplicates(subset=['merchant_node'])[['merchant_lat', 'merchant_lng']]
    ax.scatter(unique_merchants['merchant_lng'], unique_merchants['merchant_lat'],
               c='#FF4444', s=80, marker='^', alpha=0.9,
               edgecolors='white', linewidths=0.5,
               label=f'Merchants (n={len(unique_merchants)})', zorder=5)
    
    # 5. 添加colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Order Density', fontsize=11)
    
    # 6. 设置标签和标题
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Shanghai Road Network with Order Distribution\n'
                 f'(Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, '
                 f'Orders: {len(orders_df)})',
                 fontsize=13, fontweight='bold')
    
    # 7. 添加图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 8. 添加区域标注
    lat_range = (bounds['lat_max'] - bounds['lat_min']) * 111
    lng_range = (bounds['lng_max'] - bounds['lng_min']) * 111 * np.cos(np.radians(np.mean(lats)))
    ax.text(bounds['lng_min'] + 0.002, bounds['lat_min'] + 0.002,
            f'Approx. area: {lng_range:.1f}km × {lat_range:.1f}km',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=10)
    
    # 9. 设置坐标轴范围（留边距）
    margin = 0.002
    ax.set_xlim(bounds['lng_min'] - margin, bounds['lng_max'] + margin)
    ax.set_ylim(bounds['lat_min'] - margin, bounds['lat_max'] + margin)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    print("=" * 70)
    print("Plotting Uniform Grid Order Distribution on Road Network")
    print("=" * 70)
    
    # 加载路网
    G = load_road_network(NETWORK_PATH)
    
    # 加载订单数据
    print(f"\nLoading orders from {ORDER_FILE}...")
    orders_df = pd.read_csv(ORDER_FILE)
    print(f"  Loaded {len(orders_df)} orders")
    
    # 绘制分布图
    plot_distribution(G, orders_df, OUTPUT_PATH)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

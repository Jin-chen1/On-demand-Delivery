"""
绘制上海实验区域路网示意图，并叠加商家与客户分布热力图

输出：
1. road_network_overview.png - 路网结构图
2. merchant_customer_heatmap.png - 商家与客户分布热力图
3. road_network_with_orders.png - 路网+订单分布组合图
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.ndimage import gaussian_filter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件
# 使用大范围路网和多中心订单数据
ROAD_NETWORK_FILE = DATA_DIR / "processed" / "shanghai_large" / "road_network.graphml"
ORDERS_FILE = DATA_DIR / "orders" / "lade_shanghai_diverse_100.csv"


def load_road_network():
    """加载上海路网"""
    print(f"Loading road network from {ROAD_NETWORK_FILE}")
    G = nx.read_graphml(ROAD_NETWORK_FILE)
    
    # 提取节点坐标
    nodes_x = []
    nodes_y = []
    node_ids = []
    
    for node_id, data in G.nodes(data=True):
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        nodes_x.append(x)
        nodes_y.append(y)
        node_ids.append(node_id)
    
    # 提取边
    edges = []
    for u, v, data in G.edges(data=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        edges.append({
            'x1': float(u_data.get('x', 0)),
            'y1': float(u_data.get('y', 0)),
            'x2': float(v_data.get('x', 0)),
            'y2': float(v_data.get('y', 0))
        })
    
    print(f"  Nodes: {len(node_ids)}, Edges: {len(edges)}")
    
    return {
        'nodes_x': np.array(nodes_x),
        'nodes_y': np.array(nodes_y),
        'node_ids': node_ids,
        'edges': edges,
        'graph': G
    }


def load_orders():
    """加载订单数据"""
    print(f"Loading orders from {ORDERS_FILE}")
    df = pd.read_csv(ORDERS_FILE)
    
    merchants = df[['merchant_lat', 'merchant_lng']].values
    customers = df[['customer_lat', 'customer_lng']].values
    
    print(f"  Orders: {len(df)}")
    print(f"  Merchant range: lat [{merchants[:, 0].min():.4f}, {merchants[:, 0].max():.4f}], "
          f"lng [{merchants[:, 1].min():.4f}, {merchants[:, 1].max():.4f}]")
    print(f"  Customer range: lat [{customers[:, 0].min():.4f}, {customers[:, 0].max():.4f}], "
          f"lng [{customers[:, 1].min():.4f}, {customers[:, 1].max():.4f}]")
    
    return {
        'merchants': merchants,
        'customers': customers,
        'df': df
    }


def create_heatmap(points, bounds, resolution=100):
    """
    创建热力图数据
    
    Args:
        points: (N, 2) 坐标数组 [lat, lng]
        bounds: (lat_min, lat_max, lng_min, lng_max)
        resolution: 网格分辨率
    
    Returns:
        heatmap: 2D热力图数组
        extent: [lng_min, lng_max, lat_min, lat_max] for imshow
    """
    lat_min, lat_max, lng_min, lng_max = bounds
    
    # 创建网格
    heatmap = np.zeros((resolution, resolution))
    
    for lat, lng in points:
        # 计算网格位置
        i = int((lat - lat_min) / (lat_max - lat_min) * (resolution - 1))
        j = int((lng - lng_min) / (lng_max - lng_min) * (resolution - 1))
        
        # 边界检查
        i = max(0, min(i, resolution - 1))
        j = max(0, min(j, resolution - 1))
        
        heatmap[i, j] += 1
    
    # 高斯平滑
    heatmap = gaussian_filter(heatmap, sigma=3)
    
    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    extent = [lng_min, lng_max, lat_min, lat_max]
    return heatmap, extent


def plot_road_network_overview(road_data, output_path):
    """
    绘制路网概览图
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制边（道路）
    for edge in road_data['edges']:
        ax.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                color='#4A90A4', linewidth=0.5, alpha=0.6)
    
    # 绘制节点
    ax.scatter(road_data['nodes_x'], road_data['nodes_y'], 
               s=3, c='#2C5F7C', alpha=0.8, zorder=2)
    
    # 设置坐标轴
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Shanghai Experimental Area Road Network\n'
                 f'(Nodes: {len(road_data["node_ids"])}, Edges: {len(road_data["edges"])})', 
                 fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_merchant_customer_heatmap(orders_data, bounds, output_path):
    """
    绘制商家与客户分布热力图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    resolution = 80
    
    # 商家热力图
    merchant_heatmap, extent = create_heatmap(orders_data['merchants'], bounds, resolution)
    
    # 自定义颜色映射（白色到红色）
    colors_merchant = ['white', '#FFE5E5', '#FF9999', '#FF4444', '#CC0000', '#800000']
    cmap_merchant = LinearSegmentedColormap.from_list('merchant', colors_merchant)
    
    im1 = axes[0].imshow(merchant_heatmap, extent=extent, origin='lower', 
                         cmap=cmap_merchant, alpha=0.8, aspect='auto')
    axes[0].scatter(orders_data['merchants'][:, 1], orders_data['merchants'][:, 0], 
                    s=30, c='red', marker='^', edgecolors='white', linewidths=0.5,
                    label='Merchants', zorder=3)
    axes[0].set_xlabel('Longitude', fontsize=11)
    axes[0].set_ylabel('Latitude', fontsize=11)
    axes[0].set_title('Merchant Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0], label='Density', shrink=0.8)
    
    # 客户热力图
    customer_heatmap, extent = create_heatmap(orders_data['customers'], bounds, resolution)
    
    # 自定义颜色映射（白色到蓝色）
    colors_customer = ['white', '#E5F0FF', '#99C2FF', '#4488FF', '#0044CC', '#002280']
    cmap_customer = LinearSegmentedColormap.from_list('customer', colors_customer)
    
    im2 = axes[1].imshow(customer_heatmap, extent=extent, origin='lower', 
                         cmap=cmap_customer, alpha=0.8, aspect='auto')
    axes[1].scatter(orders_data['customers'][:, 1], orders_data['customers'][:, 0], 
                    s=30, c='blue', marker='o', edgecolors='white', linewidths=0.5,
                    label='Customers', zorder=3)
    axes[1].set_xlabel('Longitude', fontsize=11)
    axes[1].set_ylabel('Latitude', fontsize=11)
    axes[1].set_title('Customer Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[1], label='Density', shrink=0.8)
    
    plt.suptitle('Spatial Distribution of Merchants and Customers\n'
                 '(Cainiao-AI LaDe Shanghai Dataset, Multi-Center 100 Orders)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_network_orders(road_data, orders_data, bounds, output_path):
    """
    绘制路网+订单分布组合图
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 1. 绘制路网边（道路）- 浅灰色背景
    for edge in road_data['edges']:
        ax.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                color='#CCCCCC', linewidth=0.4, alpha=0.5, zorder=1)
    
    # 2. 绘制路网节点 - 非常小的点
    ax.scatter(road_data['nodes_x'], road_data['nodes_y'], 
               s=1, c='#999999', alpha=0.3, zorder=2)
    
    # 3. 创建组合热力图
    all_points = np.vstack([orders_data['merchants'], orders_data['customers']])
    heatmap, extent = create_heatmap(all_points, bounds, resolution=100)
    
    # 热力图颜色（透明到橙色）
    colors_heat = [(1, 1, 1, 0), (1, 0.9, 0.7, 0.3), (1, 0.7, 0.3, 0.5), 
                   (1, 0.4, 0, 0.7), (0.8, 0.2, 0, 0.9)]
    cmap_heat = LinearSegmentedColormap.from_list('heat', colors_heat)
    
    im = ax.imshow(heatmap, extent=extent, origin='lower', 
                   cmap=cmap_heat, alpha=0.7, aspect='auto', zorder=3)
    
    # 4. 绘制商家位置（红色三角）
    ax.scatter(orders_data['merchants'][:, 1], orders_data['merchants'][:, 0], 
               s=60, c='#E74C3C', marker='^', edgecolors='white', linewidths=1,
               label=f'Merchants (n={len(orders_data["merchants"])})', zorder=5)
    
    # 5. 绘制客户位置（蓝色圆点）
    ax.scatter(orders_data['customers'][:, 1], orders_data['customers'][:, 0], 
               s=40, c='#3498DB', marker='o', edgecolors='white', linewidths=0.8,
               label=f'Customers (n={len(orders_data["customers"])})', zorder=4)
    
    # 6. 绘制商家-客户连线（表示订单）
    for i in range(min(20, len(orders_data['merchants']))):  # 只画前20条避免太乱
        m = orders_data['merchants'][i]
        c = orders_data['customers'][i]
        ax.plot([m[1], c[1]], [m[0], c[0]], 
                color='#9B59B6', linewidth=0.8, alpha=0.3, linestyle='--', zorder=3)
    
    # 设置坐标轴
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Shanghai Road Network with Order Distribution\n'
                 f'(Nodes: {len(road_data["node_ids"])}, Edges: {len(road_data["edges"])}, '
                 f'Orders: {len(orders_data["df"])})', 
                 fontsize=14, fontweight='bold')
    
    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label='Order Density')
    
    # 网格
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal')
    
    # 添加比例尺说明
    # 计算大致范围
    lng_range = road_data['nodes_x'].max() - road_data['nodes_x'].min()
    lat_range = road_data['nodes_y'].max() - road_data['nodes_y'].min()
    
    # 1度经度约111km * cos(31°) ≈ 95km
    # 1度纬度约111km
    km_lng = lng_range * 95
    km_lat = lat_range * 111
    
    ax.text(0.02, 0.02, f'Approx. area: {km_lng:.1f}km × {km_lat:.1f}km', 
            transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("Road Network and Order Distribution Visualization")
    print("=" * 60)
    
    # 加载数据
    road_data = load_road_network()
    orders_data = load_orders()
    
    # 计算边界
    all_lats = np.concatenate([
        road_data['nodes_y'],
        orders_data['merchants'][:, 0],
        orders_data['customers'][:, 0]
    ])
    all_lngs = np.concatenate([
        road_data['nodes_x'],
        orders_data['merchants'][:, 1],
        orders_data['customers'][:, 1]
    ])
    
    # 扩展边界10%
    lat_margin = (all_lats.max() - all_lats.min()) * 0.1
    lng_margin = (all_lngs.max() - all_lngs.min()) * 0.1
    
    bounds = (
        all_lats.min() - lat_margin,
        all_lats.max() + lat_margin,
        all_lngs.min() - lng_margin,
        all_lngs.max() + lng_margin
    )
    
    print(f"\nBounds: lat [{bounds[0]:.4f}, {bounds[1]:.4f}], "
          f"lng [{bounds[2]:.4f}, {bounds[3]:.4f}]")
    
    # 绘制图表
    print("\nGenerating figures...")
    
    # 1. 路网概览图
    plot_road_network_overview(
        road_data, 
        OUTPUT_DIR / "road_network_overview.png"
    )
    
    # 2. 商家与客户分布热力图
    plot_merchant_customer_heatmap(
        orders_data, 
        bounds,
        OUTPUT_DIR / "merchant_customer_heatmap.png"
    )
    
    # 3. 路网+订单分布组合图
    plot_combined_network_orders(
        road_data, 
        orders_data, 
        bounds,
        OUTPUT_DIR / "road_network_with_orders.png"
    )
    
    print("\n" + "=" * 60)
    print("All figures saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()

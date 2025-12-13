"""
Fig 1. Spatio-Temporal Supply-Demand Dynamics Visualization
时空供需动态图 - 地图热力图 + 时间轴波峰

用于论文展示：
1. 左上：商家分布热力图 (Merchant Distribution Heatmap)
2. 右上：客户分布热力图 (Customer Distribution Heatmap)
3. 下方：订单到达时间分布 (Order Arrival Pattern with Peak Hours)

数据来源：使用非齐次泊松过程生成的订单数据
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import gaussian_filter
from datetime import datetime

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件配置
ROAD_NETWORK_FILE = DATA_DIR / "processed" / "shanghai" / "road_network.graphml"
ORDERS_FILE = DATA_DIR / "orders" / "uniform_grid_100.csv"

# 备选数据文件（如果主文件不存在）
BACKUP_NETWORK_FILE = DATA_DIR / "processed" / "shanghai_large" / "road_network.graphml"
BACKUP_ORDERS_FILE = DATA_DIR / "orders" / "lade_shanghai_diverse_100.csv"


def load_road_network(network_path: Path) -> dict:
    """加载路网数据"""
    print(f"Loading road network from {network_path}...")
    G = nx.read_graphml(network_path)
    
    # 提取节点坐标
    nodes_x = []
    nodes_y = []
    
    for node_id, data in G.nodes(data=True):
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        nodes_x.append(x)
        nodes_y.append(y)
    
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
    
    print(f"  Nodes: {len(nodes_x)}, Edges: {len(edges)}")
    
    return {
        'nodes_x': np.array(nodes_x),
        'nodes_y': np.array(nodes_y),
        'edges': edges,
        'graph': G
    }


def load_orders(orders_path: Path) -> dict:
    """加载订单数据"""
    print(f"Loading orders from {orders_path}...")
    df = pd.read_csv(orders_path)
    
    # 提取商家和客户坐标
    merchants = df[['merchant_lat', 'merchant_lng']].values
    customers = df[['customer_lat', 'customer_lng']].values
    
    # 提取到达时间
    arrival_times = df['arrival_time'].values
    
    print(f"  Orders: {len(df)}")
    print(f"  Time range: {arrival_times.min():.0f}s - {arrival_times.max():.0f}s")
    print(f"  Time range: {arrival_times.min()/3600:.1f}h - {arrival_times.max()/3600:.1f}h")
    
    return {
        'merchants': merchants,
        'customers': customers,
        'arrival_times': arrival_times,
        'df': df
    }


def create_heatmap(points: np.ndarray, bounds: tuple, resolution: int = 80) -> tuple:
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


def plot_fig1_spatiotemporal(road_data: dict, orders_data: dict, output_path: Path):
    """
    绘制 Fig 1: 时空供需动态图
    
    布局：
    - 上半部分：左右两个热力图（商家和客户分布）
    - 下半部分：时间轴订单到达分布
    """
    print("\nGenerating Fig 1: Spatio-Temporal Supply-Demand Dynamics...")
    
    # 创建图形和网格布局
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, height_ratios=[1.2, 0.8], hspace=0.25, wspace=0.15)
    
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
    
    # 扩展边界5%
    lat_margin = (all_lats.max() - all_lats.min()) * 0.05
    lng_margin = (all_lngs.max() - all_lngs.min()) * 0.05
    
    bounds = (
        all_lats.min() - lat_margin,
        all_lats.max() + lat_margin,
        all_lngs.min() - lng_margin,
        all_lngs.max() + lng_margin
    )
    
    # ========== 左上：商家分布热力图 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 绘制路网背景
    for edge in road_data['edges']:
        ax1.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                color='#CCCCCC', linewidth=0.3, alpha=0.4, zorder=1)
    
    # 商家热力图
    merchant_heatmap, extent = create_heatmap(orders_data['merchants'], bounds, resolution=80)
    
    # 自定义颜色映射（白色到红色）
    colors_merchant = ['#FFFFFF00', '#FFE5E5', '#FF9999', '#FF4444', '#CC0000', '#800000']
    cmap_merchant = LinearSegmentedColormap.from_list('merchant', colors_merchant)
    
    im1 = ax1.imshow(merchant_heatmap, extent=extent, origin='lower', 
                     cmap=cmap_merchant, alpha=0.8, aspect='auto', zorder=2)
    
    # 绘制商家点
    ax1.scatter(orders_data['merchants'][:, 1], orders_data['merchants'][:, 0], 
                s=50, c='#E74C3C', marker='^', edgecolors='white', linewidths=0.8,
                label=f'Merchants (n={len(np.unique(orders_data["merchants"], axis=0))})', zorder=3)
    
    ax1.set_xlabel('Longitude', fontsize=11)
    ax1.set_ylabel('Latitude', fontsize=11)
    ax1.set_title('(a) Merchant Distribution', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.2)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
    cbar1.set_label('Density', fontsize=10)
    
    # ========== 右上：客户分布热力图 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 绘制路网背景
    for edge in road_data['edges']:
        ax2.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 
                color='#CCCCCC', linewidth=0.3, alpha=0.4, zorder=1)
    
    # 客户热力图
    customer_heatmap, extent = create_heatmap(orders_data['customers'], bounds, resolution=80)
    
    # 自定义颜色映射（白色到蓝色）
    colors_customer = ['#FFFFFF00', '#E5F0FF', '#99C2FF', '#4488FF', '#0044CC', '#002280']
    cmap_customer = LinearSegmentedColormap.from_list('customer', colors_customer)
    
    im2 = ax2.imshow(customer_heatmap, extent=extent, origin='lower', 
                     cmap=cmap_customer, alpha=0.8, aspect='auto', zorder=2)
    
    # 绘制客户点
    ax2.scatter(orders_data['customers'][:, 1], orders_data['customers'][:, 0], 
                s=40, c='#3498DB', marker='o', edgecolors='white', linewidths=0.6,
                label=f'Customers (n={len(orders_data["customers"])})', zorder=3)
    
    ax2.set_xlabel('Longitude', fontsize=11)
    ax2.set_ylabel('Latitude', fontsize=11)
    ax2.set_title('(b) Customer Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.2)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
    cbar2.set_label('Density', fontsize=10)
    
    # ========== 下方：时间轴订单到达分布 ==========
    ax3 = fig.add_subplot(gs[1, :])
    
    arrival_times = orders_data['arrival_times']
    arrival_hours = arrival_times / 3600  # 转换为小时
    
    # 创建时间分布直方图
    bins = np.arange(8, 23, 0.5)  # 8:00 - 22:00，每30分钟一个bin
    counts, bin_edges = np.histogram(arrival_hours, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 绘制柱状图
    bar_colors = []
    for hour in bin_centers:
        if 11 <= hour < 13:
            bar_colors.append('#FF9800')  # 午高峰 - 橙色
        elif 17 <= hour < 20:
            bar_colors.append('#F44336')  # 晚高峰 - 红色
        else:
            bar_colors.append('#2196F3')  # 普通时段 - 蓝色
    
    bars = ax3.bar(bin_centers, counts, width=0.45, color=bar_colors, 
                   edgecolor='white', linewidth=0.5, alpha=0.85)
    
    # 添加平滑曲线（移动平均）
    if len(counts) > 3:
        window = 3
        smoothed = np.convolve(counts, np.ones(window)/window, mode='valid')
        smoothed_x = bin_centers[window-1:]
        ax3.plot(smoothed_x, smoothed, color='#1A237E', linewidth=2.5, 
                label='Smoothed Trend', zorder=5)
    
    # 标注高峰时段
    ax3.axvspan(11, 13, alpha=0.15, color='#FF9800', label='Lunch Peak (11:00-13:00)')
    ax3.axvspan(17, 20, alpha=0.15, color='#F44336', label='Dinner Peak (17:00-20:00)')
    
    # 添加峰值标注
    if len(counts) > 0:
        max_idx = np.argmax(counts)
        max_hour = bin_centers[max_idx]
        max_count = counts[max_idx]
        ax3.annotate(f'Peak: {max_count} orders\n{int(max_hour)}:{int((max_hour%1)*60):02d}',
                    xy=(max_hour, max_count), xytext=(max_hour + 1.5, max_count + 2),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlabel('Time of Day (Hour)', fontsize=12)
    ax3.set_ylabel('Number of Orders', fontsize=12)
    ax3.set_title('(c) Temporal Order Arrival Pattern (Non-Homogeneous Poisson Process)', 
                  fontsize=13, fontweight='bold', pad=10)
    
    # 设置x轴刻度
    ax3.set_xticks(range(8, 23))
    ax3.set_xticklabels([f'{h}:00' for h in range(8, 23)], rotation=45, ha='right')
    ax3.set_xlim(7.5, 22.5)
    
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # 添加统计信息文本框
    stats_text = (f"Total Orders: {len(arrival_times)}\n"
                  f"Time Range: 08:00 - 22:00\n"
                  f"Avg Arrival Rate: {len(arrival_times)/14:.1f} orders/hour")
    ax3.text(0.98, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # ========== 总标题 ==========
    fig.suptitle('Fig 1. Spatio-Temporal Supply-Demand Dynamics\n'
                 'Shanghai Food Delivery Service Area', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 保存图表
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Fig 1. Spatio-Temporal Supply-Demand Dynamics Generator")
    print("=" * 70)
    
    # 选择数据文件
    if ROAD_NETWORK_FILE.exists():
        network_file = ROAD_NETWORK_FILE
    elif BACKUP_NETWORK_FILE.exists():
        network_file = BACKUP_NETWORK_FILE
        print(f"Using backup network file: {network_file}")
    else:
        raise FileNotFoundError(f"Road network file not found: {ROAD_NETWORK_FILE}")
    
    if ORDERS_FILE.exists():
        orders_file = ORDERS_FILE
    elif BACKUP_ORDERS_FILE.exists():
        orders_file = BACKUP_ORDERS_FILE
        print(f"Using backup orders file: {orders_file}")
    else:
        raise FileNotFoundError(f"Orders file not found: {ORDERS_FILE}")
    
    # 加载数据
    road_data = load_road_network(network_file)
    orders_data = load_orders(orders_file)
    
    # 生成 Fig 1
    output_path = OUTPUT_DIR / "fig1_spatiotemporal_dynamics.png"
    plot_fig1_spatiotemporal(road_data, orders_data, output_path)
    
    print("\n" + "=" * 70)
    print("Done!")
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

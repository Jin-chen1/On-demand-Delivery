"""
可视化模块 - Day 5
绘制骑手轨迹、订单热力图、性能曲线等
用于生成论文 Fig 1, 3, 5 所需的图表
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx

logger = logging.getLogger(__name__)


class Visualizer:
    """
    可视化工具类
    提供轨迹绘制、热力图、动画等功能
    """
    
    def __init__(self, graph: nx.MultiDiGraph, output_dir: Optional[Path] = None):
        """
        初始化可视化器
        
        Args:
            graph: OSM路网图（NetworkX格式）
            output_dir: 输出目录，默认为 ./outputs/visualizations
        """
        self.graph = graph
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取节点坐标
        self.node_positions = {
            node: (data['x'], data['y']) 
            for node, data in graph.nodes(data=True)
        }
        
        self.logger = logging.getLogger(__name__)
        
        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_courier_routes(
        self,
        couriers: Dict[int, Any],
        orders: Dict[int, Any],
        title: str = "Courier Routes",
        filename: str = "courier_routes.png",
        show_graph: bool = False
    ) -> Path:
        """
        绘制骑手路线图（用于论文 Fig 5 - 轨迹对比案例）
        
        Args:
            couriers: 骑手字典
            orders: 订单字典
            title: 图表标题
            filename: 输出文件名
            show_graph: 是否显示路网背景
        
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始绘制骑手路线图: {title}")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制路网背景（可选）
        if show_graph:
            self._draw_graph_background(ax, alpha=0.1)
        
        # 为每个骑手分配颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(couriers)))
        
        # 绘制每个骑手的路线
        for idx, (courier_id, courier) in enumerate(couriers.items()):
            color = colors[idx]
            
            # 获取骑手的完整轨迹（如果有记录）
            if hasattr(courier, 'trajectory') and courier.trajectory:
                trajectory = courier.trajectory
                xs = [self.node_positions[node][0] for node in trajectory if node in self.node_positions]
                ys = [self.node_positions[node][1] for node in trajectory if node in self.node_positions]
                
                if xs and ys:
                    ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, label=f'Courier {courier_id}')
                    
                    # 标记起点
                    ax.scatter(xs[0], ys[0], c=[color], s=100, marker='o', 
                              edgecolors='black', linewidths=2, zorder=5)
                    
                    # 标记终点
                    ax.scatter(xs[-1], ys[-1], c=[color], s=100, marker='s',
                              edgecolors='black', linewidths=2, zorder=5)
            else:
                # 如果没有轨迹记录，仅绘制当前位置
                if courier.current_node in self.node_positions:
                    x, y = self.node_positions[courier.current_node]
                    ax.scatter(x, y, c=[color], s=150, marker='D',
                              edgecolors='black', linewidths=2, label=f'Courier {courier_id}',
                              zorder=5)
        
        # 绘制订单位置（商家和客户）
        merchant_nodes = set()
        customer_nodes = set()
        
        for order in orders.values():
            if order.merchant_node in self.node_positions:
                merchant_nodes.add(order.merchant_node)
            if order.customer_node in self.node_positions:
                customer_nodes.add(order.customer_node)
        
        # 商家位置（三角形）
        if merchant_nodes:
            merchant_coords = [self.node_positions[node] for node in merchant_nodes]
            merchant_xs = [c[0] for c in merchant_coords]
            merchant_ys = [c[1] for c in merchant_coords]
            ax.scatter(merchant_xs, merchant_ys, c='red', s=80, marker='^',
                      alpha=0.6, label='Merchants', zorder=3)
        
        # 客户位置（倒三角）
        if customer_nodes:
            customer_coords = [self.node_positions[node] for node in customer_nodes]
            customer_xs = [c[0] for c in customer_coords]
            customer_ys = [c[1] for c in customer_coords]
            ax.scatter(customer_xs, customer_ys, c='blue', s=80, marker='v',
                      alpha=0.6, label='Customers', zorder=3)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"路线图已保存: {output_path}")
        return output_path
    
    def plot_order_heatmap(
        self,
        orders: Dict[int, Any],
        title: str = "Order Distribution Heatmap",
        filename: str = "order_heatmap.png"
    ) -> Path:
        """
        绘制订单分布热力图（用于论文 Fig 1 - 时空供需动态图）
        
        Args:
            orders: 订单字典
            title: 图表标题
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始绘制订单热力图: {title}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # 左图：商家热力图
        merchant_positions = []
        for order in orders.values():
            if order.merchant_node in self.node_positions:
                merchant_positions.append(self.node_positions[order.merchant_node])
        
        if merchant_positions:
            merchant_xs = [pos[0] for pos in merchant_positions]
            merchant_ys = [pos[1] for pos in merchant_positions]
            
            ax1.hexbin(merchant_xs, merchant_ys, gridsize=30, cmap='Reds', alpha=0.7)
            ax1.set_title('Merchant Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Longitude', fontsize=12)
            ax1.set_ylabel('Latitude', fontsize=12)
        
        # 右图：客户热力图
        customer_positions = []
        for order in orders.values():
            if order.customer_node in self.node_positions:
                customer_positions.append(self.node_positions[order.customer_node])
        
        if customer_positions:
            customer_xs = [pos[0] for pos in customer_positions]
            customer_ys = [pos[1] for pos in customer_positions]
            
            ax2.hexbin(customer_xs, customer_ys, gridsize=30, cmap='Blues', alpha=0.7)
            ax2.set_title('Customer Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Longitude', fontsize=12)
            ax2.set_ylabel('Latitude', fontsize=12)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 保存图表
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"热力图已保存: {output_path}")
        return output_path
    
    def plot_temporal_demand(
        self,
        orders: Dict[int, Any],
        time_window: float = 300.0,
        title: str = "Temporal Order Arrival Pattern",
        filename: str = "temporal_demand.png"
    ) -> Path:
        """
        绘制订单时间分布图（用于论文 Fig 1 - 时空供需动态图的时间维度）
        
        Args:
            orders: 订单字典
            time_window: 时间窗口大小（秒），用于聚合
            title: 图表标题
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始绘制订单时间分布: {title}")
        
        # 提取订单到达时间
        arrival_times = [order.arrival_time for order in orders.values()]
        
        if not arrival_times:
            self.logger.warning("没有订单数据，跳过绘图")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制直方图
        max_time = max(arrival_times)
        bins = int(max_time / time_window) + 1
        
        ax.hist(arrival_times, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Number of Orders', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加移动平均线
        counts, bin_edges = np.histogram(arrival_times, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 简单移动平均
        if len(counts) > 5:
            window = 5
            moving_avg = np.convolve(counts, np.ones(window)/window, mode='valid')
            moving_avg_x = bin_centers[window-1:]
            ax.plot(moving_avg_x, moving_avg, color='red', linewidth=2, 
                   label=f'{window}-bin Moving Average')
            ax.legend()
        
        # 保存图表
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"时间分布图已保存: {output_path}")
        return output_path
    
    def plot_performance_comparison(
        self,
        metrics_dict: Dict[str, Any],
        title: str = "Performance Comparison",
        filename: str = "performance_comparison.png"
    ) -> Path:
        """
        绘制性能对比图（用于论文 Fig 4 - 压力测试曲线）
        
        Args:
            metrics_dict: {method_name: PerformanceMetrics} 字典
            title: 图表标题
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始绘制性能对比图: {title}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        methods = list(metrics_dict.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # 子图1: 超时率对比
        timeout_rates = [metrics_dict[m].timeout_rate * 100 for m in methods]
        axes[0, 0].bar(methods, timeout_rates, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Timeout Rate (%)', fontsize=12)
        axes[0, 0].set_title('Timeout Rate Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 子图2: 平均配送时间对比
        avg_delivery_times = [metrics_dict[m].avg_delivery_time / 60 for m in methods]
        axes[0, 1].bar(methods, avg_delivery_times, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Avg Delivery Time (minutes)', fontsize=12)
        axes[0, 1].set_title('Average Delivery Time', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 子图3: 骑手利用率对比
        utilizations = [metrics_dict[m].avg_utilization * 100 for m in methods]
        axes[1, 0].bar(methods, utilizations, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Utilization (%)', fontsize=12)
        axes[1, 0].set_title('Courier Utilization', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 子图4: 单位里程配送订单数对比
        orders_per_km = [metrics_dict[m].orders_per_km for m in methods]
        axes[1, 1].bar(methods, orders_per_km, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Orders per KM', fontsize=12)
        axes[1, 1].set_title('Delivery Efficiency', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 保存图表
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"性能对比图已保存: {output_path}")
        return output_path
    
    def create_route_animation(
        self,
        courier_trajectories: Dict[int, List[Tuple[float, int]]],
        orders: Dict[int, Any],
        duration: float = 1800.0,
        fps: int = 5,
        filename: str = "route_animation.gif"
    ) -> Path:
        """
        创建路线动画（用于演示或补充材料）
        
        Args:
            courier_trajectories: {courier_id: [(timestamp, node_id), ...]}
            orders: 订单字典
            duration: 仿真总时长（秒）
            fps: 帧率
            filename: 输出文件名
        
        Returns:
            输出文件路径
        """
        self.logger.info(f"开始创建路线动画: {filename}")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制路网背景
        self._draw_graph_background(ax, alpha=0.1)
        
        # 绘制订单位置
        merchant_coords = [
            self.node_positions[o.merchant_node] 
            for o in orders.values() 
            if o.merchant_node in self.node_positions
        ]
        customer_coords = [
            self.node_positions[o.customer_node]
            for o in orders.values()
            if o.customer_node in self.node_positions
        ]
        
        if merchant_coords:
            merchant_xs, merchant_ys = zip(*merchant_coords)
            ax.scatter(merchant_xs, merchant_ys, c='red', s=60, marker='^',
                      alpha=0.5, label='Merchants')
        
        if customer_coords:
            customer_xs, customer_ys = zip(*customer_coords)
            ax.scatter(customer_xs, customer_ys, c='blue', s=60, marker='v',
                      alpha=0.5, label='Customers')
        
        # 准备骑手动画数据
        num_couriers = len(courier_trajectories)
        colors = plt.cm.tab10(np.linspace(0, 1, num_couriers))
        
        courier_scatters = {}
        for idx, courier_id in enumerate(courier_trajectories.keys()):
            scatter = ax.scatter([], [], c=[colors[idx]], s=150, marker='o',
                               edgecolors='black', linewidths=2, 
                               label=f'Courier {courier_id}', zorder=10)
            courier_scatters[courier_id] = scatter
        
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right')
        ax.set_title('Courier Routes Animation', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        def update(frame):
            """更新动画帧"""
            current_time = frame * (duration / (fps * 10))  # 加速播放
            
            for courier_id, trajectory in courier_trajectories.items():
                # 找到当前时间的位置
                current_node = None
                for timestamp, node_id in trajectory:
                    if timestamp <= current_time:
                        current_node = node_id
                    else:
                        break
                
                if current_node and current_node in self.node_positions:
                    x, y = self.node_positions[current_node]
                    courier_scatters[courier_id].set_offsets([[x, y]])
                else:
                    courier_scatters[courier_id].set_offsets([[], []])
            
            time_text.set_text(f'Time: {current_time:.1f}s ({current_time/60:.1f}min)')
            
            return list(courier_scatters.values()) + [time_text]
        
        # 创建动画
        num_frames = fps * 10  # 10秒动画
        anim = FuncAnimation(fig, update, frames=num_frames, 
                            interval=1000/fps, blit=True)
        
        # 保存动画
        output_path = self.output_dir / filename
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        
        self.logger.info(f"动画已保存: {output_path}")
        return output_path
    
    def _draw_graph_background(self, ax, alpha: float = 0.1) -> None:
        """绘制路网背景"""
        edge_coords = []
        for u, v in self.graph.edges():
            if u in self.node_positions and v in self.node_positions:
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                edge_coords.append([(x1, y1), (x2, y2)])
        
        for coords in edge_coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color='gray', linewidth=0.5, alpha=alpha, zorder=1)

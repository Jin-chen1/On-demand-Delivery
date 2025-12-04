"""
Day 9 可视化模块
绘制论文所需的核心图表：压力测试曲线、箱线图等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# 设置绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# 调度器颜色映射
DISPATCHER_COLORS = {
    'greedy': '#E74C3C',      # Red
    'ortools': '#3498DB',     # Blue
    'alns': '#2ECC71'         # Green
}

# 调度器显示名称
DISPATCHER_NAMES = {
    'greedy': 'Greedy',
    'ortools': 'OR-Tools',
    'alns': 'ALNS'
}


class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            dpi: 图片分辨率
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        logger.info(f"可视化器初始化: {self.figures_dir}")
    
    def plot_stress_test_curve(self, 
                               timeout_capacity_data: pd.DataFrame,
                               save_name: str = "stress_test_curve.png") -> Path:
        """
        绘制压力测试曲线（论文 Fig 4 核心图表）
        X轴：运力（骑手数量）
        Y轴：超时率
        分面：不同订单量
        
        Args:
            timeout_capacity_data: 超时率-运力数据
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制压力测试曲线...")
        
        # 获取唯一的订单量
        order_volumes = sorted(timeout_capacity_data['num_orders'].unique())
        n_volumes = len(order_volumes)
        
        # 创建子图
        fig, axes = plt.subplots(1, n_volumes, figsize=(6*n_volumes, 5), sharey=True)
        if n_volumes == 1:
            axes = [axes]
        
        for idx, num_orders in enumerate(order_volumes):
            ax = axes[idx]
            
            # 筛选当前订单量的数据
            data_subset = timeout_capacity_data[
                timeout_capacity_data['num_orders'] == num_orders
            ]
            
            # 为每个调度器绘制曲线
            for dispatcher in sorted(data_subset['dispatcher_type'].unique()):
                disp_data = data_subset[
                    data_subset['dispatcher_type'] == dispatcher
                ].sort_values('num_couriers')
                
                color = DISPATCHER_COLORS.get(dispatcher, '#95A5A6')
                label = DISPATCHER_NAMES.get(dispatcher, dispatcher)
                
                # 绘制线条和误差带
                ax.plot(
                    disp_data['num_couriers'],
                    disp_data['timeout_rate'] * 100,
                    marker='o',
                    color=color,
                    label=label,
                    linewidth=2,
                    markersize=6
                )
                
                # 如果有标准差，绘制误差带
                if 'timeout_rate_std' in disp_data.columns:
                    std = disp_data['timeout_rate_std'] * 100
                    ax.fill_between(
                        disp_data['num_couriers'],
                        (disp_data['timeout_rate'] - std) * 100,
                        (disp_data['timeout_rate'] + std) * 100,
                        alpha=0.2,
                        color=color
                    )
            
            # 设置子图属性
            ax.set_xlabel('Number of Couriers', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Timeout Rate (%)', fontsize=12, fontweight='bold')
            
            ax.set_title(f'{num_orders} Orders', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', frameon=True, shadow=True)
            
            # 设置Y轴范围
            ax.set_ylim(bottom=0)
        
        # 总标题
        fig.suptitle('Stress Test: Timeout Rate vs Fleet Capacity', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 压力测试曲线已保存: {save_path}")
        return save_path
    
    def plot_stress_test_by_load(self,
                                  timeout_capacity_data: pd.DataFrame,
                                  save_name: str = "stress_test_by_load.png") -> Path:
        """
        绘制负载压力测试曲线（符合研究大纲要求的 Fig 4）
        X轴：订单量（Order Volume）- 展示系统在负载增加时的崩溃点
        Y轴：超时率（Timeout Rate %）
        线条：不同调度器算法
        分面（可选）：不同骑手配置
        
        Args:
            timeout_capacity_data: 超时率-运力数据
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制负载压力测试曲线（X轴=订单量）...")
        
        # 获取唯一的骑手数量
        courier_counts = sorted(timeout_capacity_data['num_couriers'].unique())
        n_couriers = len(courier_counts)
        
        # 创建子图（如果只有一个骑手配置，不分面；否则分面展示）
        if n_couriers == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, n_couriers, figsize=(6*n_couriers, 5), sharey=True)
            if n_couriers == 1:
                axes = [axes]
        
        for idx, num_couriers in enumerate(courier_counts):
            ax = axes[idx]
            
            # 筛选当前骑手数量的数据
            data_subset = timeout_capacity_data[
                timeout_capacity_data['num_couriers'] == num_couriers
            ]
            
            # 为每个调度器绘制曲线
            for dispatcher in sorted(data_subset['dispatcher_type'].unique()):
                disp_data = data_subset[
                    data_subset['dispatcher_type'] == dispatcher
                ].sort_values('num_orders')
                
                color = DISPATCHER_COLORS.get(dispatcher, '#95A5A6')
                label = DISPATCHER_NAMES.get(dispatcher, dispatcher)
                
                # 绘制线条和标记点
                ax.plot(
                    disp_data['num_orders'],
                    disp_data['timeout_rate'] * 100,
                    marker='o',
                    color=color,
                    label=label,
                    linewidth=2.5,
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor='white'
                )
                
                # 如果有标准差，绘制误差带
                if 'timeout_rate_std' in disp_data.columns:
                    std = disp_data['timeout_rate_std'] * 100
                    ax.fill_between(
                        disp_data['num_orders'],
                        (disp_data['timeout_rate'] - std) * 100,
                        (disp_data['timeout_rate'] + std) * 100,
                        alpha=0.15,
                        color=color
                    )
            
            # 设置子图属性
            ax.set_xlabel('Order Volume (orders/day)', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Timeout Rate (%)', fontsize=12, fontweight='bold')
            
            # 设置标题
            if n_couriers > 1:
                ax.set_title(f'{num_couriers} Couriers', fontsize=13, fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', frameon=True, shadow=True, fancybox=True)
            
            # 设置Y轴范围
            ax.set_ylim(bottom=0)
            
            # 添加参考线（如果超时率超过50%，标注系统崩溃区域）
            max_timeout = data_subset['timeout_rate'].max() * 100
            if max_timeout > 50:
                ax.axhline(y=50, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                ax.text(
                    ax.get_xlim()[1] * 0.98, 52,
                    'Critical Zone',
                    ha='right', va='bottom',
                    fontsize=9, color='red', style='italic'
                )
        
        # 总标题
        if n_couriers == 1:
            fig.suptitle(
                f'Stress Test: System Performance under Increasing Load ({courier_counts[0]} Couriers)',
                fontsize=14, fontweight='bold', y=0.98
            )
        else:
            fig.suptitle(
                'Stress Test: Timeout Rate vs Order Volume',
                fontsize=16, fontweight='bold', y=1.02
            )
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 负载压力测试曲线已保存: {save_path}")
        return save_path
    
    def plot_service_time_boxplot(self,
                                  results_df: pd.DataFrame,
                                  save_name: str = "service_time_boxplot.png") -> Path:
        """
        绘制服务时间箱线图
        对比不同算法在不同场景下的服务时间分布
        
        Args:
            results_df: 原始结果数据
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制服务时间箱线图...")
        
        # 获取唯一的订单量
        order_volumes = sorted(results_df['num_orders'].unique())
        n_volumes = len(order_volumes)
        
        # 创建子图
        fig, axes = plt.subplots(1, n_volumes, figsize=(6*n_volumes, 5), sharey=True)
        if n_volumes == 1:
            axes = [axes]
        
        for idx, num_orders in enumerate(order_volumes):
            ax = axes[idx]
            
            # 筛选数据
            data_subset = results_df[results_df['num_orders'] == num_orders]
            
            # 准备绘图数据
            plot_data = []
            labels = []
            colors = []
            
            for dispatcher in sorted(data_subset['dispatcher_type'].unique()):
                disp_data = data_subset[
                    data_subset['dispatcher_type'] == dispatcher
                ]['avg_service_time']
                
                plot_data.append(disp_data)
                labels.append(DISPATCHER_NAMES.get(dispatcher, dispatcher))
                colors.append(DISPATCHER_COLORS.get(dispatcher, '#95A5A6'))
            
            # 绘制箱线图
            bp = ax.boxplot(
                plot_data,
                labels=labels,
                patch_artist=True,
                notch=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=5)
            )
            
            # 设置颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 设置子图属性
            ax.set_xlabel('Dispatcher Algorithm', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Service Time (seconds)', fontsize=12, fontweight='bold')
            
            ax.set_title(f'{num_orders} Orders', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 总标题
        fig.suptitle('Service Time Distribution by Algorithm', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 服务时间箱线图已保存: {save_path}")
        return save_path
    
    def plot_completion_rate_comparison(self,
                                       aggregated_data: pd.DataFrame,
                                       save_name: str = "completion_rate_comparison.png") -> Path:
        """
        绘制完成率对比图
        
        Args:
            aggregated_data: 聚合数据（按订单量和调度器）
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制完成率对比图...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 获取唯一的订单量和调度器
        order_volumes = sorted(aggregated_data['num_orders'].unique())
        dispatchers = sorted(aggregated_data['dispatcher_type'].unique())
        
        # 设置柱状图位置
        x = np.arange(len(order_volumes))
        width = 0.25
        
        # 为每个调度器绘制柱状图
        for idx, dispatcher in enumerate(dispatchers):
            disp_data = aggregated_data[
                aggregated_data['dispatcher_type'] == dispatcher
            ].sort_values('num_orders')
            
            completion_rates = disp_data['completion_rate_mean'].values * 100
            std_values = disp_data.get('completion_rate_std', 
                                      pd.Series([0]*len(disp_data))).values * 100
            
            color = DISPATCHER_COLORS.get(dispatcher, '#95A5A6')
            label = DISPATCHER_NAMES.get(dispatcher, dispatcher)
            
            ax.bar(
                x + idx * width,
                completion_rates,
                width,
                label=label,
                color=color,
                alpha=0.8,
                yerr=std_values,
                capsize=5
            )
        
        # 设置图表属性
        ax.set_xlabel('Order Volume', fontsize=12, fontweight='bold')
        ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Order Completion Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{vol} Orders' for vol in order_volumes])
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 完成率对比图已保存: {save_path}")
        return save_path
    
    def plot_solve_time_comparison(self,
                                   aggregated_data: pd.DataFrame,
                                   save_name: str = "solve_time_comparison.png") -> Path:
        """
        绘制算法求解时间对比图
        
        Args:
            aggregated_data: 聚合数据
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制求解时间对比图...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 按调度器分组
        dispatchers = sorted(aggregated_data['dispatcher_type'].unique())
        
        solve_times = []
        labels = []
        colors = []
        
        for dispatcher in dispatchers:
            disp_data = aggregated_data[
                aggregated_data['dispatcher_type'] == dispatcher
            ]
            
            if 'avg_solve_time_mean' in disp_data.columns:
                avg_time = disp_data['avg_solve_time_mean'].mean()
                solve_times.append(avg_time * 1000)  # 转换为毫秒
                labels.append(DISPATCHER_NAMES.get(dispatcher, dispatcher))
                colors.append(DISPATCHER_COLORS.get(dispatcher, '#95A5A6'))
        
        # 绘制柱状图
        bars = ax.bar(labels, solve_times, color=colors, alpha=0.8, edgecolor='black')
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}ms',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # 设置图表属性
        ax.set_ylabel('Average Solve Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Algorithm Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 求解时间对比图已保存: {save_path}")
        return save_path
    
    def plot_courier_utilization_heatmap(self,
                                        timeout_capacity_data: pd.DataFrame,
                                        save_name: str = "courier_utilization_heatmap.png") -> Path:
        """
        绘制骑手利用率热力图
        展示不同配置下的资源利用情况
        
        Args:
            timeout_capacity_data: 数据
            save_name: 保存文件名
        
        Returns:
            保存的文件路径
        """
        logger.info("绘制骑手利用率热力图...")
        
        # 获取唯一的调度器
        dispatchers = sorted(timeout_capacity_data['dispatcher_type'].unique())
        n_dispatchers = len(dispatchers)
        
        # 创建子图
        fig, axes = plt.subplots(1, n_dispatchers, figsize=(7*n_dispatchers, 5))
        if n_dispatchers == 1:
            axes = [axes]
        
        for idx, dispatcher in enumerate(dispatchers):
            ax = axes[idx]
            
            # 筛选数据
            data_subset = timeout_capacity_data[
                timeout_capacity_data['dispatcher_type'] == dispatcher
            ]
            
            # 创建透视表（订单量 × 骑手数）
            if 'avg_courier_utilization' in data_subset.columns:
                pivot_data = data_subset.pivot_table(
                    values='avg_courier_utilization',
                    index='num_orders',
                    columns='num_couriers',
                    aggfunc='mean'
                ) * 100
                
                # 绘制热力图
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.1f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Utilization (%)'},
                    ax=ax,
                    vmin=0,
                    vmax=100
                )
                
                ax.set_title(f'{DISPATCHER_NAMES.get(dispatcher, dispatcher)}', 
                           fontsize=13, fontweight='bold')
                ax.set_xlabel('Number of Couriers', fontsize=11, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('Order Volume', fontsize=11, fontweight='bold')
        
        # 总标题
        fig.suptitle('Courier Utilization Rate Heatmap', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = self.figures_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ 骑手利用率热力图已保存: {save_path}")
        return save_path
    
    def generate_all_figures(self,
                            results_df: pd.DataFrame,
                            analysis: Dict) -> Dict[str, Path]:
        """
        生成所有论文所需的图表
        
        Args:
            results_df: 原始结果数据
            analysis: 分析结果字典
        
        Returns:
            图表文件路径字典
        """
        logger.info("="*80)
        logger.info("开始生成所有图表...")
        logger.info("="*80)
        
        figures = {}
        
        # 1. 核心图表：负载压力测试曲线（论文 Fig 4 - X轴=订单量，符合大纲要求）
        if 'timeout_vs_capacity' in analysis:
            try:
                fig_path = self.plot_stress_test_by_load(
                    analysis['timeout_vs_capacity']
                )
                figures['stress_test_by_load'] = fig_path
                logger.info("  [核心图表] 负载压力测试曲线（X=订单量）")
            except Exception as e:
                logger.error(f"绘制负载压力测试曲线失败: {e}")
        
        # 1b. 补充图表：运力配置曲线（X轴=骑手数量）
        if 'timeout_vs_capacity' in analysis:
            try:
                fig_path = self.plot_stress_test_curve(
                    analysis['timeout_vs_capacity']
                )
                figures['stress_test_curve'] = fig_path
                logger.info("  [补充图表] 运力配置曲线（X=骑手数量）")
            except Exception as e:
                logger.error(f"绘制运力配置曲线失败: {e}")
        
        # 2. 核心图表：服务时间箱线图
        try:
            fig_path = self.plot_service_time_boxplot(results_df)
            figures['service_time_boxplot'] = fig_path
        except Exception as e:
            logger.error(f"绘制服务时间箱线图失败: {e}")
        
        # 3. 辅助图表：完成率对比
        if 'by_order_volume' in analysis:
            try:
                fig_path = self.plot_completion_rate_comparison(
                    analysis['by_order_volume']
                )
                figures['completion_rate'] = fig_path
            except Exception as e:
                logger.error(f"绘制完成率对比图失败: {e}")
        
        # 4. 辅助图表：求解时间对比
        if 'by_dispatcher' in analysis:
            try:
                fig_path = self.plot_solve_time_comparison(
                    analysis['by_dispatcher']
                )
                figures['solve_time'] = fig_path
            except Exception as e:
                logger.error(f"绘制求解时间对比图失败: {e}")
        
        # 5. 辅助图表：骑手利用率热力图
        if 'timeout_vs_capacity' in analysis:
            try:
                fig_path = self.plot_courier_utilization_heatmap(
                    analysis['timeout_vs_capacity']
                )
                figures['utilization_heatmap'] = fig_path
            except Exception as e:
                logger.error(f"绘制骑手利用率热力图失败: {e}")
        
        logger.info("="*80)
        logger.info(f"图表生成完成！共生成 {len(figures)} 张图表")
        logger.info("="*80)
        
        for name, path in figures.items():
            logger.info(f"  - {name}: {path}")
        
        return figures

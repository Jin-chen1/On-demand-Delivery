"""
报告生成器 - Day 5
整合指标和可视化，生成综合分析报告
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from .metrics import MetricsCalculator, PerformanceMetrics
from .visualization import Visualizer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    综合报告生成器
    整合指标计算和可视化，生成完整的分析报告
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录，默认为 ./outputs/reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def generate_single_run_report(
        self,
        env,
        graph,
        report_name: str = "simulation_report"
    ) -> Dict[str, Path]:
        """
        生成单次仿真运行报告
        
        Args:
            env: SimulationEnvironment 实例
            graph: OSM路网图
            report_name: 报告名称
        
        Returns:
            生成的文件路径字典
        """
        self.logger.info(f"开始生成报告: {report_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # 1. 计算指标
        self.logger.info("计算性能指标...")
        metrics = self.metrics_calculator.calculate_from_environment(env)
        
        # 保存指标到 JSON
        metrics_path = report_dir / "metrics.json"
        self.metrics_calculator.save_metrics(metrics, metrics_path)
        output_files['metrics_json'] = metrics_path
        
        # 保存指标到 CSV
        metrics_csv_path = report_dir / "metrics.csv"
        self.metrics_calculator.save_metrics(metrics, metrics_csv_path)
        output_files['metrics_csv'] = metrics_csv_path
        
        # 2. 生成可视化
        self.logger.info("生成可视化图表...")
        visualizer = Visualizer(graph, output_dir=report_dir)
        
        # 骑手路线图
        routes_path = visualizer.plot_courier_routes(
            env.couriers,
            env.orders,
            title=f"Courier Routes - {report_name}",
            filename="courier_routes.png"
        )
        output_files['courier_routes'] = routes_path
        
        # 订单热力图
        heatmap_path = visualizer.plot_order_heatmap(
            env.orders,
            title=f"Order Distribution - {report_name}",
            filename="order_heatmap.png"
        )
        output_files['order_heatmap'] = heatmap_path
        
        # 订单时间分布
        temporal_path = visualizer.plot_temporal_demand(
            env.orders,
            title=f"Order Arrival Pattern - {report_name}",
            filename="temporal_demand.png"
        )
        output_files['temporal_demand'] = temporal_path
        
        # 3. 生成文本报告
        self.logger.info("生成文本报告...")
        text_report_path = self._generate_text_report(
            metrics, 
            env, 
            report_dir / "report.txt"
        )
        output_files['text_report'] = text_report_path
        
        # 4. 生成 Markdown 报告
        md_report_path = self._generate_markdown_report(
            metrics,
            env,
            output_files,
            report_dir / "report.md"
        )
        output_files['markdown_report'] = md_report_path
        
        self.logger.info(f"报告生成完成: {report_dir}")
        self.logger.info(f"生成的文件: {len(output_files)} 个")
        
        return output_files
    
    def generate_comparison_report(
        self,
        envs: Dict[str, Any],
        graph,
        report_name: str = "comparison_report"
    ) -> Dict[str, Path]:
        """
        生成多方法对比报告（用于论文 Fig 4）
        
        Args:
            envs: {method_name: SimulationEnvironment} 字典
            graph: OSM路网图
            report_name: 报告名称
        
        Returns:
            生成的文件路径字典
        """
        self.logger.info(f"开始生成对比报告: {report_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # 1. 计算所有方法的指标
        self.logger.info("计算各方法的性能指标...")
        metrics_dict = {}
        
        for method_name, env in envs.items():
            metrics = self.metrics_calculator.calculate_from_environment(env)
            metrics_dict[method_name] = metrics
            
            # 保存各方法的指标
            method_metrics_path = report_dir / f"metrics_{method_name}.json"
            self.metrics_calculator.save_metrics(metrics, method_metrics_path)
        
        # 2. 计算改进率（如果有baseline）
        if 'baseline' in metrics_dict or 'greedy' in metrics_dict:
            baseline_key = 'baseline' if 'baseline' in metrics_dict else 'greedy'
            baseline_metrics = metrics_dict[baseline_key]
            
            for method_name, metrics in metrics_dict.items():
                if method_name != baseline_key:
                    improvements = self.metrics_calculator.compare_metrics(
                        baseline_metrics, metrics
                    )
                    
                    # 保存改进率
                    improvements_path = report_dir / f"improvements_{method_name}_vs_{baseline_key}.json"
                    with open(improvements_path, 'w', encoding='utf-8') as f:
                        json.dump(improvements, f, indent=2)
        
        # 3. 生成对比可视化
        self.logger.info("生成对比图表...")
        visualizer = Visualizer(graph, output_dir=report_dir)
        
        comparison_path = visualizer.plot_performance_comparison(
            metrics_dict,
            title="Performance Comparison Across Methods",
            filename="performance_comparison.png"
        )
        output_files['performance_comparison'] = comparison_path
        
        # 4. 为每个方法生成路线图
        for method_name, env in envs.items():
            routes_path = visualizer.plot_courier_routes(
                env.couriers,
                env.orders,
                title=f"Routes - {method_name}",
                filename=f"routes_{method_name}.png"
            )
            output_files[f'routes_{method_name}'] = routes_path
        
        # 5. 生成对比表格（CSV）
        comparison_csv_path = self._generate_comparison_csv(
            metrics_dict,
            report_dir / "comparison_table.csv"
        )
        output_files['comparison_csv'] = comparison_csv_path
        
        # 6. 生成 Markdown 对比报告
        md_report_path = self._generate_comparison_markdown(
            metrics_dict,
            output_files,
            report_dir / "comparison_report.md"
        )
        output_files['markdown_report'] = md_report_path
        
        self.logger.info(f"对比报告生成完成: {report_dir}")
        return output_files
    
    def _generate_text_report(
        self,
        metrics: PerformanceMetrics,
        env,
        output_path: Path
    ) -> Path:
        """生成文本格式报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("仿真分析报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"仿真时长: {env.config.get('simulation_duration', 0):.1f}秒\n")
            f.write(f"调度器类型: {env.config.get('dispatcher_type', 'N/A')}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("订单统计\n")
            f.write("-"*70 + "\n")
            f.write(f"总订单数: {metrics.total_orders}\n")
            f.write(f"  已完成: {metrics.completed_orders} ({metrics.completed_orders/max(metrics.total_orders,1)*100:.1f}%)\n")
            f.write(f"  超时: {metrics.timeout_orders} (超时率: {metrics.timeout_rate*100:.2f}%)\n")
            f.write(f"  待分配: {metrics.pending_orders}\n")
            f.write(f"  已分配: {metrics.assigned_orders}\n\n")
            
            f.write("-"*70 + "\n")
            f.write("时间指标\n")
            f.write("-"*70 + "\n")
            f.write(f"平均配送时间: {metrics.avg_delivery_time:.1f}秒 ({metrics.avg_delivery_time/60:.1f}分钟)\n")
            f.write(f"平均等待时间: {metrics.avg_waiting_time:.1f}秒 ({metrics.avg_waiting_time/60:.1f}分钟)\n")
            f.write(f"最大配送时间: {metrics.max_delivery_time:.1f}秒\n")
            min_time = metrics.min_delivery_time if metrics.min_delivery_time != float('inf') else 0
            f.write(f"最小配送时间: {min_time:.1f}秒\n\n")
            
            f.write("-"*70 + "\n")
            f.write("骑手统计\n")
            f.write("-"*70 + "\n")
            f.write(f"总骑手数: {metrics.total_couriers}\n")
            f.write(f"平均利用率: {metrics.avg_utilization*100:.1f}%\n")
            f.write(f"总行驶距离: {metrics.total_distance/1000:.2f}公里\n")
            f.write(f"人均行驶距离: {metrics.avg_distance_per_courier/1000:.2f}公里\n")
            f.write(f"人均完成订单: {metrics.avg_orders_per_courier:.2f}单\n")
            f.write(f"单位里程配送: {metrics.orders_per_km:.2f}单/公里\n\n")
            
            f.write("="*70 + "\n")
        
        self.logger.info(f"文本报告已保存: {output_path}")
        return output_path
    
    def _generate_markdown_report(
        self,
        metrics: PerformanceMetrics,
        env,
        output_files: Dict[str, Path],
        output_path: Path
    ) -> Path:
        """生成 Markdown 格式报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Simulation Analysis Report\n\n")
            
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Simulation Duration:** {env.config.get('simulation_duration', 0):.1f}s\n\n")
            f.write(f"**Dispatcher:** {env.config.get('dispatcher_type', 'N/A')}\n\n")
            
            f.write("## Performance Metrics\n\n")
            
            f.write("### Order Statistics\n\n")
            f.write(f"- **Total Orders:** {metrics.total_orders}\n")
            f.write(f"- **Completed:** {metrics.completed_orders} ({metrics.completed_orders/max(metrics.total_orders,1)*100:.1f}%)\n")
            f.write(f"- **Timeout:** {metrics.timeout_orders} (Rate: {metrics.timeout_rate*100:.2f}%)\n")
            f.write(f"- **Pending:** {metrics.pending_orders}\n")
            f.write(f"- **Assigned:** {metrics.assigned_orders}\n\n")
            
            f.write("### Time Metrics\n\n")
            f.write(f"- **Avg Delivery Time:** {metrics.avg_delivery_time:.1f}s ({metrics.avg_delivery_time/60:.1f}min)\n")
            f.write(f"- **Avg Waiting Time:** {metrics.avg_waiting_time:.1f}s ({metrics.avg_waiting_time/60:.1f}min)\n\n")
            
            f.write("### Courier Statistics\n\n")
            f.write(f"- **Total Couriers:** {metrics.total_couriers}\n")
            f.write(f"- **Avg Utilization:** {metrics.avg_utilization*100:.1f}%\n")
            f.write(f"- **Total Distance:** {metrics.total_distance/1000:.2f} km\n")
            f.write(f"- **Orders per KM:** {metrics.orders_per_km:.2f}\n\n")
            
            f.write("## Visualizations\n\n")
            
            if 'courier_routes' in output_files:
                f.write(f"### Courier Routes\n\n")
                f.write(f"![Courier Routes]({output_files['courier_routes'].name})\n\n")
            
            if 'order_heatmap' in output_files:
                f.write(f"### Order Distribution\n\n")
                f.write(f"![Order Heatmap]({output_files['order_heatmap'].name})\n\n")
            
            if 'temporal_demand' in output_files:
                f.write(f"### Temporal Demand Pattern\n\n")
                f.write(f"![Temporal Demand]({output_files['temporal_demand'].name})\n\n")
        
        self.logger.info(f"Markdown报告已保存: {output_path}")
        return output_path
    
    def _generate_comparison_csv(
        self,
        metrics_dict: Dict[str, PerformanceMetrics],
        output_path: Path
    ) -> Path:
        """生成对比表格（CSV格式）"""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Method', 'Completed Orders', 'Timeout Rate (%)', 
                         'Avg Delivery Time (min)', 'Avg Utilization (%)',
                         'Total Distance (km)', 'Orders per KM']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for method, metrics in metrics_dict.items():
                writer.writerow({
                    'Method': method,
                    'Completed Orders': metrics.completed_orders,
                    'Timeout Rate (%)': f"{metrics.timeout_rate*100:.2f}",
                    'Avg Delivery Time (min)': f"{metrics.avg_delivery_time/60:.2f}",
                    'Avg Utilization (%)': f"{metrics.avg_utilization*100:.1f}",
                    'Total Distance (km)': f"{metrics.total_distance/1000:.2f}",
                    'Orders per KM': f"{metrics.orders_per_km:.2f}"
                })
        
        self.logger.info(f"对比CSV已保存: {output_path}")
        return output_path
    
    def _generate_comparison_markdown(
        self,
        metrics_dict: Dict[str, PerformanceMetrics],
        output_files: Dict[str, Path],
        output_path: Path
    ) -> Path:
        """生成对比报告（Markdown格式）"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Method Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Comparison Table\n\n")
            
            # 生成 Markdown 表格
            f.write("| Method | Completed | Timeout Rate | Avg Delivery Time | Utilization | Distance | Orders/KM |\n")
            f.write("|--------|-----------|--------------|-------------------|-------------|----------|----------|\n")
            
            for method, metrics in metrics_dict.items():
                f.write(f"| {method} | {metrics.completed_orders} | "
                       f"{metrics.timeout_rate*100:.2f}% | "
                       f"{metrics.avg_delivery_time/60:.2f}min | "
                       f"{metrics.avg_utilization*100:.1f}% | "
                       f"{metrics.total_distance/1000:.2f}km | "
                       f"{metrics.orders_per_km:.2f} |\n")
            
            f.write("\n## Performance Comparison Chart\n\n")
            
            if 'performance_comparison' in output_files:
                f.write(f"![Performance Comparison]({output_files['performance_comparison'].name})\n\n")
            
            f.write("## Route Visualizations\n\n")
            
            for method in metrics_dict.keys():
                routes_key = f'routes_{method}'
                if routes_key in output_files:
                    f.write(f"### {method}\n\n")
                    f.write(f"![Routes - {method}]({output_files[routes_key].name})\n\n")
        
        self.logger.info(f"对比Markdown报告已保存: {output_path}")
        return output_path

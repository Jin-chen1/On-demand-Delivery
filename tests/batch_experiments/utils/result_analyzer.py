"""
结果分析器 - Day 8
收集、聚合和分析批量实验结果
生成论文所需的数据和图表
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultCollector:
    """结果收集器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化结果收集器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"结果收集器初始化: {output_dir}")
    
    def save_raw_results(self, results: List[Dict[str, Any]], 
                        filename: str = "raw_results.csv") -> Path:
        """
        保存原始结果到CSV
        
        Args:
            results: 结果列表
            filename: 文件名
        
        Returns:
            保存的文件路径
        """
        if not results:
            logger.warning("没有结果可保存")
            return None
        
        logger.info(f"保存 {len(results)} 个实验结果...")
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV
        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"✓ 原始结果已保存: {output_file}")
        return output_file
    
    def save_results_json(self, results: List[Dict[str, Any]], 
                         filename: str = "raw_results.json") -> Path:
        """保存结果到JSON"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ JSON结果已保存: {output_file}")
        return output_file


class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化数据分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据分析器初始化: {output_dir}")
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        全面分析实验结果
        
        Args:
            results_df: 结果DataFrame
        
        Returns:
            分析报告字典
        """
        logger.info("开始分析实验结果...")
        
        analysis = {}
        
        # 1. 基础统计
        analysis['basic_stats'] = self._compute_basic_stats(results_df)
        
        # 2. 按调度器聚合
        analysis['by_dispatcher'] = self._aggregate_by_dispatcher(results_df)
        
        # 3. 按订单量聚合
        analysis['by_order_volume'] = self._aggregate_by_orders(results_df)
        
        # 4. 按骑手数聚合
        analysis['by_courier_count'] = self._aggregate_by_couriers(results_df)
        
        # 5. 生成超时率-运力曲线数据（论文核心）
        analysis['timeout_vs_capacity'] = self._generate_timeout_capacity_data(results_df)
        
        logger.info("✓ 结果分析完成")
        
        return analysis
    
    def _compute_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算基础统计"""
        stats = {
            'total_experiments': len(df),
            'unique_dispatchers': df['dispatcher_type'].nunique(),
            'unique_order_volumes': sorted(df['num_orders'].unique().tolist()),
            'unique_courier_counts': sorted(df['num_couriers'].unique().tolist()),
            'avg_completion_rate': df['completion_rate'].mean(),
            'avg_timeout_rate': df['timeout_rate'].mean(),
            'avg_service_time': df['avg_service_time'].mean(),
            'total_duration_hours': df['duration_seconds'].sum() / 3600
        }
        
        return stats
    
    def _aggregate_by_dispatcher(self, df: pd.DataFrame) -> pd.DataFrame:
        """按调度器聚合"""
        agg_funcs = {
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std'],
            'avg_service_time': ['mean', 'std'],
            'avg_solve_time': ['mean', 'std'],
            'total_distance': ['mean', 'std'],
            'avg_courier_utilization': ['mean', 'std']
        }
        
        grouped = df.groupby('dispatcher_type').agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return grouped.reset_index()
    
    def _aggregate_by_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """按订单量聚合"""
        agg_funcs = {
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std'],
            'avg_service_time': ['mean', 'std']
        }
        
        grouped = df.groupby(['num_orders', 'dispatcher_type']).agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return grouped.reset_index()
    
    def _aggregate_by_couriers(self, df: pd.DataFrame) -> pd.DataFrame:
        """按骑手数聚合"""
        agg_funcs = {
            'completion_rate': ['mean', 'std'],
            'timeout_rate': ['mean', 'std'],
            'avg_courier_utilization': ['mean', 'std']
        }
        
        grouped = df.groupby(['num_couriers', 'dispatcher_type']).agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return grouped.reset_index()
    
    def _generate_timeout_capacity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成超时率-运力曲线数据（论文 Fig 4 核心）
        
        Returns:
            DataFrame with columns: num_orders, num_couriers, dispatcher_type, 
                                    timeout_rate_mean, timeout_rate_std
        """
        logger.info("生成超时率-运力曲线数据...")
        
        # 按订单量、骑手数、调度器分组
        grouped = df.groupby(['num_orders', 'num_couriers', 'dispatcher_type']).agg({
            'timeout_rate': ['mean', 'std', 'count'],
            'completion_rate': 'mean',
            'avg_service_time': 'mean'
        })
        
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        result = grouped.reset_index()
        
        # 重命名列以便使用
        result = result.rename(columns={
            'timeout_rate_mean': 'timeout_rate',
            'timeout_rate_std': 'timeout_rate_std',
            'timeout_rate_count': 'num_repeats'
        })
        
        return result
    
    def save_analysis_results(self, analysis: Dict[str, Any]) -> None:
        """
        保存分析结果
        
        Args:
            analysis: 分析结果字典
        """
        logger.info("保存分析结果...")
        
        # 1. 保存基础统计（JSON）
        stats_file = self.output_dir / "analysis_summary.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换DataFrame为可序列化格式
            serializable_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, pd.DataFrame):
                    serializable_analysis[key] = value.to_dict(orient='records')
                else:
                    serializable_analysis[key] = value
            
            json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 分析摘要: {stats_file}")
        
        # 2. 保存聚合结果（CSV）
        if 'by_dispatcher' in analysis:
            file_path = self.output_dir / "aggregated_by_dispatcher.csv"
            analysis['by_dispatcher'].to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 按调度器聚合: {file_path}")
        
        if 'by_order_volume' in analysis:
            file_path = self.output_dir / "aggregated_by_order_volume.csv"
            analysis['by_order_volume'].to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 按订单量聚合: {file_path}")
        
        if 'by_courier_count' in analysis:
            file_path = self.output_dir / "aggregated_by_courier_count.csv"
            analysis['by_courier_count'].to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 按骑手数聚合: {file_path}")
        
        # 3. 保存超时率-运力曲线数据（论文核心）
        if 'timeout_vs_capacity' in analysis:
            timeout_dir = self.output_dir / "timeout_vs_capacity"
            timeout_dir.mkdir(exist_ok=True)
            
            # 保存完整数据
            file_path = timeout_dir / "timeout_capacity_data.csv"
            analysis['timeout_vs_capacity'].to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"✓ 超时率-运力曲线数据: {file_path}")
            
            # 为每个订单量单独保存一份（便于绘图）
            for num_orders in analysis['timeout_vs_capacity']['num_orders'].unique():
                subset = analysis['timeout_vs_capacity'][
                    analysis['timeout_vs_capacity']['num_orders'] == num_orders
                ]
                file_path = timeout_dir / f"timeout_capacity_{num_orders}orders.csv"
                subset.to_csv(file_path, index=False, encoding='utf-8-sig')
                logger.info(f"  - {num_orders}订单场景: {file_path}")
    
    def generate_report(self, analysis: Dict[str, Any], 
                       results_df: pd.DataFrame) -> str:
        """
        生成Markdown格式的实验报告
        
        Args:
            analysis: 分析结果
            results_df: 原始结果DataFrame
        
        Returns:
            报告内容（Markdown格式）
        """
        logger.info("生成实验报告...")
        
        report_lines = []
        
        # 标题
        report_lines.append("# Day 8 Batch Experiment Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 实验概览
        report_lines.append("## Experiment Overview")
        report_lines.append("")
        basic = analysis['basic_stats']
        report_lines.append(f"- **Total Experiments:** {basic['total_experiments']}")
        report_lines.append(f"- **Dispatchers:** {basic['unique_dispatchers']} types")
        report_lines.append(f"- **Order Volumes:** {basic['unique_order_volumes']}")
        report_lines.append(f"- **Courier Counts:** {basic['unique_courier_counts']}")
        report_lines.append(f"- **Total Duration:** {basic['total_duration_hours']:.2f} hours")
        report_lines.append("")
        
        # 整体性能
        report_lines.append("## Overall Performance")
        report_lines.append("")
        report_lines.append(f"- **Average Completion Rate:** {basic['avg_completion_rate']*100:.1f}%")
        report_lines.append(f"- **Average Timeout Rate:** {basic['avg_timeout_rate']*100:.1f}%")
        report_lines.append(f"- **Average Service Time:** {basic['avg_service_time']:.1f} seconds")
        report_lines.append("")
        
        # 按调度器对比
        report_lines.append("## Performance by Dispatcher")
        report_lines.append("")
        
        if 'by_dispatcher' in analysis:
            df_disp = analysis['by_dispatcher']
            
            report_lines.append("| Dispatcher | Timeout Rate | Completion Rate | Avg Service Time | Avg Solve Time |")
            report_lines.append("|-----------|--------------|-----------------|------------------|----------------|")
            
            for _, row in df_disp.iterrows():
                timeout_rate = row.get('timeout_rate_mean', 0) * 100
                completion_rate = row.get('completion_rate_mean', 0) * 100
                service_time = row.get('avg_service_time_mean', 0)
                solve_time = row.get('avg_solve_time_mean', 0)
                
                report_lines.append(
                    f"| {row['dispatcher_type']} | "
                    f"{timeout_rate:.1f}% | "
                    f"{completion_rate:.1f}% | "
                    f"{service_time:.1f}s | "
                    f"{solve_time:.3f}s |"
                )
            
            report_lines.append("")
        
        # 压力测试发现
        report_lines.append("## Stress Test Findings")
        report_lines.append("")
        report_lines.append("### Timeout Rate vs. Capacity Curve")
        report_lines.append("")
        report_lines.append("Key data for plotting the timeout-capacity curve:")
        report_lines.append("")
        report_lines.append("- Data file: `timeout_vs_capacity/timeout_capacity_data.csv`")
        report_lines.append("- Separate files for each order volume available")
        report_lines.append("")
        
        # 结论
        report_lines.append("## Conclusions")
        report_lines.append("")
        report_lines.append("### Algorithm Comparison")
        report_lines.append("")
        
        # 找出最佳调度器
        if 'by_dispatcher' in analysis:
            df_disp = analysis['by_dispatcher']
            best_timeout = df_disp.loc[df_disp['timeout_rate_mean'].idxmin()]
            best_solve = df_disp.loc[df_disp['avg_solve_time_mean'].idxmin()]
            
            report_lines.append(f"- **Lowest Timeout Rate:** {best_timeout['dispatcher_type']} "
                              f"({best_timeout['timeout_rate_mean']*100:.1f}%)")
            report_lines.append(f"- **Fastest Solver:** {best_solve['dispatcher_type']} "
                              f"({best_solve['avg_solve_time_mean']:.3f}s)")
            report_lines.append("")
        
        report_lines.append("### Recommendations for Next Steps")
        report_lines.append("")
        report_lines.append("1. Visualize the timeout-capacity curves using the generated data")
        report_lines.append("2. Analyze algorithm behavior under extreme stress (1500+ orders)")
        report_lines.append("3. Consider implementing RL-based dispatching as Phase 2")
        report_lines.append("4. Prepare figures for paper submission (Fig 4: Stress Test Curve)")
        report_lines.append("")
        
        # 文件清单
        report_lines.append("## Output Files")
        report_lines.append("")
        report_lines.append("- `raw_results.csv` - Raw experiment data")
        report_lines.append("- `analysis_summary.json` - Statistical summary")
        report_lines.append("- `aggregated_by_dispatcher.csv` - Aggregated by algorithm")
        report_lines.append("- `timeout_vs_capacity/` - Timeout-capacity curve data")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_file = self.output_dir / "experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✓ 实验报告已生成: {report_file}")
        
        return report_content

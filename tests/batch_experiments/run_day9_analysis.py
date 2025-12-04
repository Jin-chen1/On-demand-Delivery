"""
Day 9 数据分析与可视化主脚本
处理Day 8批量实验数据，绘制压力测试曲线和箱线图
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.result_analyzer import DataAnalyzer, ResultCollector
from utils.visualization import ExperimentVisualizer


def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"day9_analysis_{timestamp}.log"
    
    # 清除已有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    
    return logger


def find_latest_experiment_dir(base_dir: Path) -> Path:
    """
    查找最新的实验输出目录
    
    Args:
        base_dir: 基础目录
    
    Returns:
        最新的实验目录
    """
    # 查找所有时间戳目录
    timestamp_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.replace('_', '').isdigit()]
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"在 {base_dir} 中未找到实验输出目录")
    
    # 按名称排序（时间戳格式自然排序）
    latest_dir = sorted(timestamp_dirs, key=lambda x: x.name)[-1]
    
    return latest_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Day 9: Analyze and visualize experiment results')
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing raw_results.csv (default: auto-detect latest)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for analysis and figures (default: same as input)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure resolution (default: 300)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Day 9 数据分析与可视化")
    print("任务: 处理数据，绘制压力测试曲线和箱线图")
    print("="*80)
    print()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*80)
    logger.info("Day 9 分析任务启动")
    logger.info("="*80)
    
    try:
        # ========================================
        # 阶段1: 定位数据文件
        # ========================================
        logger.info("\n阶段1: 定位数据文件")
        logger.info("-"*80)
        
        if args.input_dir:
            input_dir = Path(args.input_dir)
        else:
            # 自动查找最新的实验目录
            day8_base = project_root / "outputs" / "day8_test"
            if not day8_base.exists():
                # 尝试查找 day8_batch_experiment
                day8_base = project_root / "outputs" / "day8_batch_experiment"
            
            if not day8_base.exists():
                raise FileNotFoundError(
                    "未找到Day 8实验输出目录。请先运行 run_day8_batch.py 或使用 --input-dir 指定目录"
                )
            
            input_dir = find_latest_experiment_dir(day8_base)
        
        logger.info(f"[OK] Input directory: {input_dir}")
        
        # 检查 raw_results.csv 是否存在
        raw_results_file = input_dir / "raw_results.csv"
        if not raw_results_file.exists():
            raise FileNotFoundError(
                f"未找到 raw_results.csv 文件在 {input_dir}\n"
                "请确保Day 8批量实验已成功运行"
            )
        
        logger.info(f"[OK] Found data file: {raw_results_file}")
        
        # 设置输出目录
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = input_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[OK] Output directory: {output_dir}")
        
        # ========================================
        # 阶段2: 加载数据
        # ========================================
        logger.info("\n阶段2: 加载实验数据")
        logger.info("-"*80)
        
        results_df = pd.read_csv(raw_results_file)
        logger.info(f"[OK] Loaded {len(results_df)} experiment records")
        
        # 显示数据概览
        logger.info(f"  - 调度器类型: {results_df['dispatcher_type'].unique().tolist()}")
        logger.info(f"  - 订单量: {sorted(results_df['num_orders'].unique().tolist())}")
        logger.info(f"  - 骑手数: {sorted(results_df['num_couriers'].unique().tolist())}")
        
        # ========================================
        # 阶段3: 数据分析
        # ========================================
        logger.info("\n阶段3: 执行数据分析")
        logger.info("-"*80)
        
        analyzer = DataAnalyzer(output_dir)
        analysis = analyzer.analyze_results(results_df)
        
        # 保存分析结果
        analyzer.save_analysis_results(analysis)
        
        # 生成报告
        report_content = analyzer.generate_report(analysis, results_df)
        
        logger.info("[OK] Data analysis completed")
        
        # ========================================
        # 阶段4: 生成图表（Day 9 核心任务）
        # ========================================
        logger.info("\n阶段4: 生成可视化图表")
        logger.info("-"*80)
        
        visualizer = ExperimentVisualizer(output_dir, dpi=args.dpi)
        figures = visualizer.generate_all_figures(results_df, analysis)
        
        logger.info("[OK] All figures generated")
        
        # ========================================
        # 阶段5: 总结
        # ========================================
        logger.info("\n" + "="*80)
        logger.info("Day 9 分析任务全部完成！")
        logger.info("="*80)
        
        logger.info(f"\n输出目录: {output_dir}")
        logger.info("\n关键输出文件:")
        logger.info(f"  - 实验报告: experiment_report.md")
        logger.info(f"  - 分析摘要: analysis_summary.json")
        logger.info(f"  - 图表目录: figures/")
        
        logger.info("\n生成的图表:")
        for name, path in figures.items():
            logger.info(f"  - {name}: {path.name}")
        
        logger.info("\n论文核心图表:")
        logger.info("  [OK] Fig 4: Stress Test by Load (stress_test_by_load.png) - X=订单量（符合大纲）")
        logger.info("  [OK] Capacity Planning Curve (stress_test_curve.png) - X=骑手数量（补充）")
        logger.info("  [OK] Box Plot: Service Time Distribution (service_time_boxplot.png)")
        
        logger.info("\n下一步建议:")
        logger.info("  1. 查看 figures/ 目录中的所有图表")
        logger.info("  2. 根据需要调整图表样式或参数")
        logger.info("  3. 将核心图表用于论文撰写（Day 10）")
        logger.info("  4. 准备进入Day 10: 报告撰写与RL准备")
        
        # 打印基础统计到控制台
        print("\n" + "="*80)
        print("实验结果摘要")
        print("="*80)
        
        basic_stats = analysis['basic_stats']
        print(f"\n总实验次数: {basic_stats['total_experiments']}")
        print(f"平均完成率: {basic_stats['avg_completion_rate']*100:.1f}%")
        print(f"平均超时率: {basic_stats['avg_timeout_rate']*100:.1f}%")
        print(f"平均服务时间: {basic_stats['avg_service_time']:.1f}秒")
        
        if 'by_dispatcher' in analysis:
            print("\n按调度器性能对比:")
            disp_df = analysis['by_dispatcher']
            for _, row in disp_df.iterrows():
                print(f"  {row['dispatcher_type']:8s} - "
                      f"超时率: {row.get('timeout_rate_mean', 0)*100:5.1f}%, "
                      f"求解时间: {row.get('avg_solve_time_mean', 0)*1000:6.2f}ms")
        
        print("\n图表已保存至: " + str(output_dir / "figures"))
        print("="*80)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}")
        logger.error("请确保已运行Day 8批量实验")
        return False
        
    except Exception as e:
        logger.error(f"分析任务失败: {str(e)}")
        logger.exception("详细错误:")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断分析")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序异常退出: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

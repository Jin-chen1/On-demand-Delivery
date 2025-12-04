"""
Day 8 批量实验主脚本
自动化运行压力测试：3订单量 × 4骑手数 × 3调度器 × 2重复 = 72次实验
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices

from utils.experiment_task import ExperimentConfigManager
from utils.experiment_runner import ExperimentRunner
from utils.result_analyzer import ResultCollector, DataAnalyzer


def setup_logging(log_dir: Path) -> logging.Logger:
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"day8_batch_{timestamp}.log"
    
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


def main():
    """主函数"""
    print("="*80)
    print("Day 8 批量实验 - 压力测试")
    print("实验配置: 3订单量 × 4骑手数 × 3调度器 × 2重复 = 72次实验")
    print("="*80)
    print()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*80)
    logger.info("Day 8 批量实验启动")
    logger.info("="*80)
    
    try:
        # ========================================
        # 阶段1: 加载配置
        # ========================================
        logger.info("\n阶段1: 加载配置")
        logger.info("-"*80)
        
        config_file = Path(__file__).parent / "experiment_config.yaml"
        config_manager = ExperimentConfigManager(config_file)
        
        # 生成实验任务
        tasks = config_manager.generate_tasks()
        logger.info(f"✓ 生成了 {len(tasks)} 个实验任务")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config_manager.get_output_dir(timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ 输出目录: {output_dir}")
        
        # ========================================
        # 阶段2: 加载共享数据（路网、距离矩阵）
        # ========================================
        logger.info("\n阶段2: 加载共享数据")
        logger.info("-"*80)
        
        # 加载项目配置
        project_config = get_config()
        processed_dir = project_config.get_data_dir("processed")
        network_config = project_config.get_network_config()
        matrix_config = project_config.get_distance_matrix_config()
        
        # 加载路网
        logger.info("加载路网图...")
        graph, _ = extract_osm_network(network_config, processed_dir)
        logger.info(f"✓ 路网加载完成 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
        
        # 加载距离矩阵
        logger.info("加载距离矩阵...")
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        logger.info(f"✓ 距离矩阵加载完成 - 大小: {dist_matrix.shape}")
        
        # ========================================
        # 阶段3: 运行批量实验
        # ========================================
        logger.info("\n阶段3: 运行批量实验")
        logger.info("-"*80)
        
        execution_config = config_manager.get_execution_config()
        
        runner = ExperimentRunner(
            tasks=tasks,
            output_dir=output_dir,
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            execution_config=execution_config
        )
        
        # 运行所有实验
        logger.info("开始执行实验...")
        logger.info(f"预计总时长: 8-12小时（取决于硬件性能）")
        logger.info("")
        
        results = runner.run_all_experiments()
        
        logger.info(f"\n✓ 实验执行完成，共收集 {len(results)} 个结果")
        
        # ========================================
        # 阶段4: 保存原始结果
        # ========================================
        logger.info("\n阶段4: 保存原始结果")
        logger.info("-"*80)
        
        collector = ResultCollector(output_dir)
        
        # 保存CSV
        raw_csv = collector.save_raw_results(results, "raw_results.csv")
        
        # 保存JSON
        raw_json = collector.save_results_json(results, "raw_results.json")
        
        logger.info("✓ 原始结果保存完成")
        
        # ========================================
        # 阶段5: 分析结果
        # ========================================
        logger.info("\n阶段5: 分析结果")
        logger.info("-"*80)
        
        analyzer = DataAnalyzer(output_dir)
        
        # 读取结果数据
        results_df = pd.read_csv(raw_csv)
        
        # 执行分析
        analysis = analyzer.analyze_results(results_df)
        
        # 保存分析结果
        analyzer.save_analysis_results(analysis)
        
        # 生成报告
        report_content = analyzer.generate_report(analysis, results_df)
        
        logger.info("✓ 结果分析完成")
        
        # ========================================
        # 阶段6: 总结
        # ========================================
        logger.info("\n" + "="*80)
        logger.info("Day 8 批量实验全部完成！")
        logger.info("="*80)
        
        logger.info(f"\n输出目录: {output_dir}")
        logger.info("\n关键输出文件:")
        logger.info(f"  - 原始结果: raw_results.csv")
        logger.info(f"  - 实验报告: experiment_report.md")
        logger.info(f"  - 超时率-运力曲线数据: timeout_vs_capacity/")
        logger.info(f"  - 按调度器聚合: aggregated_by_dispatcher.csv")
        
        logger.info("\n下一步:")
        logger.info("  1. 查看实验报告了解整体结果")
        logger.info("  2. 使用timeout_vs_capacity数据绘制论文核心图表")
        logger.info("  3. 对比不同调度器在压力场景下的表现")
        
        # 打印报告摘要到控制台
        print("\n" + "="*80)
        print("实验报告摘要")
        print("="*80)
        print(report_content[:1000])  # 打印前1000字符
        print("\n... (完整报告见 experiment_report.md)")
        
        return True
        
    except Exception as e:
        logger.error(f"批量实验失败: {str(e)}")
        logger.exception("详细错误:")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断实验")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序异常退出: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

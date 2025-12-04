"""
小批量测试脚本
验证批量实验系统的基本功能
2订单量 × 2骑手数 × 2调度器 × 1重复 = 8次实验（预计15-20分钟）
"""

import sys
from pathlib import Path

# 设置路径
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))

# 导入主脚本的核心逻辑
from run_day8_batch import setup_logging, project_root
import logging
from datetime import datetime
import pandas as pd

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices

from utils.experiment_task import ExperimentConfigManager
from utils.experiment_runner import ExperimentRunner
from utils.result_analyzer import ResultCollector, DataAnalyzer


def main():
    """测试主函数"""
    print("="*80)
    print("Day 8 小批量测试")
    print("测试配置: 2订单量 × 2骑手数 × 2调度器 × 1重复 = 8次实验")
    print("预计耗时: 15-20分钟")
    print("="*80)
    print()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*80)
    logger.info("Day 8 小批量测试启动")
    logger.info("="*80)
    
    try:
        # 加载测试配置
        logger.info("\n阶段1: 加载测试配置")
        logger.info("-"*80)
        
        config_file = Path(__file__).parent / "test_config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"测试配置文件不存在: {config_file}")
        
        config_manager = ExperimentConfigManager(config_file)
        tasks = config_manager.generate_tasks()
        logger.info(f"✓ 生成了 {len(tasks)} 个测试任务")
        
        if len(tasks) != 8:
            logger.warning(f"预期8个任务，实际生成{len(tasks)}个")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config_manager.get_output_dir(timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ 测试输出目录: {output_dir}")
        
        # 加载共享数据
        logger.info("\n阶段2: 加载共享数据")
        logger.info("-"*80)
        
        project_config = get_config()
        processed_dir = project_config.get_data_dir("processed")
        network_config = project_config.get_network_config()
        matrix_config = project_config.get_distance_matrix_config()
        
        logger.info("加载路网...")
        graph, _ = extract_osm_network(network_config, processed_dir)
        logger.info(f"✓ 路网加载完成")
        
        logger.info("加载距离矩阵...")
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        logger.info(f"✓ 距离矩阵加载完成")
        
        # 运行测试实验
        logger.info("\n阶段3: 运行测试实验")
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
        
        logger.info("开始执行测试...")
        results = runner.run_all_experiments()
        
        logger.info(f"\n✓ 测试完成，共收集 {len(results)} 个结果")
        
        # 验证结果
        logger.info("\n阶段4: 验证结果")
        logger.info("-"*80)
        
        success_count = len([r for r in results if r is not None])
        expected_count = len(tasks)
        
        logger.info(f"成功实验: {success_count}/{expected_count}")
        
        if success_count < expected_count:
            logger.warning(f"有 {expected_count - success_count} 个实验失败")
        else:
            logger.info("✓ 所有测试实验成功完成")
        
        # 保存结果
        logger.info("\n阶段5: 保存测试结果")
        logger.info("-"*80)
        
        collector = ResultCollector(output_dir)
        raw_csv = collector.save_raw_results(results, "test_results.csv")
        collector.save_results_json(results, "test_results.json")
        
        # 简单分析
        logger.info("\n阶段6: 分析测试结果")
        logger.info("-"*80)
        
        analyzer = DataAnalyzer(output_dir)
        results_df = pd.read_csv(raw_csv)
        
        analysis = analyzer.analyze_results(results_df)
        analyzer.save_analysis_results(analysis)
        analyzer.generate_report(analysis, results_df)
        
        logger.info("✓ 测试分析完成")
        
        # 总结
        logger.info("\n" + "="*80)
        logger.info("小批量测试完成！")
        logger.info("="*80)
        
        logger.info(f"\n测试输出: {output_dir}")
        logger.info("\n基本验证:")
        logger.info(f"  ✓ 任务生成: {len(tasks)}个任务")
        logger.info(f"  ✓ 实验执行: {success_count}/{expected_count}成功")
        logger.info(f"  ✓ 结果保存: CSV和JSON格式")
        logger.info(f"  ✓ 数据分析: 聚合和报告生成")
        
        if success_count == expected_count:
            logger.info("\n✅ 所有功能验证通过，可以运行完整实验！")
            return True
        else:
            logger.warning("\n⚠️  部分实验失败，请检查日志后再运行完整实验")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        logger.exception("详细错误:")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试异常退出: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
数据准备主执行脚本
一键运行OSM路网提取、距离矩阵计算和订单生成
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.data_preparation.order_generator import generate_orders


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_preparation_{timestamp}.log"
    
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


def main(force_download: bool = False, force_recalculate: bool = False):
    """
    主执行函数
    
    Args:
        force_download: 是否强制重新下载路网
        force_recalculate: 是否强制重新计算距离矩阵
    """
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("开始数据准备流程")
    logger.info("="*60)
    
    # 获取配置
    network_config = config.get_network_config()
    matrix_config = config.get_distance_matrix_config()
    order_config = config.get_order_generation_config()
    random_seed = config.get_random_seed()
    
    # 输出目录
    processed_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    logger.info(f"数据输出目录: {processed_dir}")
    logger.info(f"订单输出目录: {orders_dir}")
    
    try:
        # 步骤1: 提取OSM路网
        logger.info("\n" + "="*60)
        logger.info("步骤1/3: 提取OSM路网")
        logger.info("="*60)
        
        graph, network_files = extract_osm_network(
            network_config,
            output_dir=processed_dir,
            force_download=force_download
        )
        
        logger.info(f"✓ 路网提取完成 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
        
        # 步骤2: 计算距离矩阵
        logger.info("\n" + "="*60)
        logger.info("步骤2/3: 计算距离矩阵")
        logger.info("="*60)
        
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph,
            matrix_config,
            output_dir=processed_dir,
            force_recalculate=force_recalculate
        )
        
        logger.info(f"✓ 距离矩阵计算完成 - 大小: {dist_matrix.shape}")
        
        # 步骤3: 生成订单
        logger.info("\n" + "="*60)
        logger.info("步骤3/3: 生成订单流")
        logger.info("="*60)
        
        orders, order_files = generate_orders(
            graph,
            order_config,
            node_list=mapping['node_list'],
            output_dir=orders_dir,
            random_seed=random_seed
        )
        
        logger.info(f"✓ 订单生成完成 - 数量: {len(orders)}")
        
        # 总结
        logger.info("\n" + "="*60)
        logger.info("数据准备流程完成！")
        logger.info("="*60)
        
        logger.info("\n生成的文件:")
        logger.info("\n路网文件:")
        for key, path in network_files.items():
            logger.info(f"  {key}: {path}")
        
        logger.info("\n订单文件:")
        for key, path in order_files.items():
            logger.info(f"  {key}: {path}")
        
        logger.info("\n数据准备完成，可以开始仿真实验！")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ 数据准备失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="即时配送仿真系统 - 数据准备"
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='强制重新下载OSM路网（即使已存在）'
    )
    
    parser.add_argument(
        '--force-recalculate',
        action='store_true',
        help='强制重新计算距离矩阵（即使已存在）'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='强制重新生成所有数据'
    )
    
    args = parser.parse_args()
    
    # 如果指定了--all，则强制重新生成所有数据
    force_download = args.force_download or args.all
    force_recalculate = args.force_recalculate or args.all
    
    success = main(
        force_download=force_download,
        force_recalculate=force_recalculate
    )
    
    sys.exit(0 if success else 1)

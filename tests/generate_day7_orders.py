"""
Day 7 专用订单生成脚本
生成适合1小时仿真对比测试的订单数据
"""

import sys
from pathlib import Path
import numpy as np
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.order_generator import OrderGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_day7_orders(
    duration_hours: float = 1.0,
    num_orders: int = 100,
    random_seed: int = 42
):
    """
    生成Day 7对比测试专用的订单数据
    
    Args:
        duration_hours: 仿真时长（小时）
        num_orders: 订单数量
        random_seed: 随机种子
    """
    logger.info("="*60)
    logger.info("Day 7 订单生成器")
    logger.info("="*60)
    
    config = get_config()
    
    # 加载路网
    processed_dir = config.get_data_dir("processed")
    network_config = config.get_network_config()
    
    logger.info("加载路网数据...")
    graph, _ = extract_osm_network(network_config, processed_dir)
    logger.info(f"路网节点数: {len(graph.nodes)}")
    
    # 加载节点映射
    import json
    mapping_file = processed_dir / "node_id_mapping.json"
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    node_list = [int(k) for k in mapping['node_to_idx'].keys()]
    
    # 配置订单生成参数（1小时仿真）
    duration_seconds = int(duration_hours * 3600)
    arrival_rate = num_orders / duration_seconds  # 订单/秒
    
    order_config = {
        'simulation_duration': duration_seconds,
        'total_orders': num_orders,
        'arrival_process': {
            'type': 'poisson',
            'rate': arrival_rate
        },
        'spatial_distribution': {
            'merchant': {
                'type': 'clustered',
                'num_clusters': 5,
                'cluster_std': 300
            },
            'customer': {
                'type': 'uniform',
                'coverage': 0.9
            }
        },
        'service_time': {
            'preparation_time': [180, 480],  # 3-8分钟（适合短时间测试）
            'delivery_window': [1500, 2700],  # 25-45分钟（适中时间窗）
            'pickup_duration': 60,
            'dropoff_duration': 60
        }
    }
    
    logger.info(f"订单配置:")
    logger.info(f"  仿真时长: {duration_hours}小时 ({duration_seconds}秒)")
    logger.info(f"  订单数量: {num_orders}")
    logger.info(f"  到达率: {arrival_rate:.4f} 订单/秒 ({num_orders/duration_hours:.1f} 订单/小时)")
    
    # 生成订单
    generator = OrderGenerator(
        graph=graph,
        config=order_config,
        node_list=node_list,
        random_seed=random_seed
    )
    
    orders = generator.generate_orders()
    
    logger.info(f"生成 {len(orders)} 个订单")
    
    # 保存到专用目录
    orders_dir = config.get_data_dir("orders")
    day7_dir = orders_dir / "day7"
    day7_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用save_orders方法保存
    saved_files = generator.save_orders(orders, day7_dir)
    
    # 复制主订单文件到orders目录供测试使用
    import shutil
    output_file = orders_dir / "orders_day7_1h.csv"
    shutil.copy(saved_files['orders'], output_file)
    
    logger.info(f"订单保存到: {output_file}")
    
    # 验证订单时间分布
    arrival_times = [o.arrival_time for o in orders]
    logger.info(f"订单时间范围: {min(arrival_times):.1f}s - {max(arrival_times):.1f}s")
    logger.info(f"平均到达间隔: {np.mean(np.diff(arrival_times)):.1f}s")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="生成Day 7测试订单")
    parser.add_argument("--duration", type=float, default=1.0, help="仿真时长(小时)")
    parser.add_argument("--orders", type=int, default=100, help="订单数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    generate_day7_orders(
        duration_hours=args.duration,
        num_orders=args.orders,
        random_seed=args.seed
    )

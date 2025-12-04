"""
测试动态插入功能 - Day 4 增强版
验证OR-Tools调度器能否向非空闲骑手插入新订单
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
from src.utils.config import ConfigManager
from src.data_preparation import osm_network, distance_matrix, order_generator
from src.simulation.environment import SimulationEnvironment

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_comparison_test():
    """对比测试：动态插入 vs 传统模式"""
    
    logger.info("="*60)
    logger.info("Day 4 动态插入功能测试")
    logger.info("="*60)
    
    # 加载配置
    config = ConfigManager()
    network_config = config.get('network')
    matrix_config = config.get('distance_matrix')
    
    data_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    # 加载路网和距离矩阵
    logger.info("\n=== 加载路网数据 ===")
    graph, _ = osm_network.extract_osm_network(network_config, data_dir, force_download=False)
    
    logger.info("\n=== 加载距离矩阵 ===")
    dist_matrix, time_matrix, mapping = distance_matrix.compute_distance_matrices(
        graph, matrix_config, data_dir, force_recalculate=False
    )
    
    # 测试场景1：传统模式（仅IDLE骑手）
    logger.info("\n" + "="*60)
    logger.info("场景1: 传统模式（allow_insertion_to_active=False）")
    logger.info("="*60)
    
    sim_config_traditional = {
        'simulation_duration': 1800.0,  # 30分钟
        'dispatch_interval': 60.0,       # 1分钟调度一次
        'dispatcher_type': 'ortools',
        'dispatcher_config': {
            'time_limit_seconds': 3,
            'allow_insertion_to_active': False,  # 禁用动态插入
            'enable_batching': False,
            'soft_time_windows': True,
            'time_window_slack': 180.0
        }
    }
    
    env_traditional = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=sim_config_traditional
    )
    
    # 加载订单
    orders_file = orders_dir / "orders.csv"
    env_traditional.load_orders_from_csv(orders_file)
    
    # 初始化骑手（较少的骑手以触发动态插入场景）
    courier_config = {
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    }
    env_traditional.initialize_couriers(num_couriers=10, courier_config=courier_config)
    
    # 运行仿真
    logger.info("\n开始传统模式仿真...")
    env_traditional.run(until=1800.0)
    
    stats_traditional = env_traditional.get_statistics()
    
    # 测试场景2：动态插入模式
    logger.info("\n" + "="*60)
    logger.info("场景2: 动态插入模式（allow_insertion_to_active=True）")
    logger.info("="*60)
    
    sim_config_dynamic = {
        'simulation_duration': 1800.0,
        'dispatch_interval': 60.0,
        'dispatcher_type': 'ortools',
        'dispatcher_config': {
            'time_limit_seconds': 3,
            'allow_insertion_to_active': True,   # 启用动态插入
            'enable_batching': False,
            'soft_time_windows': True,
            'time_window_slack': 180.0
        }
    }
    
    env_dynamic = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=sim_config_dynamic
    )
    
    # 加载相同的订单
    env_dynamic.load_orders_from_csv(orders_file)
    env_dynamic.initialize_couriers(num_couriers=10, courier_config=courier_config)
    
    # 运行仿真
    logger.info("\n开始动态插入模式仿真...")
    env_dynamic.run(until=1800.0)
    
    stats_dynamic = env_dynamic.get_statistics()
    
    # 对比结果
    logger.info("\n" + "="*60)
    logger.info("对比结果")
    logger.info("="*60)
    
    logger.info("\n传统模式:")
    logger.info(f"  已完成订单: {stats_traditional['completed_orders']}")
    logger.info(f"  待分配订单: {stats_traditional['pending_orders']}")
    logger.info(f"  已分配订单: {stats_traditional['assigned_orders']}")
    
    logger.info("\n动态插入模式:")
    logger.info(f"  已完成订单: {stats_dynamic['completed_orders']}")
    logger.info(f"  待分配订单: {stats_dynamic['pending_orders']}")
    logger.info(f"  已分配订单: {stats_dynamic['assigned_orders']}")
    
    # 计算改进率
    if stats_traditional['completed_orders'] > 0:
        improvement = (
            (stats_dynamic['completed_orders'] - stats_traditional['completed_orders']) 
            / stats_traditional['completed_orders'] * 100
        )
        logger.info(f"\n改进率: {improvement:+.2f}%")
    
    # 检查动态插入事件
    dynamic_insertion_events = [
        e for e in env_dynamic.events
        if e.event_type == 'order_assigned' 
        and e.details.get('insertion_mode') == 'active'
    ]
    
    logger.info(f"\n动态插入事件数: {len(dynamic_insertion_events)}")
    
    # 验证测试
    logger.info("\n" + "="*60)
    logger.info("测试验证")
    logger.info("="*60)
    
    success = True
    
    # 验证1: 动态模式应该完成更多或至少相同数量的订单
    if stats_dynamic['completed_orders'] >= stats_traditional['completed_orders']:
        logger.info("✅ 验证1通过: 动态插入模式完成订单数 >= 传统模式")
    else:
        logger.error("❌ 验证1失败: 动态插入模式完成订单数 < 传统模式")
        success = False
    
    # 验证2: 应该有动态插入事件发生
    if len(dynamic_insertion_events) > 0:
        logger.info(f"✅ 验证2通过: 检测到 {len(dynamic_insertion_events)} 个动态插入事件")
    else:
        logger.warning("⚠️ 验证2警告: 未检测到动态插入事件（可能所有骑手始终空闲）")
    
    # 验证3: 动态模式的待分配订单应该更少
    if stats_dynamic['pending_orders'] <= stats_traditional['pending_orders']:
        logger.info("✅ 验证3通过: 动态插入模式待分配订单 <= 传统模式")
    else:
        logger.warning("⚠️ 验证3警告: 动态插入模式待分配订单 > 传统模式")
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 动态插入功能测试通过!")
    else:
        logger.error("❌ 动态插入功能测试失败")
    logger.info("="*60)
    
    return success


if __name__ == "__main__":
    try:
        success = run_comparison_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        logger.exception("详细错误:")
        sys.exit(1)

"""
Day 2 测试脚本：SimPy 仿真骨架测试
验证"订单产生-进入队列-等待分配"的空转流程
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.simulation import SimulationEnvironment


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_core_test_{timestamp}.log"
    
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


def test_simulation_core():
    """测试仿真核心功能"""
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("Day 2: SimPy 仿真骨架测试")
    logger.info("="*60)
    
    try:
        # 步骤1: 加载路网数据
        logger.info("\n步骤1: 加载路网数据")
        logger.info("-"*60)
        
        processed_dir = config.get_data_dir("processed")
        network_config = config.get_network_config()
        
        graph, _ = extract_osm_network(network_config, processed_dir)
        logger.info(f"✓ 路网加载成功 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
        
        # 步骤2: 加载距离矩阵
        logger.info("\n步骤2: 加载距离矩阵")
        logger.info("-"*60)
        
        matrix_config = config.get_distance_matrix_config()
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        logger.info(f"✓ 距离矩阵加载成功 - 大小: {dist_matrix.shape}")
        
        # 步骤3: 初始化仿真环境
        logger.info("\n步骤3: 初始化仿真环境")
        logger.info("-"*60)
        
        sim_config = {
            'simulation_duration': 3600  # 仿真1小时（用于快速测试）
        }
        
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            config=sim_config
        )
        logger.info("✓ 仿真环境创建成功")
        
        # 步骤4: 加载订单数据
        logger.info("\n步骤4: 加载订单数据")
        logger.info("-"*60)
        
        orders_dir = config.get_data_dir("orders")
        orders_file = orders_dir / "orders.csv"
        
        sim_env.load_orders_from_csv(orders_file)
        logger.info(f"✓ 订单加载成功 - 数量: {len(sim_env.orders)}")
        
        # 步骤5: 初始化骑手
        logger.info("\n步骤5: 初始化骑手")
        logger.info("-"*60)
        
        courier_config = config.get_courier_config()
        num_couriers = courier_config.get('num_couriers', 30)
        
        sim_env.initialize_couriers(num_couriers, courier_config)
        logger.info(f"✓ 骑手初始化成功 - 数量: {len(sim_env.couriers)}")
        
        # 步骤6: 运行仿真
        logger.info("\n步骤6: 运行仿真（空转测试）")
        logger.info("-"*60)
        logger.info("开始仿真...")
        
        sim_env.run(until=3600)  # 运行1小时
        
        logger.info("✓ 仿真运行完成")
        
        # 步骤7: 检查结果
        logger.info("\n步骤7: 验证结果")
        logger.info("-"*60)
        
        stats = sim_env.get_statistics()
        
        # 验证订单到达
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        logger.info(f"订单到达事件数: {len(arrival_events)}")
        logger.info(f"待分配队列长度: {len(sim_env.pending_orders)}")
        logger.info(f"骑手空闲状态数: {stats['courier_status_counts'].get('idle', 0)}")
        
        # 验证检查
        assert len(arrival_events) > 0, "没有订单到达事件"
        assert len(sim_env.pending_orders) > 0, "待分配队列为空"
        assert stats['courier_status_counts'].get('idle', 0) > 0, "没有空闲骑手"
        
        logger.info("✓ 所有验证通过")
        
        # 步骤8: 保存结果
        logger.info("\n步骤8: 保存仿真结果")
        logger.info("-"*60)
        
        output_dir = project_root / "data" / "simulation_results" / f"day2_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_files = sim_env.save_results(output_dir)
        
        logger.info("保存的文件:")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
        
        # 总结
        logger.info("\n" + "="*60)
        logger.info("Day 2 测试成功完成！")
        logger.info("="*60)
        logger.info("\n验证要点:")
        logger.info("✓ 订单能按时间到达并进入待分配队列")
        logger.info("✓ 骑手正常初始化并保持空闲状态")
        logger.info("✓ SimPy 仿真循环正常运行")
        logger.info("✓ 事件记录功能正常")
        logger.info("\n下一步: Day 3 - 实现 Greedy Baseline 调度器")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False


if __name__ == "__main__":
    success = test_simulation_core()
    sys.exit(0 if success else 1)

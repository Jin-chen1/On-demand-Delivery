"""
Day 3 测试脚本：Greedy Dispatcher 功能验证
诊断并修复调度器问题
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
    log_file = log_dir / f"day3_greedy_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,  # 改为DEBUG级别以获取详细日志
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    
    return logger


def test_greedy_dispatcher():
    """测试Greedy调度器"""
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("Day 3: Greedy Dispatcher 测试")
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
        logger.info(f"  采样节点数: {len(mapping['node_list'])}")
        
        # 步骤3: 初始化仿真环境（关键修改：缩短调度间隔）
        logger.info("\n步骤3: 初始化仿真环境")
        logger.info("-"*60)
        
        sim_config = {
            'simulation_duration': 3600,  # 仿真1小时
            'dispatch_interval': 10.0     # 🔧 改为10秒调度一次
        }
        
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            config=sim_config
        )
        logger.info("✓ 仿真环境创建成功")
        logger.info(f"  调度间隔: {sim_env.dispatch_interval}秒")
        
        # 步骤4: 加载订单数据
        logger.info("\n步骤4: 加载订单数据")
        logger.info("-"*60)
        
        orders_dir = config.get_data_dir("orders")
        orders_file = orders_dir / "orders.csv"
        
        sim_env.load_orders_from_csv(orders_file)
        logger.info(f"✓ 订单加载成功 - 数量: {len(sim_env.orders)}")
        
        # 🔍 诊断：检查订单节点是否在采样节点中
        node_set = set(mapping['node_list'])
        merchant_nodes_in_mapping = sum(
            1 for order in sim_env.orders.values() 
            if order.merchant_node in node_set
        )
        customer_nodes_in_mapping = sum(
            1 for order in sim_env.orders.values() 
            if order.customer_node in node_set
        )
        
        logger.info(f"  订单节点覆盖率:")
        logger.info(f"    商家节点在采样中: {merchant_nodes_in_mapping}/{len(sim_env.orders)} ({merchant_nodes_in_mapping/len(sim_env.orders)*100:.1f}%)")
        logger.info(f"    客户节点在采样中: {customer_nodes_in_mapping}/{len(sim_env.orders)} ({customer_nodes_in_mapping/len(sim_env.orders)*100:.1f}%)")
        
        if merchant_nodes_in_mapping < len(sim_env.orders) * 0.8:
            logger.warning("⚠️  警告：订单节点覆盖率低于80%，可能导致调度失败")
        
        # 步骤5: 初始化骑手
        logger.info("\n步骤5: 初始化骑手")
        logger.info("-"*60)
        
        courier_config = config.get_courier_config()
        num_couriers = 20  # 🔧 减少骑手数量以便观察
        
        sim_env.initialize_couriers(num_couriers, courier_config)
        logger.info(f"✓ 骑手初始化成功 - 数量: {len(sim_env.couriers)}")
        
        # 🔍 检查骑手初始位置是否在采样节点中
        couriers_in_mapping = sum(
            1 for courier in sim_env.couriers.values()
            if courier.current_node in node_set
        )
        logger.info(f"  骑手位置覆盖率: {couriers_in_mapping}/{num_couriers} ({couriers_in_mapping/num_couriers*100:.1f}%)")
        
        # 步骤6: 运行仿真
        logger.info("\n步骤6: 运行仿真")
        logger.info("-"*60)
        logger.info("开始仿真...")
        
        sim_env.run(until=3600)  # 运行1小时
        
        logger.info("✓ 仿真运行完成")
        
        # 步骤7: 详细分析结果
        logger.info("\n步骤7: 结果分析")
        logger.info("-"*60)
        
        stats = sim_env.get_statistics()
        
        # 统计各类事件
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        assigned_events = [e for e in sim_env.events if e.event_type == 'order_assigned']
        pickup_events = [e for e in sim_env.events if e.event_type == 'pickup_complete']
        delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
        
        logger.info(f"事件统计:")
        logger.info(f"  订单到达: {len(arrival_events)}")
        logger.info(f"  订单分配: {len(assigned_events)}")
        logger.info(f"  取货完成: {len(pickup_events)}")
        logger.info(f"  配送完成: {len(delivery_events)}")
        
        logger.info(f"\n订单状态:")
        logger.info(f"  待分配: {len(sim_env.pending_orders)}")
        logger.info(f"  已分配: {len(sim_env.assigned_orders)}")
        logger.info(f"  已完成: {len(sim_env.completed_orders)}")
        logger.info(f"  完成率: {len(sim_env.completed_orders)/len(arrival_events)*100:.1f}%")
        
        logger.info(f"\n骑手状态:")
        logger.info(f"  {stats['courier_status_counts']}")
        
        # 计算性能指标
        if len(delivery_events) > 0:
            completed_order_ids = [e.entity_id for e in delivery_events]
            timeout_count = sum(
                1 for oid in completed_order_ids
                if sim_env.orders[oid].is_timeout(sim_env.env.now)
            )
            logger.info(f"\n性能指标:")
            logger.info(f"  超时订单: {timeout_count}/{len(delivery_events)}")
            logger.info(f"  超时率: {timeout_count/len(delivery_events)*100:.1f}%")
            
            # 平均配送时长
            service_times = [
                sim_env.orders[oid].get_total_service_time()
                for oid in completed_order_ids
            ]
            avg_service_time = sum(service_times) / len(service_times)
            logger.info(f"  平均服务时长: {avg_service_time:.1f}秒 ({avg_service_time/60:.1f}分钟)")
        
        # 验证检查
        logger.info("\n步骤8: 验证检查")
        logger.info("-"*60)
        
        if len(assigned_events) == 0:
            logger.error("❌ 失败：没有订单被分配")
            logger.error("   可能原因：")
            logger.error("   1. 调度器未被触发")
            logger.error("   2. 订单节点不在距离矩阵采样范围内")
            logger.error("   3. 所有骑手都不可用")
            return False
        
        if len(delivery_events) == 0:
            logger.warning("⚠️  警告：有订单分配但没有完成配送")
            logger.warning("   可能原因：仿真时间不够长")
        
        logger.info(f"✓ 订单分配功能正常 ({len(assigned_events)}个订单已分配)")
        
        if len(delivery_events) > 0:
            logger.info(f"✓ 配送功能正常 ({len(delivery_events)}个订单已完成)")
        
        # 步骤9: 保存结果
        logger.info("\n步骤9: 保存仿真结果")
        logger.info("-"*60)
        
        output_dir = project_root / "data" / "simulation_results" / f"day3_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_files = sim_env.save_results(output_dir)
        
        logger.info("保存的文件:")
        for key, path in saved_files.items():
            logger.info(f"  {key}: {path}")
        
        # 总结
        logger.info("\n" + "="*60)
        logger.info("Day 3 测试完成！")
        logger.info("="*60)
        
        if len(assigned_events) > 0 and len(delivery_events) > 0:
            logger.info("\n✅ 测试成功:")
            logger.info(f"  ✓ {len(assigned_events)} 个订单成功分配")
            logger.info(f"  ✓ {len(delivery_events)} 个订单成功配送")
            logger.info(f"  ✓ Greedy调度器工作正常")
            return True
        else:
            logger.warning("\n⚠️  测试部分成功:")
            logger.warning(f"  订单分配: {len(assigned_events)}")
            logger.warning(f"  订单完成: {len(delivery_events)}")
            return False
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False


if __name__ == "__main__":
    success = test_greedy_dispatcher()
    sys.exit(0 if success else 1)

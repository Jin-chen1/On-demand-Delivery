"""
测试 ALNS 调度器 - Day 6
验证ALNS调度器的基本功能
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.simulation import SimulationEnvironment
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.utils.config import get_config


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"day6_alns_{timestamp}.log"
    
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


def test_alns_basic():
    """测试ALNS基本调度功能"""
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("\n" + "="*70)
    logger.info("测试 ALNS 调度器 - Day 6")
    logger.info("="*70)
    
    # 2. 加载路网数据
    logger.info("\n步骤1: 加载路网数据")
    logger.info("-"*70)
    
    processed_dir = config.get_data_dir("processed")
    network_config = config.get_network_config()
    
    graph, _ = extract_osm_network(network_config, processed_dir)
    logger.info(f"✓ 路网加载成功 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
    
    # 3. 加载距离矩阵
    logger.info("\n步骤2: 加载距离矩阵")
    logger.info("-"*70)
    
    matrix_config = config.get_distance_matrix_config()
    dist_matrix, time_matrix, mapping = compute_distance_matrices(
        graph, matrix_config, processed_dir
    )
    logger.info(f"✓ 距离矩阵加载成功 - 大小: {dist_matrix.shape}")
    
    # 4. 创建仿真环境（使用ALNS调度器）
    logger.info("\n步骤3: 初始化仿真环境")
    logger.info("-"*70)
    
    sim_config = {
        'simulation_duration': 43200,  # 仿真12小时（与OR-Tools测试一致）
        'dispatch_interval': 60.0,
        'use_gps_coords': False,  # 使用路网最短路径距离（与Day 7一致）
        'dispatcher_type': 'alns',  # 使用ALNS调度器
        'dispatcher_config': {
            'iterations': 100,  # 增加迭代次数以获得更好结果
            'destroy_degree_min': 0.1,
            'destroy_degree_max': 0.3,
            'temperature_start': 5000.0,
            'temperature_end': 1.0,
            'temperature_decay': 0.95,
            'random_seed': 42
        }
    }
    
    env = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=sim_config
    )
    logger.info("✓ 仿真环境创建成功")
    logger.info(f"  调度器类型: ALNS")
    
    # 5. 加载订单数据
    logger.info("\n步骤4: 加载订单数据")
    logger.info("-"*70)
    
    orders_dir = config.get_data_dir("orders")
    
    # 使用均匀网格采样数据（与OR-Tools测试一致）
    uniform_orders_file = orders_dir / "uniform_grid_100.csv"
    if uniform_orders_file.exists():
        orders_file = uniform_orders_file
        logger.info("使用均匀网格采样订单数据（uniform_grid_100.csv）")
    else:
        orders_file = orders_dir / "orders.csv"
        logger.info("使用默认模拟订单数据")
    
    env.load_orders_from_csv(orders_file)
    logger.info(f"✓ 订单加载成功 - 数量: {len(env.orders)}")
    
    # 检查订单节点覆盖率
    node_set = set(mapping['node_list'])
    merchant_nodes_in_mapping = sum(
        1 for order in env.orders.values() 
        if order.merchant_node in node_set
    )
    customer_nodes_in_mapping = sum(
        1 for order in env.orders.values() 
        if order.customer_node in node_set
    )
    
    logger.info(f"  订单节点覆盖率:")
    logger.info(f"    商家节点: {merchant_nodes_in_mapping}/{len(env.orders)} ({merchant_nodes_in_mapping/len(env.orders)*100:.1f}%)")
    logger.info(f"    客户节点: {customer_nodes_in_mapping}/{len(env.orders)} ({customer_nodes_in_mapping/len(env.orders)*100:.1f}%)")
    
    # 调整订单到达时间到仿真范围内（与OR-Tools测试一致）
    arrival_times = [order.arrival_time for order in env.orders.values()]
    min_arrival = min(arrival_times)
    max_arrival = max(arrival_times)
    simulation_duration = sim_config['simulation_duration']
    
    if max_arrival > simulation_duration * 0.7:
        target_max = simulation_duration * 0.7
        for order in env.orders.values():
            if max_arrival > min_arrival:
                order.arrival_time = (order.arrival_time - min_arrival) / (max_arrival - min_arrival) * target_max
            else:
                order.arrival_time = 0
        logger.info(f"  订单到达时间已调整: [{min_arrival:.0f}s-{max_arrival:.0f}s] -> [0s-{target_max:.0f}s]")
    
    # 6. 初始化骑手
    logger.info("\n步骤5: 初始化骑手")
    logger.info("-"*70)
    
    courier_config = config.get_courier_config()
    num_couriers = 20  # 与OR-Tools测试一致
    
    env.initialize_couriers(num_couriers, courier_config)
    logger.info(f"✓ 骑手初始化成功 - 数量: {len(env.couriers)}")
    
    # 7. 运行仿真
    logger.info("\n步骤6: 运行仿真")
    logger.info("-"*70)
    logger.info("开始仿真...\n")
    
    env.run(until=43200)  # 运行12小时（覆盖所有订单到达时间）
    
    logger.info("\n✓ 仿真完成")
    
    # 8. 详细结果分析
    logger.info("\n步骤7: 结果分析")
    logger.info("-"*70)
    
    stats = env.get_statistics()
    
    # 统计各类事件
    arrival_events = [e for e in env.events if e.event_type == 'order_arrival']
    assigned_events = [e for e in env.events if e.event_type == 'order_assigned']
    pickup_events = [e for e in env.events if e.event_type == 'pickup_complete']
    delivery_events = [e for e in env.events if e.event_type == 'delivery_complete']
    
    logger.info(f"事件统计:")
    logger.info(f"  订单到达: {len(arrival_events)}")
    logger.info(f"  订单分配: {len(assigned_events)}")
    logger.info(f"  取货完成: {len(pickup_events)}")
    logger.info(f"  配送完成: {len(delivery_events)}")
    
    logger.info(f"\n订单状态:")
    logger.info(f"  待分配: {len(env.pending_orders)}")
    logger.info(f"  已分配: {len(env.assigned_orders)}")
    logger.info(f"  已完成: {len(env.completed_orders)}")
    if len(arrival_events) > 0:
        logger.info(f"  完成率: {len(env.completed_orders)/len(arrival_events)*100:.1f}%")
    
    # 计算性能指标
    if len(delivery_events) > 0:
        completed_order_ids = [e.entity_id for e in delivery_events]
        timeout_count = sum(
            1 for oid in completed_order_ids
            if env.orders[oid].is_timeout(env.env.now)
        )
        
        logger.info(f"\n性能指标:")
        logger.info(f"  超时订单: {timeout_count}/{len(delivery_events)}")
        logger.info(f"  超时率: {timeout_count/len(delivery_events)*100:.1f}%")
        
        # 平均配送时长
        service_times = [
            env.orders[oid].get_total_service_time()
            for oid in completed_order_ids
        ]
        avg_service_time = sum(service_times) / len(service_times)
        logger.info(f"  平均服务时长: {avg_service_time:.1f}秒 ({avg_service_time/60:.1f}分钟)")
        
        # 骑手利用率
        total_utilization = sum(c.get_utilization() for c in env.couriers.values())
        avg_utilization = total_utilization / len(env.couriers)
        logger.info(f"  平均骑手利用率: {avg_utilization*100:.1f}%")
    
    # ALNS 调度器统计
    dispatcher_stats = env.dispatcher.get_statistics()
    logger.info(f"\nALNS 调度器统计:")
    logger.info(f"  调度次数: {dispatcher_stats.get('dispatch_count', 0)}")
    logger.info(f"  总分配订单: {dispatcher_stats.get('total_assigned', 0)}")
    logger.info(f"  总迭代次数: {dispatcher_stats.get('total_iterations', 0)}")
    logger.info(f"  平均每次调度时间: {dispatcher_stats.get('avg_dispatch_time', 0):.3f}秒")
    
    # 验证检查
    logger.info("\n步骤8: 验证检查")
    logger.info("-"*70)
    
    success = True
    
    if len(assigned_events) == 0:
        logger.error("❌ 失败：没有订单被分配")
        success = False
    else:
        logger.info(f"✓ 订单分配功能正常 ({len(assigned_events)}个订单已分配)")
    
    if len(delivery_events) == 0:
        logger.warning("⚠️  警告：有订单分配但没有完成配送（可能需要更长仿真时间）")
    else:
        logger.info(f"✓ 配送功能正常 ({len(delivery_events)}个订单已完成)")
    
    if dispatcher_stats.get('dispatch_count', 0) == 0:
        logger.error("❌ 失败：ALNS 从未执行调度")
        success = False
    else:
        logger.info(f"✓ ALNS 调度功能正常 (执行{dispatcher_stats.get('dispatch_count', 0)}次)")
    
    # 步骤9: 保存结果
    logger.info("\n步骤9: 保存仿真结果")
    logger.info("-"*70)
    
    output_dir = project_root / "outputs" / "simulation_results" / f"day6_alns_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_files = env.save_results(output_dir)
    
    # 保存调度器统计
    dispatcher_stats_file = output_dir / "dispatcher_statistics.json"
    with open(dispatcher_stats_file, 'w', encoding='utf-8') as f:
        json.dump(dispatcher_stats, f, indent=2, ensure_ascii=False)
    saved_files['dispatcher_stats'] = dispatcher_stats_file
    
    # 保存详细性能指标（与RL测试一致）
    total_orders = len(arrival_events) if arrival_events else len(env.orders)
    timeout_count = 0
    service_times_list = []
    if delivery_events:
        completed_order_ids = [e.entity_id for e in delivery_events]
        for oid in completed_order_ids:
            order = env.orders.get(oid)
            if order:
                if order.is_timeout(env.env.now):
                    timeout_count += 1
                if order.delivery_complete_time is not None:
                    service_time = order.delivery_complete_time - order.arrival_time
                    service_times_list.append(service_time)
    
    avg_service_time_val = sum(service_times_list) / len(service_times_list) if service_times_list else 0
    timeout_rate_val = timeout_count / len(delivery_events) if delivery_events else 0
    total_utilization = sum(c.get_utilization() for c in env.couriers.values())
    avg_utilization = total_utilization / len(env.couriers) if env.couriers else 0
    
    performance_info = {
        'total_orders': total_orders,
        'completed_orders': len(env.completed_orders),
        'pending_orders': len(env.pending_orders),
        'timeout_orders': timeout_count,
        'completion_rate': len(env.completed_orders) / total_orders if total_orders > 0 else 0,
        'timeout_rate': timeout_rate_val,
        'avg_service_time': avg_service_time_val,
        'avg_service_time_minutes': avg_service_time_val / 60,
        'avg_courier_utilization': avg_utilization
    }
    performance_file = output_dir / "performance_info.json"
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(performance_info, f, indent=2, ensure_ascii=False)
    saved_files['performance_info'] = performance_file
    
    logger.info("保存的文件:")
    for key, path in saved_files.items():
        logger.info(f"  {key}: {path}")
    
    # 总结
    logger.info("\n" + "="*70)
    logger.info("Day 6 ALNS 测试完成！")
    logger.info("="*70)
    
    if success and len(assigned_events) > 0:
        logger.info("\n✅ 测试成功:")
        logger.info(f"  ✓ {len(assigned_events)} 个订单成功分配")
        logger.info(f"  ✓ {len(delivery_events)} 个订单成功配送")
        logger.info(f"  ✓ ALNS 执行 {dispatcher_stats.get('dispatch_count', 0)} 次调度")
        return True
    else:
        logger.warning("\n⚠️  测试部分成功，存在问题需要调查")
        return False




if __name__ == "__main__":
    success = test_alns_basic()
    sys.exit(0 if success else 1)

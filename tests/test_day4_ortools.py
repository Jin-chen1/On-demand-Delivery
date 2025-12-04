"""
Day 4 测试脚本：OR-Tools VRP Dispatcher 功能验证
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

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
    log_file = log_dir / f"day4_ortools_{timestamp}.log"
    
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


def test_ortools_dispatcher():
    """测试 OR-Tools 调度器"""
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*70)
    logger.info("Day 4: OR-Tools VRP Dispatcher 测试")
    logger.info("="*70)
    
    try:
        # 步骤1: 加载路网数据
        logger.info("\n步骤1: 加载路网数据")
        logger.info("-"*70)
        
        processed_dir = config.get_data_dir("processed")
        network_config = config.get_network_config()
        
        graph, _ = extract_osm_network(network_config, processed_dir)
        logger.info(f"✓ 路网加载成功 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
        
        # 步骤2: 加载距离矩阵
        logger.info("\n步骤2: 加载距离矩阵")
        logger.info("-"*70)
        
        matrix_config = config.get_distance_matrix_config()
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        logger.info(f"✓ 距离矩阵加载成功 - 大小: {dist_matrix.shape}")
        logger.info(f"  采样节点数: {len(mapping['node_list'])}")
        
        # 步骤3: 初始化仿真环境（使用 OR-Tools 调度器）
        logger.info("\n步骤3: 初始化仿真环境")
        logger.info("-"*70)
        
        sim_config = {
            'simulation_duration': 43200,  # 仿真12小时（覆盖所有订单到达时间36000秒）
            'dispatch_interval': 60.0,    # 每60秒调度一次
            'dispatcher_type': 'ortools',  # 🔧 使用 OR-Tools 调度器（在线模式）
            'use_gps_coords': False,  # 使用路网最短路径距离（与Day 7一致）
            'dispatcher_config': {
                'offline_mode': False,  # 使用在线模式（动态调度，符合DVRPTW）
                'time_limit_seconds': 10,   # 求解时间限制10秒
                'soft_time_windows': True,  # 使用软时间窗
                'time_window_slack': 600.0,  # 时间窗松弛10分钟（与Day 7一致）
                'enable_batching': True,  # 启用分批处理（处理大量订单）
                'allow_insertion_to_active': True  # 允许向非空闲骑手插入订单
            }
        }
        
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            config=sim_config
        )
        logger.info("✓ 仿真环境创建成功")
        logger.info(f"  调度器类型: OR-Tools VRP")
        logger.info(f"  调度间隔: {sim_env.dispatch_interval}秒")
        
        # 步骤4: 加载订单数据
        logger.info("\n步骤4: 加载订单数据")
        logger.info("-"*70)
        
        orders_dir = config.get_data_dir("orders")
        
        # 使用均匀网格采样数据（与Day 7测试一致）
        uniform_orders_file = orders_dir / "uniform_grid_100.csv"
        if uniform_orders_file.exists():
            orders_file = uniform_orders_file
            logger.info("使用均匀网格采样订单数据（uniform_grid_100.csv）")
        else:
            orders_file = orders_dir / "orders.csv"
            logger.info("使用默认模拟订单数据")
        
        sim_env.load_orders_from_csv(orders_file)
        logger.info(f"✓ 订单加载成功 - 数量: {len(sim_env.orders)}")
        
        # 调整订单到达时间到仿真范围内
        # 订单生成脚本使用8:00-22:00时间（28800s-79200s），需要缩放到仿真时间范围
        arrival_times = [order.arrival_time for order in sim_env.orders.values()]
        min_arrival = min(arrival_times)
        max_arrival = max(arrival_times)
        simulation_duration = sim_config['simulation_duration']
        
        if max_arrival > simulation_duration * 0.7:
            # 将到达时间线性缩放到 [0, simulation_duration * 0.7]
            target_max = simulation_duration * 0.7
            for order in sim_env.orders.values():
                # 线性映射: new_time = (old_time - min) / (max - min) * target_max
                if max_arrival > min_arrival:
                    order.arrival_time = (order.arrival_time - min_arrival) / (max_arrival - min_arrival) * target_max
                else:
                    order.arrival_time = 0
            logger.info(f"  订单到达时间已调整: [{min_arrival:.0f}s-{max_arrival:.0f}s] -> [0s-{target_max:.0f}s]")
        
        # 检查订单节点覆盖率
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
        logger.info(f"    商家节点: {merchant_nodes_in_mapping}/{len(sim_env.orders)} ({merchant_nodes_in_mapping/len(sim_env.orders)*100:.1f}%)")
        logger.info(f"    客户节点: {customer_nodes_in_mapping}/{len(sim_env.orders)} ({customer_nodes_in_mapping/len(sim_env.orders)*100:.1f}%)")
        
        # 步骤5: 初始化骑手
        logger.info("\n步骤5: 初始化骑手")
        logger.info("-"*70)
        
        courier_config = config.get_courier_config()
        num_couriers = 20  # 20个骑手
        
        sim_env.initialize_couriers(num_couriers, courier_config)
        logger.info(f"✓ 骑手初始化成功 - 数量: {len(sim_env.couriers)}")
        
        # 步骤6: 运行仿真
        logger.info("\n步骤6: 运行仿真")
        logger.info("-"*70)
        logger.info("开始仿真...")
        
        sim_env.run(until=43200)  # 运行12小时（覆盖所有订单到达时间）
        
        logger.info("✓ 仿真运行完成")
        
        # 步骤7: 详细分析结果
        logger.info("\n步骤7: 结果分析")
        logger.info("-"*70)
        
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
        if len(arrival_events) > 0:
            logger.info(f"  完成率: {len(sim_env.completed_orders)/len(arrival_events)*100:.1f}%")
        
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
            
            # 骑手利用率
            total_utilization = sum(c.get_utilization() for c in sim_env.couriers.values())
            avg_utilization = total_utilization / len(sim_env.couriers)
            logger.info(f"  平均骑手利用率: {avg_utilization*100:.1f}%")
        
        # OR-Tools 调度器统计
        dispatcher_stats = sim_env.dispatcher.get_statistics()
        logger.info(f"\nOR-Tools 调度器统计:")
        logger.info(f"  调度次数: {dispatcher_stats['dispatch_count']}")
        logger.info(f"  成功求解: {dispatcher_stats['solve_success_count']}")
        logger.info(f"  求解失败: {dispatcher_stats['solve_failure_count']}")
        logger.info(f"  平均求解时间: {dispatcher_stats['average_solve_time']:.2f}秒")
        
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
        
        if dispatcher_stats['solve_success_count'] == 0:
            logger.error("❌ 失败：OR-Tools 从未成功求解")
            success = False
        else:
            logger.info(f"✓ OR-Tools 求解功能正常 (成功{dispatcher_stats['solve_success_count']}次)")
        
        # 步骤9: 保存结果
        logger.info("\n步骤9: 保存仿真结果")
        logger.info("-"*70)
        
        output_dir = project_root / "outputs" / "simulation_results" / f"day4_ortools_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_files = sim_env.save_results(output_dir)
        
        # 保存调度器统计
        dispatcher_stats_file = output_dir / "dispatcher_statistics.json"
        with open(dispatcher_stats_file, 'w', encoding='utf-8') as f:
            json.dump(dispatcher_stats, f, indent=2, ensure_ascii=False)
        saved_files['dispatcher_stats'] = dispatcher_stats_file
        
        # 保存详细性能指标（与RL测试一致）
        total_orders = len(arrival_events) if arrival_events else len(sim_env.orders)
        timeout_count = 0
        service_times = []
        if delivery_events:
            completed_order_ids = [e.entity_id for e in delivery_events]
            for oid in completed_order_ids:
                order = sim_env.orders.get(oid)
                if order:
                    if order.is_timeout(sim_env.env.now):
                        timeout_count += 1
                    if order.delivery_complete_time is not None:
                        service_time = order.delivery_complete_time - order.arrival_time
                        service_times.append(service_time)
        
        avg_service_time = sum(service_times) / len(service_times) if service_times else 0
        timeout_rate = timeout_count / len(delivery_events) if delivery_events else 0
        total_utilization = sum(c.get_utilization() for c in sim_env.couriers.values())
        avg_utilization = total_utilization / len(sim_env.couriers) if sim_env.couriers else 0
        
        performance_info = {
            'total_orders': total_orders,
            'completed_orders': len(sim_env.completed_orders),
            'pending_orders': len(sim_env.pending_orders),
            'timeout_orders': timeout_count,
            'completion_rate': len(sim_env.completed_orders) / total_orders if total_orders > 0 else 0,
            'timeout_rate': timeout_rate,
            'avg_service_time': avg_service_time,
            'avg_service_time_minutes': avg_service_time / 60,
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
        logger.info("Day 4 测试完成！")
        logger.info("="*70)
        
        if success and len(assigned_events) > 0 and dispatcher_stats['solve_success_count'] > 0:
            logger.info("\n✅ 测试成功:")
            logger.info(f"  ✓ {len(assigned_events)} 个订单成功分配")
            logger.info(f"  ✓ {len(delivery_events)} 个订单成功配送")
            logger.info(f"  ✓ OR-Tools 成功求解 {dispatcher_stats['solve_success_count']} 次")
            logger.info(f"  ✓ 平均求解时间: {dispatcher_stats['average_solve_time']:.2f}秒")
            return True
        else:
            logger.warning("\n⚠️  测试部分成功，存在问题需要调查")
            return False
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {str(e)}")
        logger.exception("详细错误信息:")
        return False


if __name__ == "__main__":
    success = test_ortools_dispatcher()
    sys.exit(0 if success else 1)

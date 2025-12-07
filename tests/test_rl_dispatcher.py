"""
测试 RL 调度器 - Day 13
使用训练好的PPO模型作为调度器运行仿真
类似于 run_day4_test.py (OR-Tools) 和 test_alns_dispatcher.py (ALNS)
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
from src.simulation import SimulationEnvironment
from src.simulation.dispatchers.rl_dispatcher import RLDispatcher
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.utils.config import get_config


def setup_logging(log_dir: Path):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"day13_rl_{timestamp}.log"
    
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


def find_latest_rl_model(models_dir: Path) -> Path:
    """
    查找最新的RL模型
    
    Args:
        models_dir: 模型目录
        
    Returns:
        最新模型的路径
    """
    # 查找所有模型目录
    model_dirs = []
    for d in models_dir.iterdir():
        if d.is_dir() and d.name.startswith('20'):
            # 检查是否有final_curriculum_model或final_model
            if (d / 'final_curriculum_model.zip').exists():
                model_dirs.append((d, d / 'final_curriculum_model'))
            elif (d / 'final_model.zip').exists():
                model_dirs.append((d, d / 'final_model'))
    
    if not model_dirs:
        raise FileNotFoundError(f"在 {models_dir} 中未找到训练好的RL模型")
    
    # 按目录名排序（时间戳），取最新的
    model_dirs.sort(key=lambda x: x[0].name, reverse=True)
    return model_dirs[0][1]


def test_rl_dispatcher(model_path: str = None):
    """
    测试RL调度器
    
    Args:
        model_path: 可选，指定模型路径。如果不指定，自动查找最新模型
    """
    # 加载配置
    config = get_config()
    
    # 设置日志
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("\n" + "="*70)
    logger.info("测试 RL 调度器 - Day 13")
    logger.info("="*70)
    
    # 1. 查找或验证模型路径
    logger.info("\n步骤1: 加载RL模型")
    logger.info("-"*70)
    
    if model_path:
        rl_model_path = Path(model_path)
        if not rl_model_path.exists() and not Path(str(rl_model_path) + '.zip').exists():
            raise FileNotFoundError(f"指定的模型路径不存在: {model_path}")
    else:
        # 自动查找最新模型
        models_dir = project_root / "outputs" / "rl_training" / "models"
        rl_model_path = find_latest_rl_model(models_dir)
    
    logger.info(f"✓ 使用模型: {rl_model_path}")
    
    # 2. 加载路网数据
    logger.info("\n步骤2: 加载路网数据")
    logger.info("-"*70)
    
    processed_dir = config.get_data_dir("processed")
    network_config = config.get_network_config()
    
    graph, _ = extract_osm_network(network_config, processed_dir)
    logger.info(f"✓ 路网加载成功 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
    
    # 3. 加载距离矩阵
    logger.info("\n步骤3: 加载距离矩阵")
    logger.info("-"*70)
    
    matrix_config = config.get_distance_matrix_config()
    dist_matrix, time_matrix, mapping = compute_distance_matrices(
        graph, matrix_config, processed_dir
    )
    logger.info(f"✓ 距离矩阵加载成功 - 大小: {dist_matrix.shape}")
    
    # 4. 创建仿真环境
    logger.info("\n步骤4: 初始化仿真环境")
    logger.info("-"*70)
    
    sim_config = {
        'simulation_duration': 43200,  # 仿真12小时（与OR-Tools/ALNS测试一致）
        'dispatch_interval': 60.0,
        'use_gps_coords': False,  # 使用路网最短路径距离
        'dispatcher_type': 'greedy',  # 初始使用greedy，后面替换为RL
        'enable_auto_dispatch': True,  # 启用自动调度
    }
    
    env = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=sim_config
    )
    logger.info("✓ 仿真环境创建成功")
    
    # 5. 加载订单数据
    logger.info("\n步骤5: 加载订单数据")
    logger.info("-"*70)
    
    orders_dir = config.get_data_dir("orders")
    
    # 使用均匀网格采样数据（与OR-Tools/ALNS测试一致）
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
    
    # 调整订单到达时间到仿真范围内（与OR-Tools/ALNS测试一致）
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
    logger.info("\n步骤6: 初始化骑手")
    logger.info("-"*70)
    
    courier_config = config.get_courier_config()
    num_couriers = 20  # 与OR-Tools/ALNS测试一致
    
    env.initialize_couriers(num_couriers, courier_config)
    logger.info(f"✓ 骑手初始化成功 - 数量: {len(env.couriers)}")
    
    # 7. 创建RL调度器并替换默认调度器
    logger.info("\n步骤7: 创建RL调度器")
    logger.info("-"*70)
    
    # 注意：max_couriers必须与训练时的配置一致（50），而不是当前仿真的骑手数
    # 这确保状态编码维度与模型期望的维度匹配
    rl_config = {
        'max_pending_orders': 50,
        'max_couriers': 50,  # 必须与训练配置一致
        'use_action_masking': True  # 启用动作屏蔽，这对MaskablePPO模型是必须的
    }
    
    rl_dispatcher = RLDispatcher(env, model_path=str(rl_model_path), config=rl_config)
    env.dispatcher = rl_dispatcher
    
    logger.info(f"✓ RL调度器创建成功")
    logger.info(f"  模型路径: {rl_model_path}")
    logger.info(f"  max_pending_orders: {rl_config['max_pending_orders']}")
    logger.info(f"  max_couriers: {rl_config['max_couriers']}")
    
    # 8. 运行仿真
    logger.info("\n步骤8: 运行仿真")
    logger.info("-"*70)
    logger.info("开始仿真...\n")
    
    env.run(until=43200)  # 运行12小时
    
    logger.info("\n✓ 仿真完成")
    
    # 9. 详细结果分析
    logger.info("\n步骤9: 结果分析")
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
    
    total_orders = len(arrival_events) if arrival_events else len(env.orders)
    completion_rate = len(env.completed_orders) / total_orders if total_orders > 0 else 0
    logger.info(f"  完成率: {completion_rate*100:.1f}%")
    
    # 计算性能指标
    timeout_count = 0
    service_times = []
    
    if delivery_events:
        completed_order_ids = [e.entity_id for e in delivery_events]
        for oid in completed_order_ids:
            order = env.orders.get(oid)
            if order:
                if order.is_timeout(env.env.now):
                    timeout_count += 1
                if order.delivery_complete_time is not None:
                    service_time = order.delivery_complete_time - order.arrival_time
                    service_times.append(service_time)
        
        logger.info(f"\n性能指标:")
        logger.info(f"  超时订单: {timeout_count}/{len(delivery_events)}")
        timeout_rate = timeout_count / len(delivery_events) if len(delivery_events) > 0 else 0
        logger.info(f"  超时率: {timeout_rate*100:.1f}%")
        
        if service_times:
            avg_service_time = np.mean(service_times)
            logger.info(f"  平均服务时长: {avg_service_time:.1f}秒 ({avg_service_time/60:.1f}分钟)")
        
        # 骑手利用率
        total_utilization = sum(c.get_utilization() for c in env.couriers.values())
        avg_utilization = total_utilization / len(env.couriers)
        logger.info(f"  平均骑手利用率: {avg_utilization*100:.1f}%")
    
    # RL调度器统计
    dispatcher_stats = env.dispatcher.get_statistics()
    logger.info(f"\nRL 调度器统计:")
    logger.info(f"  调度次数: {dispatcher_stats.get('dispatch_count', 0)}")
    logger.info(f"  总分配订单: {dispatcher_stats.get('total_assigned', 0)}")
    logger.info(f"  平均每次调度时间: {dispatcher_stats.get('avg_dispatch_time', 0):.3f}秒")
    
    # 验证检查
    logger.info("\n步骤10: 验证检查")
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
        logger.error("❌ 失败：RL调度器从未执行调度")
        success = False
    else:
        logger.info(f"✓ RL调度功能正常 (执行{dispatcher_stats.get('dispatch_count', 0)}次)")
    
    # 步骤11: 保存结果
    logger.info("\n步骤11: 保存仿真结果")
    logger.info("-"*70)
    
    output_dir = project_root / "outputs" / "simulation_results" / f"day13_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    saved_files = env.save_results(output_dir)
    
    # 保存调度器统计
    dispatcher_stats_file = output_dir / "dispatcher_statistics.json"
    with open(dispatcher_stats_file, 'w', encoding='utf-8') as f:
        json.dump(dispatcher_stats, f, indent=2, ensure_ascii=False)
    saved_files['dispatcher_stats'] = dispatcher_stats_file
    
    # 保存模型信息
    model_info = {
        'model_path': str(rl_model_path),
        'rl_config': rl_config,
        'completion_rate': completion_rate,
        'timeout_rate': timeout_rate if delivery_events else 0,
        'avg_service_time': float(np.mean(service_times)) if service_times else 0,
        'total_orders': total_orders,
        'completed_orders': len(env.completed_orders),
        'timeout_orders': timeout_count
    }
    model_info_file = output_dir / "model_info.json"
    with open(model_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    saved_files['model_info'] = model_info_file
    
    logger.info("保存的文件:")
    for key, path in saved_files.items():
        logger.info(f"  {key}: {path}")
    
    # 总结
    logger.info("\n" + "="*70)
    logger.info("Day 13 RL调度器测试完成！")
    logger.info("="*70)
    
    if success and len(assigned_events) > 0:
        logger.info("\n✅ 测试成功:")
        logger.info(f"  ✓ {len(assigned_events)} 个订单成功分配")
        logger.info(f"  ✓ {len(delivery_events)} 个订单成功配送")
        logger.info(f"  ✓ 完成率: {completion_rate*100:.1f}%")
        logger.info(f"  ✓ 超时率: {timeout_rate*100:.1f}%" if delivery_events else "  ✓ 超时率: N/A")
        logger.info(f"  ✓ RL调度器执行 {dispatcher_stats.get('dispatch_count', 0)} 次调度")
        return True
    else:
        logger.warning("\n⚠️  测试部分成功，存在问题需要调查")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试RL调度器')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='指定模型路径（不指定则自动查找最新模型）'
    )
    
    args = parser.parse_args()
    
    success = test_rl_dispatcher(model_path=args.model)
    sys.exit(0 if success else 1)

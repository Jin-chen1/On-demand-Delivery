"""
Day 7 测试脚本：OR-Tools vs ALNS 求解速度对比
重点：单次求解速度、算法性能对比
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import pandas as pd
import numpy as np

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
    log_file = log_dir / f"day7_comparison_{timestamp}.log"
    
    # 清除已有的handlers
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


def run_simulation(dispatcher_type: str, dispatcher_config: dict, 
                   graph, dist_matrix, time_matrix, mapping,
                   orders_file: Path, num_couriers: int, 
                   courier_config: dict, logger,
                   use_gps_coords: bool = False):
    """
    运行单次仿真实验
    
    Args:
        dispatcher_type: 'ortools' 或 'alns'
        dispatcher_config: 调度器配置
        graph: 路网图
        dist_matrix: 距离矩阵
        time_matrix: 时间矩阵
        mapping: 节点映射
        orders_file: 订单文件路径
        num_couriers: 骑手数量
        courier_config: 骑手配置
        logger: 日志记录器
        use_gps_coords: 是否使用GPS坐标模式
    
    Returns:
        统计结果字典
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"运行 {dispatcher_type.upper()} 调度器仿真 (GPS模式: {use_gps_coords})")
    logger.info(f"{'='*70}")
    
    try:
        # 配置仿真参数
        sim_config = {
            'simulation_duration': 14400,  # 4小时（GPS数据距离较大需要更长时间）
            'dispatch_interval': 60.0,    # 60秒调度间隔
            'dispatcher_type': dispatcher_type,
            'dispatcher_config': dispatcher_config,
            'use_gps_coords': use_gps_coords  # GPS坐标模式
        }
        
        # 创建仿真环境
        logger.info("\n步骤1: 创建仿真环境")
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            config=sim_config
        )
        logger.info(f"✓ 仿真环境创建成功 - 调度器: {dispatcher_type.upper()}")
        
        # 加载订单
        logger.info("\n步骤2: 加载订单数据")
        sim_env.load_orders_from_csv(orders_file)
        logger.info(f"✓ 加载 {len(sim_env.orders)} 个订单")
        
        # 初始化骑手
        logger.info("\n步骤3: 初始化骑手")
        sim_env.initialize_couriers(num_couriers, courier_config)
        logger.info(f"✓ 初始化 {num_couriers} 个骑手")
        
        # 运行仿真
        logger.info("\n步骤4: 运行仿真")
        sim_env.run()  # 使用配置的simulation_duration
        logger.info("✓ 仿真完成")
        
        # 收集统计数据
        logger.info("\n步骤5: 收集统计数据")
        stats = collect_statistics(sim_env, dispatcher_type, logger)
        
        return stats
        
    except Exception as e:
        logger.error(f"仿真运行失败: {str(e)}")
        logger.exception("详细错误:")
        return None


def collect_statistics(sim_env, dispatcher_type: str, logger):
    """收集详细统计数据"""
    
    # 基础统计
    arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
    assigned_events = [e for e in sim_env.events if e.event_type == 'order_assigned']
    delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
    
    results = {
        'dispatcher_type': dispatcher_type,
        'total_orders': len(arrival_events),
        'assigned_orders': len(assigned_events),
        'completed_orders': len(delivery_events),
        'pending_orders': len(sim_env.pending_orders),
        'completion_rate': len(delivery_events) / len(arrival_events) if arrival_events else 0
    }
    
    # 订单性能指标
    if delivery_events:
        completed_order_ids = [e.entity_id for e in delivery_events]
        
        # 超时率
        timeout_count = sum(
            1 for oid in completed_order_ids
            if sim_env.orders[oid].is_timeout(sim_env.env.now)
        )
        results['timeout_count'] = timeout_count
        results['timeout_rate'] = timeout_count / len(delivery_events)
        
        # 服务时长
        service_times = [
            sim_env.orders[oid].get_total_service_time()
            for oid in completed_order_ids
            if sim_env.orders[oid].get_total_service_time() is not None
        ]
        
        if service_times:
            results['avg_service_time'] = np.mean(service_times)
            results['median_service_time'] = np.median(service_times)
            results['std_service_time'] = np.std(service_times)
        else:
            results['avg_service_time'] = 0
            results['median_service_time'] = 0
            results['std_service_time'] = 0
        
        # 骑手统计
        total_distance = sum(c.total_distance for c in sim_env.couriers.values())
        results['total_distance'] = total_distance
        results['avg_distance_per_order'] = total_distance / len(delivery_events)
        
        total_utilization = sum(c.get_utilization() for c in sim_env.couriers.values())
        results['avg_courier_utilization'] = total_utilization / len(sim_env.couriers)
    else:
        results.update({
            'timeout_count': 0,
            'timeout_rate': 0,
            'avg_service_time': 0,
            'median_service_time': 0,
            'std_service_time': 0,
            'total_distance': 0,
            'avg_distance_per_order': 0,
            'avg_courier_utilization': 0
        })
    
    # 调度器特定统计（重点：求解时间）
    if hasattr(sim_env.dispatcher, 'get_statistics'):
        dispatcher_stats = sim_env.dispatcher.get_statistics()
        results['dispatcher_stats'] = dispatcher_stats
        
        # 提取关键求解指标
        results['dispatch_count'] = dispatcher_stats.get('dispatch_count', 0)
        results['solve_success_count'] = dispatcher_stats.get('solve_success_count', 0)
        results['solve_failure_count'] = dispatcher_stats.get('solve_failure_count', 0)
        results['avg_solve_time'] = dispatcher_stats.get('average_solve_time', 0)
        results['solve_success_rate'] = dispatcher_stats.get('solve_success_rate', 0)
        
        logger.info(f"\n调度器统计:")
        logger.info(f"  调度次数: {results['dispatch_count']}")
        logger.info(f"  成功次数: {results['solve_success_count']}")
        logger.info(f"  失败次数: {results['solve_failure_count']}")
        logger.info(f"  平均求解时间: {results['avg_solve_time']:.3f}秒")
        logger.info(f"  求解成功率: {results['solve_success_rate']*100:.1f}%")
    
    return results


def compare_results(ortools_results, alns_results, logger):
    """对比两种调度器的结果"""
    logger.info("\n" + "="*70)
    logger.info("Day 7: OR-Tools vs ALNS 性能对比")
    logger.info("="*70)
    
    # 创建对比表格
    comparison = {
        '指标': [],
        'OR-Tools': [],
        'ALNS': [],
        '差异': [],
        '改进率': []
    }
    
    # 定义对比指标（lower_is_better表示越低越好）
    metrics = [
        ('完成订单数', 'completed_orders', False, '.0f'),
        ('完成率', 'completion_rate', False, '.1%'),
        ('超时订单数', 'timeout_count', True, '.0f'),
        ('超时率', 'timeout_rate', True, '.1%'),
        ('平均服务时长(秒)', 'avg_service_time', True, '.1f'),
        ('中位服务时长(秒)', 'median_service_time', True, '.1f'),
        ('总行驶距离(米)', 'total_distance', True, '.0f'),
        ('单均行驶距离(米)', 'avg_distance_per_order', True, '.0f'),
        ('骑手平均利用率', 'avg_courier_utilization', False, '.1%'),
        # 求解性能指标（Day 7重点）
        ('平均求解时间(秒)', 'avg_solve_time', True, '.3f'),
        ('求解成功率', 'solve_success_rate', False, '.1%'),
        ('调度次数', 'dispatch_count', False, '.0f')
    ]
    
    for metric_name, key, lower_is_better, fmt in metrics:
        ortools_val = ortools_results.get(key, 0)
        alns_val = alns_results.get(key, 0)
        
        # 格式化数值
        if '%' in fmt:
            diff = (alns_val - ortools_val) * 100
            improvement = ((ortools_val - alns_val) / ortools_val * 100) if ortools_val != 0 else 0
            if not lower_is_better:
                improvement = -improvement
            
            ortools_str = f"{ortools_val*100:.1f}%"
            alns_str = f"{alns_val*100:.1f}%"
            diff_str = f"{diff:+.1f}pp"
        else:
            diff = alns_val - ortools_val
            improvement = ((ortools_val - alns_val) / ortools_val * 100) if ortools_val != 0 else 0
            if not lower_is_better:
                improvement = -improvement
            
            try:
                ortools_str = f"{ortools_val:{fmt}}"
                alns_str = f"{alns_val:{fmt}}"
                diff_str = f"{diff:+.2f}"
            except ValueError:
                ortools_str = str(ortools_val)
                alns_str = str(alns_val)
                diff_str = str(diff)
        
        comparison['指标'].append(metric_name)
        comparison['OR-Tools'].append(ortools_str)
        comparison['ALNS'].append(alns_str)
        comparison['差异'].append(diff_str)
        comparison['改进率'].append(f"{improvement:+.1f}%")
    
    # 打印对比表格
    df = pd.DataFrame(comparison)
    logger.info("\n" + df.to_string(index=False))
    
    # 重点分析：求解速度对比（Day 7核心）
    logger.info("\n" + "="*70)
    logger.info("Day 7 核心发现：求解速度对比")
    logger.info("="*70)
    
    solve_time_ratio = alns_results['avg_solve_time'] / ortools_results['avg_solve_time'] if ortools_results['avg_solve_time'] > 0 else 0
    
    logger.info(f"\n求解速度分析:")
    logger.info(f"  OR-Tools 平均求解时间: {ortools_results['avg_solve_time']:.3f}秒")
    logger.info(f"  ALNS 平均求解时间: {alns_results['avg_solve_time']:.3f}秒")
    logger.info(f"  速度比 (ALNS/OR-Tools): {solve_time_ratio:.2f}x")
    
    if solve_time_ratio < 1:
        speedup = (1 - solve_time_ratio) * 100
        logger.info(f"  ✓ ALNS 比 OR-Tools 快 {speedup:.1f}%")
    else:
        slowdown = (solve_time_ratio - 1) * 100
        logger.info(f"  ✗ ALNS 比 OR-Tools 慢 {slowdown:.1f}%")
    
    # 性能质量对比
    logger.info(f"\n性能质量对比:")
    logger.info(f"  超时率: OR-Tools {ortools_results['timeout_rate']*100:.1f}% vs ALNS {alns_results['timeout_rate']*100:.1f}%")
    logger.info(f"  平均服务时长: OR-Tools {ortools_results['avg_service_time']:.1f}s vs ALNS {alns_results['avg_service_time']:.1f}s")
    logger.info(f"  单均距离: OR-Tools {ortools_results['avg_distance_per_order']:.0f}m vs ALNS {alns_results['avg_distance_per_order']:.0f}m")
    
    # 综合评价
    logger.info(f"\n综合评价:")
    if alns_results['avg_solve_time'] < ortools_results['avg_solve_time'] and alns_results['timeout_rate'] <= ortools_results['timeout_rate']:
        logger.info("  ✓ ALNS在保证服务质量的同时，求解速度更快，适合实时调度场景")
    elif alns_results['avg_solve_time'] > ortools_results['avg_solve_time'] and alns_results['timeout_rate'] < ortools_results['timeout_rate']:
        logger.info("  ✓ ALNS虽然求解速度较慢，但服务质量更好，适合高质量要求场景")
    elif alns_results['avg_solve_time'] < ortools_results['avg_solve_time']:
        logger.info("  ~ ALNS求解更快，但需要在服务质量上进一步优化")
    else:
        logger.info("  ~ OR-Tools在当前场景下表现更优")
    
    return df


def main():
    """主函数"""
    config = get_config()
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*70)
    logger.info("Day 7: OR-Tools vs ALNS 求解速度对比测试")
    logger.info("="*70)
    
    # 步骤1: 加载共享数据（路网、距离矩阵）
    logger.info("\n步骤1: 加载路网和距离矩阵")
    logger.info("-"*70)
    
    processed_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    # 优先使用均匀网格采样数据（Solomon Random分布，覆盖全路网）
    lade_shanghai_100 = orders_dir / "uniform_grid_100.csv"
    lade_shanghai_500 = orders_dir / "uniform_grid_500.csv"
    shanghai_dir = processed_dir / "shanghai"
    
    if lade_shanghai_100.exists() and shanghai_dir.exists():
        # 使用均匀网格数据 + 上海路网
        logger.info("使用均匀网格采样数据（Solomon Random分布，覆盖全路网）")
        
        network_config = config.get_network_config()
        graph, _ = extract_osm_network(network_config, processed_dir)
        logger.info(f"✓ 上海路网加载成功 - 节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
        
        matrix_config = config.get_distance_matrix_config()
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        logger.info(f"✓ 距离矩阵加载成功 - 大小: {dist_matrix.shape}")
        
        orders_file = lade_shanghai_100
        use_gps_coords = False  # 使用路网最短路径距离
        logger.info(f"使用LaDe上海订单文件: {orders_file} (100个真实订单)")
    else:
        # 必须使用LaDe上海数据
        raise FileNotFoundError(
            "未找到LaDe上海订单文件，请先运行: python scripts/filter_lade_shanghai.py"
        )
    
    courier_config = config.get_courier_config()
    num_couriers = 20
    
    # 步骤3: 配置两种调度器
    # OR-Tools配置：使用在线模式（真正的DVRPTW动态调度）
    ortools_config = {
        'offline_mode': False,  # 使用在线模式（动态调度，符合DVRPTW）
        'time_limit_seconds': 10,  # 每次调度求解时间限制10秒
        'soft_time_windows': True,
        'time_window_slack': 600.0,  # 10分钟松弛
        'enable_batching': True,  # 启用分批处理（处理大量订单）
        'allow_insertion_to_active': True  # 允许向非空闲骑手插入订单
    }
    
    # ALNS配置：增加迭代次数以匹配OR-Tools的求解质量
    alns_config = {
        'iterations': 200,  # 增加到200次迭代
        'destroy_degree_min': 0.1,
        'destroy_degree_max': 0.3,
        'temperature_start': 5000.0,
        'temperature_end': 1.0,
        'temperature_decay': 0.95,
        'random_seed': 42
    }
    
    # 步骤4: 运行 OR-Tools 调度器
    logger.info("\n开始测试 OR-Tools 调度器...")
    logger.info("-"*70)
    ortools_results = run_simulation(
        'ortools', ortools_config,
        graph, dist_matrix, time_matrix, mapping,
        orders_file, num_couriers, courier_config, logger,
        use_gps_coords=use_gps_coords
    )
    
    if ortools_results is None:
        logger.error("OR-Tools 调度器测试失败")
        return False
    
    # 步骤5: 运行 ALNS 调度器
    logger.info("\n开始测试 ALNS 调度器...")
    logger.info("-"*70)
    alns_results = run_simulation(
        'alns', alns_config,
        graph, dist_matrix, time_matrix, mapping,
        orders_file, num_couriers, courier_config, logger,
        use_gps_coords=use_gps_coords
    )
    
    if alns_results is None:
        logger.error("ALNS 调度器测试失败")
        return False
    
    # 步骤6: 对比结果
    comparison_df = compare_results(ortools_results, alns_results, logger)
    
    # 步骤7: 保存结果
    output_dir = project_root / "outputs" / "day7_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存对比表格
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False, encoding='utf-8-sig')
    
    # 保存详细结果
    results = {
        'ortools': ortools_results,
        'alns': alns_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ 对比结果已保存到: {output_dir}")
    logger.info("\n" + "="*70)
    logger.info("Day 7 测试完成！")
    logger.info("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
        logging.exception("详细错误:")
        sys.exit(1)

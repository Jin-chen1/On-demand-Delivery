"""
调度器对比测试脚本
对比 Greedy 和 OR-Tools 两种调度器的性能
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.simulation import SimulationEnvironment


def setup_logging(log_dir: Path, test_name: str):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"compare_{test_name}_{timestamp}.log"
    
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


def run_simulation(dispatcher_type: str, config: dict, logger):
    """
    运行单次仿真
    
    Args:
        dispatcher_type: 'greedy' 或 'ortools'
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        统计结果字典
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"运行 {dispatcher_type.upper()} 调度器仿真")
    logger.info(f"{'='*70}")
    
    try:
        # 加载路网数据
        logger.info("\n步骤1: 加载路网和距离矩阵")
        processed_dir = config.get_data_dir("processed")
        network_config = config.get_network_config()
        
        graph, _ = extract_osm_network(network_config, processed_dir)
        
        matrix_config = config.get_distance_matrix_config()
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph, matrix_config, processed_dir
        )
        
        logger.info(f"✓ 数据加载完成")
        
        # 配置仿真参数
        sim_config = {
            'simulation_duration': 3600,  # 1小时
            'dispatch_interval': 60.0,    # 60秒调度间隔
            'dispatcher_type': dispatcher_type
        }
        
        if dispatcher_type == 'ortools':
            sim_config['dispatcher_config'] = {
                'time_limit_seconds': 5,
                'soft_time_windows': True,
                'time_window_slack': 300.0
            }
        
        # 创建仿真环境
        logger.info("\n步骤2: 创建仿真环境")
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=dist_matrix,
            time_matrix=time_matrix,
            node_mapping=mapping,
            config=sim_config
        )
        
        # 加载订单
        logger.info("\n步骤3: 加载订单数据")
        orders_dir = config.get_data_dir("orders")
        orders_file = orders_dir / "orders.csv"
        sim_env.load_orders_from_csv(orders_file)
        logger.info(f"✓ 加载 {len(sim_env.orders)} 个订单")
        
        # 初始化骑手
        logger.info("\n步骤4: 初始化骑手")
        courier_config = config.get_courier_config()
        num_couriers = 15
        sim_env.initialize_couriers(num_couriers, courier_config)
        logger.info(f"✓ 初始化 {num_couriers} 个骑手")
        
        # 运行仿真
        logger.info("\n步骤5: 运行仿真")
        sim_env.run(until=3600)
        logger.info("✓ 仿真完成")
        
        # 收集统计数据
        logger.info("\n步骤6: 收集统计数据")
        stats = sim_env.get_statistics()
        
        # 事件统计
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        assigned_events = [e for e in sim_env.events if e.event_type == 'order_assigned']
        delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
        
        # 计算性能指标
        results = {
            'dispatcher_type': dispatcher_type,
            'total_orders': len(arrival_events),
            'assigned_orders': len(assigned_events),
            'completed_orders': len(delivery_events),
            'completion_rate': len(delivery_events) / len(arrival_events) if arrival_events else 0,
            'pending_orders': len(sim_env.pending_orders)
        }
        
        if delivery_events:
            completed_order_ids = [e.entity_id for e in delivery_events]
            
            # 超时率
            timeout_count = sum(
                1 for oid in completed_order_ids
                if sim_env.orders[oid].is_timeout(sim_env.env.now)
            )
            results['timeout_count'] = timeout_count
            results['timeout_rate'] = timeout_count / len(delivery_events)
            
            # 平均服务时长
            service_times = [
                sim_env.orders[oid].get_total_service_time()
                for oid in completed_order_ids
                if sim_env.orders[oid].get_total_service_time() is not None
            ]
            results['avg_service_time'] = sum(service_times) / len(service_times) if service_times else 0
            
            # 骑手统计
            total_distance = sum(c.total_distance for c in sim_env.couriers.values())
            results['total_distance'] = total_distance
            results['avg_distance_per_order'] = total_distance / len(delivery_events) if delivery_events else 0
            
            total_utilization = sum(c.get_utilization() for c in sim_env.couriers.values())
            results['avg_courier_utilization'] = total_utilization / len(sim_env.couriers)
        else:
            results['timeout_count'] = 0
            results['timeout_rate'] = 0
            results['avg_service_time'] = 0
            results['total_distance'] = 0
            results['avg_distance_per_order'] = 0
            results['avg_courier_utilization'] = 0
        
        # 调度器特定统计
        if hasattr(sim_env.dispatcher, 'get_statistics'):
            dispatcher_stats = sim_env.dispatcher.get_statistics()
            results['dispatcher_stats'] = dispatcher_stats
            
            if dispatcher_type == 'ortools':
                results['avg_solve_time'] = dispatcher_stats.get('average_solve_time', 0)
                results['solve_success_rate'] = (
                    dispatcher_stats['solve_success_count'] / 
                    (dispatcher_stats['solve_success_count'] + dispatcher_stats['solve_failure_count'])
                    if (dispatcher_stats['solve_success_count'] + dispatcher_stats['solve_failure_count']) > 0
                    else 0
                )
        
        logger.info("✓ 统计数据收集完成")
        
        return results
        
    except Exception as e:
        logger.error(f"仿真运行失败: {str(e)}")
        logger.exception("详细错误:")
        return None


def compare_results(greedy_results, ortools_results, logger):
    """对比两种调度器的结果"""
    logger.info("\n" + "="*70)
    logger.info("调度器性能对比")
    logger.info("="*70)
    
    # 创建对比表格
    comparison = {
        '指标': [],
        'Greedy': [],
        'OR-Tools': [],
        '差异': [],
        '改进率': []
    }
    
    metrics = [
        ('完成订单数', 'completed_orders', False),
        ('完成率', 'completion_rate', False, '.1%'),
        ('超时订单数', 'timeout_count', True),
        ('超时率', 'timeout_rate', True, '.1%'),
        ('平均服务时长(秒)', 'avg_service_time', True, '.1f'),
        ('总行驶距离(米)', 'total_distance', True, '.0f'),
        ('单均行驶距离(米)', 'avg_distance_per_order', True, '.0f'),
        ('骑手平均利用率', 'avg_courier_utilization', False, '.1%')
    ]
    
    for metric_name, key, lower_is_better, *fmt in metrics:
        greedy_val = greedy_results.get(key, 0)
        ortools_val = ortools_results.get(key, 0)
        
        if len(fmt) > 0:
            format_str = fmt[0]
            if '%' in format_str:
                diff = (ortools_val - greedy_val) * 100
                improvement = ((greedy_val - ortools_val) / greedy_val * 100) if greedy_val > 0 else 0
                if lower_is_better:
                    improvement = -improvement
                
                greedy_str = f"{greedy_val*100:{format_str.replace('%', '')}}"
                ortools_str = f"{ortools_val*100:{format_str.replace('%', '')}}"
                diff_str = f"{diff:+.1f}pp"
            else:
                diff = ortools_val - greedy_val
                improvement = ((greedy_val - ortools_val) / greedy_val * 100) if greedy_val > 0 else 0
                if lower_is_better:
                    improvement = -improvement
                
                greedy_str = f"{greedy_val:{format_str}}"
                ortools_str = f"{ortools_val:{format_str}}"
                diff_str = f"{diff:+{format_str}}"
        else:
            diff = ortools_val - greedy_val
            improvement = ((greedy_val - ortools_val) / greedy_val * 100) if greedy_val > 0 else 0
            if lower_is_better:
                improvement = -improvement
            
            greedy_str = str(greedy_val)
            ortools_str = str(ortools_val)
            diff_str = f"{diff:+.0f}"
        
        comparison['指标'].append(metric_name)
        comparison['Greedy'].append(greedy_str)
        comparison['OR-Tools'].append(ortools_str)
        comparison['差异'].append(diff_str)
        comparison['改进率'].append(f"{improvement:+.1f}%")
    
    # OR-Tools 特有指标
    if 'avg_solve_time' in ortools_results:
        comparison['指标'].append('平均求解时间(秒)')
        comparison['Greedy'].append('-')
        comparison['OR-Tools'].append(f"{ortools_results['avg_solve_time']:.2f}")
        comparison['差异'].append('-')
        comparison['改进率'].append('-')
    
    # 打印表格
    df = pd.DataFrame(comparison)
    logger.info("\n" + df.to_string(index=False))
    
    # 总结
    logger.info("\n" + "="*70)
    logger.info("对比总结")
    logger.info("="*70)
    
    if ortools_results['timeout_rate'] < greedy_results['timeout_rate']:
        improvement = (greedy_results['timeout_rate'] - ortools_results['timeout_rate']) / greedy_results['timeout_rate'] * 100
        logger.info(f"✓ OR-Tools 超时率降低 {improvement:.1f}%")
    
    if ortools_results['avg_service_time'] < greedy_results['avg_service_time']:
        improvement = (greedy_results['avg_service_time'] - ortools_results['avg_service_time']) / greedy_results['avg_service_time'] * 100
        logger.info(f"✓ OR-Tools 平均服务时长减少 {improvement:.1f}%")
    
    if ortools_results['avg_distance_per_order'] < greedy_results['avg_distance_per_order']:
        improvement = (greedy_results['avg_distance_per_order'] - ortools_results['avg_distance_per_order']) / greedy_results['avg_distance_per_order'] * 100
        logger.info(f"✓ OR-Tools 单均行驶距离减少 {improvement:.1f}%")
    
    return df


def main():
    """主函数"""
    config = get_config()
    log_dir = project_root / "logs"
    logger = setup_logging(log_dir, "all")
    
    logger.info("="*70)
    logger.info("调度器对比测试")
    logger.info("="*70)
    
    # 运行 Greedy 调度器
    logger.info("\n开始测试 Greedy 调度器...")
    greedy_results = run_simulation('greedy', config, logger)
    
    if greedy_results is None:
        logger.error("Greedy 调度器测试失败")
        return False
    
    # 运行 OR-Tools 调度器
    logger.info("\n开始测试 OR-Tools 调度器...")
    ortools_results = run_simulation('ortools', config, logger)
    
    if ortools_results is None:
        logger.error("OR-Tools 调度器测试失败")
        return False
    
    # 对比结果
    comparison_df = compare_results(greedy_results, ortools_results, logger)
    
    # 保存对比结果
    output_dir = project_root / "data" / "simulation_results" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存对比表格
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False, encoding='utf-8-sig')
    
    # 保存详细结果
    results = {
        'greedy': greedy_results,
        'ortools': ortools_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n对比结果已保存到: {output_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

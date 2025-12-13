"""
Day 5 测试：评估指标和可视化系统
演示如何使用分析模块生成论文所需的图表和数据
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.utils.config import ConfigManager
from src.data_preparation import osm_network, distance_matrix
from src.simulation.environment import SimulationEnvironment
from src.analysis import MetricsCalculator, Visualizer, ReportGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_comparison_analysis():
    """测试多方法对比分析（模拟论文 Fig 4）"""
    
    logger.info("\n" + "="*70)
    logger.info("Day 5 测试：多方法对比分析")
    logger.info("="*70)
    
    # 1. 加载基础数据
    config = ConfigManager()
    network_config = config.get('network')
    matrix_config = config.get('distance_matrix')
    
    data_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    logger.info("\n=== 加载路网和距离矩阵 ===")
    graph, _ = osm_network.extract_osm_network(network_config, data_dir, force_download=False)
    dist_matrix, time_matrix, mapping = distance_matrix.compute_distance_matrices(
        graph, matrix_config, data_dir, force_recalculate=False
    )
    
    # 使用均匀网格采样数据（与Day 4一致）
    orders_file = orders_dir / "uniform_grid_100.csv"
    
    # 仿真配置（与Day 4一致）
    simulation_duration = 43200.0  # 12小时
    
    # 辅助函数：调整订单到达时间到仿真范围内
    def adjust_order_times(env, simulation_duration):
        """将订单到达时间线性缩放到仿真范围内"""
        arrival_times = [order.arrival_time for order in env.orders.values()]
        min_arrival = min(arrival_times)
        max_arrival = max(arrival_times)
        
        if max_arrival > simulation_duration * 0.7:
            target_max = simulation_duration * 0.7
            for order in env.orders.values():
                if max_arrival > min_arrival:
                    order.arrival_time = (order.arrival_time - min_arrival) / (max_arrival - min_arrival) * target_max
                else:
                    order.arrival_time = 0
            logger.info(f"  订单到达时间已调整: [{min_arrival:.0f}s-{max_arrival:.0f}s] -> [0s-{target_max:.0f}s]")
    
    # 2. 运行多个方法的仿真
    logger.info("\n=== 运行多方法仿真 ===")
    envs = {}
    
    # 方法1: Greedy
    logger.info("\n--- 运行 Greedy 调度器 ---")
    greedy_config = {
        'simulation_duration': simulation_duration,
        'dispatch_interval': 60.0,
        'dispatcher_type': 'greedy',
        'dispatcher_config': {}
    }
    
    env_greedy = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=greedy_config
    )
    env_greedy.load_orders_from_csv(orders_file)
    adjust_order_times(env_greedy, simulation_duration)
    env_greedy.initialize_couriers(num_couriers=20, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    env_greedy.run(until=simulation_duration)
    envs['Greedy'] = env_greedy
    
    # 方法2: OR-Tools (动态插入)
    logger.info("\n--- 运行 OR-Tools (动态插入模式) ---")
    ortools_dynamic_config = {
        'simulation_duration': simulation_duration,
        'dispatch_interval': 60.0,
        'dispatcher_type': 'ortools',
        'dispatcher_config': {
            'time_limit_seconds': 3,
            'allow_insertion_to_active': True,  # 动态插入
            'enable_batching': False
        }
    }
    
    env_ortools_dyn = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=ortools_dynamic_config
    )
    env_ortools_dyn.load_orders_from_csv(orders_file)
    adjust_order_times(env_ortools_dyn, simulation_duration)
    env_ortools_dyn.initialize_couriers(num_couriers=20, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    env_ortools_dyn.run(until=simulation_duration)
    envs['OR-Tools-Dynamic'] = env_ortools_dyn
    
    # 3. 生成对比报告
    logger.info("\n=== 生成对比报告 ===")
    report_gen = ReportGenerator(output_dir=project_root / "outputs" / "reports")
    
    output_files = report_gen.generate_comparison_report(
        envs=envs,
        graph=graph,
        report_name="method_comparison"
    )
    
    logger.info("\n生成的对比文件:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")
    
    # 4. 输出简单的性能对比摘要
    logger.info("\n=== 性能对比摘要 ===")
    calculator = MetricsCalculator()
    
    for method_name, env in envs.items():
        metrics = calculator.calculate_from_environment(env)
        logger.info(f"\n{method_name}:")
        logger.info(f"  完成订单: {metrics.completed_orders}")
        logger.info(f"  超时率: {metrics.timeout_rate*100:.2f}%")
        logger.info(f"  平均配送时间: {metrics.avg_delivery_time/60:.2f}分钟")
        logger.info(f"  骑手利用率: {metrics.avg_utilization*100:.1f}%")
        logger.info(f"  单位里程配送: {metrics.orders_per_km:.2f}单/公里")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Day 5 多方法对比分析测试完成")
    logger.info("="*70)
    
    return envs, output_files


if __name__ == "__main__":
    try:
        logger.info("\n" + "🚀"*35)
        logger.info("Day 5: Greedy vs OR-Tools 轨迹对比")
        logger.info("🚀"*35 + "\n")
        
        # 运行 Greedy 和 OR-Tools动态 两种方法的对比分析
        envs, comparison_files = test_comparison_analysis()
        
        logger.info("\n" + "🎉"*35)
        logger.info("Day 5 所有测试完成！")
        logger.info("论文图表所需的数据和可视化系统已就绪")
        logger.info("🎉"*35)
        
        logger.info("\n📊 生成的图表可用于:")
        logger.info("  - Fig 1: 订单热力图和时间分布 (order_heatmap.png, temporal_demand.png)")
        logger.info("  - Fig 3: 滚动时域快照 (courier_routes.png)")
        logger.info("  - Fig 4: 压力测试曲线 (performance_comparison.png)")
        logger.info("  - Fig 5: 轨迹对比案例 (routes_*.png)")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")
        logger.exception("详细错误:")
        sys.exit(1)

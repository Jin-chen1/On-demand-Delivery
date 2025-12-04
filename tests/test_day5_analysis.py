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


def test_single_simulation_analysis():
    """测试单次仿真的分析功能"""
    
    logger.info("="*70)
    logger.info("Day 5 测试：单次仿真分析")
    logger.info("="*70)
    
    # 1. 加载配置和数据
    config = ConfigManager()
    network_config = config.get('network')
    matrix_config = config.get('distance_matrix')
    
    data_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    logger.info("\n=== 加载路网数据 ===")
    graph, _ = osm_network.extract_osm_network(network_config, data_dir, force_download=False)
    
    logger.info("\n=== 加载距离矩阵 ===")
    dist_matrix, time_matrix, mapping = distance_matrix.compute_distance_matrices(
        graph, matrix_config, data_dir, force_recalculate=False
    )
    
    # 2. 运行仿真
    logger.info("\n=== 配置并运行仿真 ===")
    sim_config = {
        'simulation_duration': 1800.0,  # 30分钟
        'dispatch_interval': 60.0,
        'dispatcher_type': 'ortools',
        'dispatcher_config': {
            'time_limit_seconds': 3,
            'allow_insertion_to_active': True,
            'enable_batching': False
        }
    }
    
    env = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=sim_config
    )
    
    # 加载订单
    orders_file = orders_dir / "orders.csv"
    env.load_orders_from_csv(orders_file)
    
    # 初始化骑手
    courier_config = {
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    }
    env.initialize_couriers(num_couriers=10, courier_config=courier_config)
    
    # 运行仿真
    logger.info("开始仿真...")
    env.run(until=1800.0)
    
    # 3. 生成分析报告
    logger.info("\n=== 生成分析报告 ===")
    report_gen = ReportGenerator(output_dir=project_root / "outputs" / "reports")
    
    output_files = report_gen.generate_single_run_report(
        env=env,
        graph=graph,
        report_name="ortools_dynamic_insertion"
    )
    
    logger.info("\n生成的文件:")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")
    
    # 4. 单独测试各个可视化功能
    logger.info("\n=== 测试独立可视化功能 ===")
    vis_output_dir = project_root / "outputs" / "visualizations" / "test"
    visualizer = Visualizer(graph, output_dir=vis_output_dir)
    
    # 测试骑手路线图
    logger.info("生成骑手路线图...")
    routes_path = visualizer.plot_courier_routes(
        env.couriers,
        env.orders,
        title="Test: Courier Routes",
        filename="test_courier_routes.png",
        show_graph=True
    )
    logger.info(f"  路线图: {routes_path}")
    
    # 测试订单热力图
    logger.info("生成订单热力图...")
    heatmap_path = visualizer.plot_order_heatmap(
        env.orders,
        title="Test: Order Distribution",
        filename="test_order_heatmap.png"
    )
    logger.info(f"  热力图: {heatmap_path}")
    
    # 测试时间分布图
    logger.info("生成订单时间分布图...")
    temporal_path = visualizer.plot_temporal_demand(
        env.orders,
        time_window=300.0,
        title="Test: Order Arrival Pattern",
        filename="test_temporal_demand.png"
    )
    logger.info(f"  时间分布: {temporal_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Day 5 单次仿真分析测试完成")
    logger.info("="*70)
    
    return env, output_files


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
    
    orders_file = orders_dir / "orders.csv"
    
    # 2. 运行多个方法的仿真
    logger.info("\n=== 运行多方法仿真 ===")
    envs = {}
    
    # 方法1: Greedy
    logger.info("\n--- 运行 Greedy 调度器 ---")
    greedy_config = {
        'simulation_duration': 1800.0,
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
    env_greedy.initialize_couriers(num_couriers=10, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    env_greedy.run(until=1800.0)
    envs['Greedy'] = env_greedy
    
    # 方法2: OR-Tools (传统模式)
    logger.info("\n--- 运行 OR-Tools (传统模式) ---")
    ortools_traditional_config = {
        'simulation_duration': 1800.0,
        'dispatch_interval': 60.0,
        'dispatcher_type': 'ortools',
        'dispatcher_config': {
            'time_limit_seconds': 3,
            'allow_insertion_to_active': False,  # 传统模式
            'enable_batching': False
        }
    }
    
    env_ortools_trad = SimulationEnvironment(
        graph=graph,
        distance_matrix=dist_matrix,
        time_matrix=time_matrix,
        node_mapping=mapping,
        config=ortools_traditional_config
    )
    env_ortools_trad.load_orders_from_csv(orders_file)
    env_ortools_trad.initialize_couriers(num_couriers=10, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    env_ortools_trad.run(until=1800.0)
    envs['OR-Tools-Traditional'] = env_ortools_trad
    
    # 方法3: OR-Tools (动态插入)
    logger.info("\n--- 运行 OR-Tools (动态插入模式) ---")
    ortools_dynamic_config = {
        'simulation_duration': 1800.0,
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
    env_ortools_dyn.initialize_couriers(num_couriers=10, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    env_ortools_dyn.run(until=1800.0)
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
        logger.info("Day 5: 评估指标与可视化系统测试")
        logger.info("🚀"*35 + "\n")
        
        # 测试1: 单次仿真分析
        env, single_files = test_single_simulation_analysis()
        
        # 测试2: 多方法对比分析
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

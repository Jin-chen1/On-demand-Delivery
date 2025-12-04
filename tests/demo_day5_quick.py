"""
Day 5 快速演示：基于已有数据的分析
使用之前运行的仿真结果，快速生成图表
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.utils.config import ConfigManager
from src.data_preparation import osm_network, distance_matrix
from src.simulation.environment import SimulationEnvironment
from src.analysis import MetricsCalculator, Visualizer, ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_demo():
    """快速演示分析功能（基于最小数据集）"""
    
    logger.info("="*70)
    logger.info("Day 5 快速演示：基于现有数据的分析")
    logger.info("="*70)
    
    # 加载配置和数据
    config = ConfigManager()
    network_config = config.get('network')
    matrix_config = config.get('distance_matrix')
    
    data_dir = config.get_data_dir("processed")
    orders_dir = config.get_data_dir("orders")
    
    logger.info("\n=== Step 1: 加载路网数据 ===")
    graph, _ = osm_network.extract_osm_network(network_config, data_dir, force_download=False)
    logger.info(f"路网节点数: {graph.number_of_nodes()}")
    
    logger.info("\n=== Step 2: 加载距离矩阵 ===")
    dist_matrix, time_matrix, mapping = distance_matrix.compute_distance_matrices(
        graph, matrix_config, data_dir, force_recalculate=False
    )
    logger.info(f"距离矩阵大小: {len(mapping)} x {len(mapping)}")
    
    logger.info("\n=== Step 3: 运行快速仿真（5分钟）===")
    sim_config = {
        'simulation_duration': 300.0,  # 仅5分钟
        'dispatch_interval': 60.0,
        'dispatcher_type': 'greedy',  # 使用快速的Greedy
        'dispatcher_config': {}
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
    env.initialize_couriers(num_couriers=5, courier_config={
        'speed': {'mean': 15.0, 'std': 2.0, 'min': 10.0, 'max': 20.0},
        'capacity': {'max_orders': 3}
    })
    
    # 运行仿真
    env.run(until=300.0)
    
    logger.info("\n=== Step 4: 计算性能指标 ===")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_from_environment(env)
    
    # 保存指标
    metrics_dir = project_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    calculator.save_metrics(metrics, metrics_dir / "quick_demo_metrics.json")
    calculator.save_metrics(metrics, metrics_dir / "quick_demo_metrics.csv")
    
    logger.info("\n=== Step 5: 生成可视化 ===")
    vis_output_dir = project_root / "outputs" / "visualizations" / "quick_demo"
    visualizer = Visualizer(graph, output_dir=vis_output_dir)
    
    # 生成各类图表
    logger.info("生成骑手路线图...")
    routes_path = visualizer.plot_courier_routes(
        env.couriers,
        env.orders,
        title="Quick Demo: Courier Routes",
        filename="courier_routes.png",
        show_graph=False  # 不显示路网背景以加快速度
    )
    
    logger.info("生成订单热力图...")
    heatmap_path = visualizer.plot_order_heatmap(
        env.orders,
        title="Quick Demo: Order Distribution",
        filename="order_heatmap.png"
    )
    
    logger.info("生成订单时间分布...")
    temporal_path = visualizer.plot_temporal_demand(
        env.orders,
        time_window=60.0,  # 1分钟窗口
        title="Quick Demo: Order Arrival Pattern",
        filename="temporal_demand.png"
    )
    
    logger.info("\n=== 生成的文件 ===")
    logger.info(f"指标 JSON: {metrics_dir / 'quick_demo_metrics.json'}")
    logger.info(f"指标 CSV: {metrics_dir / 'quick_demo_metrics.csv'}")
    logger.info(f"路线图: {routes_path}")
    logger.info(f"热力图: {heatmap_path}")
    logger.info(f"时间分布: {temporal_path}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Day 5 快速演示完成")
    logger.info("="*70)
    
    logger.info("\n💡 提示:")
    logger.info("  1. 查看 outputs/metrics/ 目录获取性能指标数据")
    logger.info("  2. 查看 outputs/visualizations/quick_demo/ 目录获取图表")
    logger.info("  3. 运行完整测试: python tests/test_day5_analysis.py")
    
    return env, metrics


if __name__ == "__main__":
    try:
        env, metrics = quick_demo()
        sys.exit(0)
    except Exception as e:
        logger.error(f"演示过程出错: {str(e)}")
        logger.exception("详细错误:")
        sys.exit(1)

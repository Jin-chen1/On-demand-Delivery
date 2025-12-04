"""调试OR-Tools调度器问题"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.simulation import SimulationEnvironment

config = get_config()
processed_dir = config.get_data_dir('processed')
orders_dir = config.get_data_dir('orders')

# 加载路网
network_config = config.get_network_config()
graph, _ = extract_osm_network(network_config, processed_dir)

# 加载距离矩阵  
matrix_config = config.get_distance_matrix_config()
dist_matrix, time_matrix, mapping = compute_distance_matrices(graph, matrix_config, processed_dir)

# 创建仿真环境
sim_config = {
    'simulation_duration': 7200,  # 2小时测试
    'dispatch_interval': 60.0,
    'dispatcher_type': 'ortools',
    'use_gps_coords': False,
    'dispatcher_config': {
        'offline_mode': False,
        'time_limit_seconds': 5,  # 降低到5秒
        'soft_time_windows': True,
        'time_window_slack': 1200.0,  # 增加到20分钟
        'enable_batching': True,
        'allow_insertion_to_active': True
    }
}

sim_env = SimulationEnvironment(
    graph=graph,
    distance_matrix=dist_matrix,
    time_matrix=time_matrix,
    node_mapping=mapping,
    config=sim_config
)

# 加载订单
orders_file = orders_dir / 'uniform_grid_100.csv'
sim_env.load_orders_from_csv(orders_file)

# 调整订单到达时间
arrival_times = [order.arrival_time for order in sim_env.orders.values()]
min_arrival = min(arrival_times)
max_arrival = max(arrival_times)

# 缩放到[0, 5040] (7200*0.7)
target_max = 5040
for order in sim_env.orders.values():
    order.arrival_time = (order.arrival_time - min_arrival) / (max_arrival - min_arrival) * target_max

new_arrivals = [order.arrival_time for order in sim_env.orders.values()]
print(f'调整后到达时间: {min(new_arrivals):.0f}s - {max(new_arrivals):.0f}s')
print(f'平均到达率: {100 / max(new_arrivals) * 3600:.1f} 订单/小时')

# 初始化骑手
courier_config = config.get_courier_config()
sim_env.initialize_couriers(20, courier_config)

# 运行仿真
print('开始运行仿真...')
sim_env.run(until=7200)

# 检查结果
stats = sim_env.get_statistics()
print(f'\n结果:')
print(f'  总订单: {stats["total_orders"]}')
print(f'  待分配: {stats["pending_orders"]}')
print(f'  已完成: {stats["completed_orders"]}')
print(f'  完成率: {stats["completed_orders"]/stats["total_orders"]*100:.1f}%')

# 调度器统计
dispatcher_stats = sim_env.dispatcher.get_statistics()
print(f'\nOR-Tools统计:')
print(f'  调度次数: {dispatcher_stats["dispatch_count"]}')
print(f'  成功次数: {dispatcher_stats["solve_success_count"]}')
print(f'  失败次数: {dispatcher_stats["solve_failure_count"]}')

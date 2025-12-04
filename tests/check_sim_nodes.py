"""检查仿真环境中的节点匹配"""
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
    'simulation_duration': 7200,
    'dispatch_interval': 60.0,
    'dispatcher_type': 'ortools',
    'use_gps_coords': False,
    'dispatcher_config': {
        'offline_mode': False,
        'time_limit_seconds': 5,
        'soft_time_windows': True,
        'time_window_slack': 1200.0,
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

# 检查sim_env.node_to_idx
print(f'sim_env.node_to_idx节点数: {len(sim_env.node_to_idx)}')
node_keys = list(sim_env.node_to_idx.keys())[:5]
print(f'node_to_idx键示例: {node_keys}')
print(f'键类型: {type(node_keys[0])}')

# 检查订单节点
node_set = set(str(k) for k in sim_env.node_to_idx.keys())
print(f'转换为str后的节点集合大小: {len(node_set)}')

# 检查第一个订单
order = list(sim_env.orders.values())[0]
print(f'\n第一个订单:')
print(f'  merchant_node: {order.merchant_node} (type: {type(order.merchant_node)})')
print(f'  customer_node: {order.customer_node} (type: {type(order.customer_node)})')
print(f'  str(merchant_node): "{str(order.merchant_node)}"')
print(f'  str(merchant_node) in node_set: {str(order.merchant_node) in node_set}')

# 检查所有订单
merchant_match = sum(1 for o in sim_env.orders.values() if str(o.merchant_node) in node_set)
customer_match = sum(1 for o in sim_env.orders.values() if str(o.customer_node) in node_set)
print(f'\n所有订单检查:')
print(f'  商家节点匹配: {merchant_match}/{len(sim_env.orders)}')
print(f'  客户节点匹配: {customer_match}/{len(sim_env.orders)}')

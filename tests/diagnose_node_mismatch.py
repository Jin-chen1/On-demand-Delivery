"""诊断订单节点与距离矩阵节点不匹配的问题"""

import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent

# 加载距离矩阵的节点映射
with open(project_root / "data/processed/node_id_mapping.json", 'r') as f:
    mapping = json.load(f)

matrix_nodes = set(int(n) for n in mapping['node_to_idx'].keys())
print(f"距离矩阵节点数: {len(matrix_nodes)}")
print(f"示例节点: {list(matrix_nodes)[:10]}")

# 加载订单数据
orders_df = pd.read_csv(project_root / "data/orders/orders.csv")
print(f"\n订单总数: {len(orders_df)}")

# 检查商家节点覆盖率
merchant_nodes = set(orders_df['merchant_node'].unique())
print(f"唯一商家节点数: {len(merchant_nodes)}")
merchant_in_matrix = merchant_nodes & matrix_nodes
print(f"在距离矩阵中的商家节点: {len(merchant_in_matrix)}/{len(merchant_nodes)} ({len(merchant_in_matrix)/len(merchant_nodes)*100:.1f}%)")

# 检查客户节点覆盖率
customer_nodes = set(orders_df['customer_node'].unique())
print(f"唯一客户节点数: {len(customer_nodes)}")
customer_in_matrix = customer_nodes & matrix_nodes
print(f"在距离矩阵中的客户节点: {len(customer_in_matrix)}/{len(customer_nodes)} ({len(customer_in_matrix)/len(customer_nodes)*100:.1f}%)")

# 检查整体覆盖
all_order_nodes = merchant_nodes | customer_nodes
nodes_in_matrix = all_order_nodes & matrix_nodes
print(f"\n所有订单涉及的节点数: {len(all_order_nodes)}")
print(f"在距离矩阵中的节点数: {len(nodes_in_matrix)} ({len(nodes_in_matrix)/len(all_order_nodes)*100:.1f}%)")

# 检查路网总节点数
network_nodes_df = pd.read_csv(project_root / "data/processed/network_nodes.csv")
total_network_nodes = set(network_nodes_df['osmid'].astype(int))
print(f"\n路网总节点数: {len(total_network_nodes)}")

# 检查订单节点是否在路网中
nodes_in_network = all_order_nodes & total_network_nodes
print(f"订单节点在路网中的数量: {len(nodes_in_network)}/{len(all_order_nodes)} ({len(nodes_in_network)/len(all_order_nodes)*100:.1f}%)")

print("\n" + "="*60)
print("问题诊断：")
if len(nodes_in_matrix) == 0:
    print("❌ 严重问题：订单节点与距离矩阵节点完全不匹配！")
    print("   原因：订单生成使用全路网节点，距离矩阵使用采样节点")
    print("   解决方案：")
    print("   1. 修改订单生成器，使用距离矩阵的节点列表")
    print("   2. 或者重新生成距离矩阵，使用全部节点（不采样）")
elif len(nodes_in_matrix) / len(all_order_nodes) < 0.8:
    print("⚠️  警告：节点覆盖率过低（<80%）")
    print(f"   当前覆盖率: {len(nodes_in_matrix)/len(all_order_nodes)*100:.1f}%")
else:
    print("✓ 节点覆盖率良好")

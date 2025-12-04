"""将订单GPS坐标映射到苏拉特路网节点"""
import osmnx as ox
import pandas as pd
import numpy as np
from pathlib import Path

# 读取苏拉特路网
print("读取苏拉特路网...")
graphml_path = Path("D:/0On-demand Delivery/data/processed/surat/road_network.graphml")
G = ox.load_graphml(graphml_path)
print(f"路网: {G.number_of_nodes()} 节点")

# 读取订单数据
print("\n读取订单数据...")
df = pd.read_csv('D:/0On-demand Delivery/data/orders/real_delivery_single_city.csv')
print(f"订单数: {len(df)}")

# 获取所有路网节点坐标
print("\n提取路网节点坐标...")
node_coords = {}
for node in G.nodes():
    node_coords[node] = (G.nodes[node]['y'], G.nodes[node]['x'])

node_list = list(node_coords.keys())
node_lats = np.array([node_coords[n][0] for n in node_list])
node_lngs = np.array([node_coords[n][1] for n in node_list])

def find_nearest_node(lat, lng):
    """找到最近的路网节点"""
    # 使用简单的欧氏距离近似（对于小范围足够准确）
    distances = np.sqrt((node_lats - lat)**2 + (node_lngs - lng)**2)
    nearest_idx = np.argmin(distances)
    return node_list[nearest_idx], distances[nearest_idx] * 111000  # 转换为米

# 映射订单位置
print("\n映射订单位置...")
merchant_nodes = []
customer_nodes = []
merchant_dists = []
customer_dists = []

for idx, row in df.iterrows():
    # 映射商家位置
    m_node, m_dist = find_nearest_node(row['merchant_lat'], row['merchant_lng'])
    merchant_nodes.append(m_node)
    merchant_dists.append(m_dist)
    
    # 映射客户位置
    c_node, c_dist = find_nearest_node(row['customer_lat'], row['customer_lng'])
    customer_nodes.append(c_node)
    customer_dists.append(c_dist)

# 添加映射结果到数据框
df['merchant_node'] = merchant_nodes
df['customer_node'] = customer_nodes
df['merchant_mapping_dist'] = merchant_dists
df['customer_mapping_dist'] = customer_dists

print(f"\n映射统计:")
print(f"  商家映射距离: 平均={np.mean(merchant_dists):.0f}m, 最大={np.max(merchant_dists):.0f}m")
print(f"  客户映射距离: 平均={np.mean(customer_dists):.0f}m, 最大={np.max(customer_dists):.0f}m")

# 保存映射后的订单
output_path = 'D:/0On-demand Delivery/data/orders/real_delivery_surat_mapped.csv'
df.to_csv(output_path, index=False)
print(f"\n已保存映射后的订单: {output_path}")

# 统计使用的节点
unique_merchant_nodes = len(set(merchant_nodes))
unique_customer_nodes = len(set(customer_nodes))
all_used_nodes = set(merchant_nodes) | set(customer_nodes)
print(f"\n使用的节点数:")
print(f"  商家节点: {unique_merchant_nodes}")
print(f"  客户节点: {unique_customer_nodes}")
print(f"  总计: {len(all_used_nodes)}")

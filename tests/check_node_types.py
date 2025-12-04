"""检查节点类型匹配问题"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import pandas as pd

# 检查node_mapping的键格式
with open('data/processed/shanghai/node_id_mapping.json', 'r') as f:
    mapping = json.load(f)
node_keys = list(mapping['node_to_idx'].keys())[:5]
print(f'node_to_idx键格式: {node_keys}')
print(f'键类型: {type(node_keys[0])}')

# 检查订单的节点格式
df = pd.read_csv('data/orders/uniform_grid_100.csv')
merchant_nodes = df['merchant_node'].head().tolist()
print(f'订单merchant_node格式: {merchant_nodes}')
print(f'节点类型: {type(merchant_nodes[0])}')

# 检查转换后的格式
print(f'str(merchant_node[0]): "{str(merchant_nodes[0])}"')
print(f'str(int(merchant_node[0])): "{str(int(merchant_nodes[0]))}"')

# 检查匹配
node_set = set(mapping['node_to_idx'].keys())
match_direct = str(merchant_nodes[0]) in node_set
match_int = str(int(merchant_nodes[0])) in node_set
print(f'直接str匹配: {match_direct}')
print(f'先int再str匹配: {match_int}')

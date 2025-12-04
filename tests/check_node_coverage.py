"""检查订单节点覆盖率"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
import pandas as pd

# 加载node_mapping
with open('data/processed/shanghai/node_id_mapping.json', 'r') as f:
    mapping = json.load(f)
node_set = set(mapping['node_to_idx'].keys())
print(f'距离矩阵节点数: {len(node_set)}')

# 检查订单节点
df = pd.read_csv('data/orders/uniform_grid_100.csv')

merchant_match = 0
customer_match = 0
both_match = 0

for _, row in df.iterrows():
    m_in = str(int(row['merchant_node'])) in node_set
    c_in = str(int(row['customer_node'])) in node_set
    if m_in:
        merchant_match += 1
    if c_in:
        customer_match += 1
    if m_in and c_in:
        both_match += 1

print(f'商家节点匹配: {merchant_match}/{len(df)} ({merchant_match/len(df)*100:.1f}%)')
print(f'客户节点匹配: {customer_match}/{len(df)} ({customer_match/len(df)*100:.1f}%)')
print(f'两者都匹配: {both_match}/{len(df)} ({both_match/len(df)*100:.1f}%)')

# 检查不匹配的节点
print('\n不匹配的商家节点示例:')
for _, row in df.iterrows():
    m = str(int(row['merchant_node']))
    if m not in node_set:
        print(f'  {m}')
        break

print('\n不匹配的客户节点示例:')
for _, row in df.iterrows():
    c = str(int(row['customer_node']))
    if c not in node_set:
        print(f'  {c}')
        break

"""筛选单城市订单数据"""
import pandas as pd
import numpy as np

# 读取原始数据
df = pd.read_csv('D:/0On-demand Delivery/data/orders/real_delivery_100.csv')

# 选择苏拉特区域 (21-22N, 72-73E) - 与下载的路网匹配
mask = (df['merchant_lat'] >= 21) & (df['merchant_lat'] <= 22) & \
       (df['merchant_lng'] >= 72) & (df['merchant_lng'] <= 73) & \
       (df['merchant_lat'] > 1)  # 过滤异常坐标
df_city = df[mask].copy()

# 重新编号订单
df_city['order_id'] = range(1, len(df_city) + 1)

print(f'筛选出 {len(df_city)} 个单城市订单')
print(f'商家坐标范围:')
print(f'  lat: {df_city["merchant_lat"].min():.2f} - {df_city["merchant_lat"].max():.2f}')
print(f'  lng: {df_city["merchant_lng"].min():.2f} - {df_city["merchant_lng"].max():.2f}')

# 计算配送距离
R = 6371000
dlat = np.radians(df_city['customer_lat'] - df_city['merchant_lat'])
dlng = np.radians(df_city['customer_lng'] - df_city['merchant_lng'])
a = np.sin(dlat/2)**2 + np.cos(np.radians(df_city['merchant_lat'])) * np.cos(np.radians(df_city['customer_lat'])) * np.sin(dlng/2)**2
dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
print(f'\n配送距离统计:')
print(f'  平均: {dist.mean()/1000:.1f}km')
print(f'  最大: {dist.max()/1000:.1f}km')
print(f'  最小: {dist.min()/1000:.1f}km')

# 保存
output_path = 'D:/0On-demand Delivery/data/orders/real_delivery_single_city.csv'
df_city.to_csv(output_path, index=False)
print(f'\n已保存到: {output_path}')

"""
筛选LaDe上海数据中在现有路网范围内的订单
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 现有上海路网范围
SHANGHAI_BOUNDS = {
    'lat_min': 31.20, 'lat_max': 31.25,
    'lng_min': 121.42, 'lng_max': 121.47
}

INPUT_FILE = "D:/0On-demand Delivery/data/lade/delivery/delivery_sh.csv"
OUTPUT_DIR = Path("D:/0On-demand Delivery/data/orders")

print("=" * 60)
print("筛选LaDe上海数据（匹配现有路网范围）")
print("=" * 60)
print(f"路网范围: ({SHANGHAI_BOUNDS['lat_min']}-{SHANGHAI_BOUNDS['lat_max']}N, "
      f"{SHANGHAI_BOUNDS['lng_min']}-{SHANGHAI_BOUNDS['lng_max']}E)")

# 读取数据
print("\n读取LaDe上海数据...")
df = pd.read_csv(INPUT_FILE)
print(f"原始记录: {len(df):,}")

# 筛选在路网范围内的订单
mask = (
    (df['accept_gps_lat'] >= SHANGHAI_BOUNDS['lat_min']) &
    (df['accept_gps_lat'] <= SHANGHAI_BOUNDS['lat_max']) &
    (df['accept_gps_lng'] >= SHANGHAI_BOUNDS['lng_min']) &
    (df['accept_gps_lng'] <= SHANGHAI_BOUNDS['lng_max']) &
    (df['lat'] >= SHANGHAI_BOUNDS['lat_min']) &
    (df['lat'] <= SHANGHAI_BOUNDS['lat_max']) &
    (df['lng'] >= SHANGHAI_BOUNDS['lng_min']) &
    (df['lng'] <= SHANGHAI_BOUNDS['lng_max'])
)
df_filtered = df[mask].copy()
print(f"路网范围内订单: {len(df_filtered):,}")

# 解析时间
def parse_time(time_str):
    try:
        parts = time_str.split(' ')
        time_part = parts[1] if len(parts) > 1 else parts[0]
        h, m, s = map(int, time_part.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return None

df_filtered['arrival_seconds'] = df_filtered['accept_time'].apply(parse_time)
df_filtered['delivery_seconds'] = df_filtered['delivery_time'].apply(parse_time)
df_filtered = df_filtered.dropna(subset=['arrival_seconds', 'delivery_seconds'])

# 计算配送时间
df_filtered['actual_delivery_time'] = df_filtered['delivery_seconds'] - df_filtered['arrival_seconds']
df_filtered.loc[df_filtered['actual_delivery_time'] < 0, 'actual_delivery_time'] += 86400

# 过滤合理配送时间（5分钟到2小时）
reasonable_mask = (df_filtered['actual_delivery_time'] >= 300) & (df_filtered['actual_delivery_time'] <= 7200)
df_filtered = df_filtered[reasonable_mask]
print(f"合理配送时间订单: {len(df_filtered):,}")

# 计算配送距离
def haversine_distance(lat1, lng1, lat2, lng2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

df_filtered['distance'] = haversine_distance(
    df_filtered['accept_gps_lat'], df_filtered['accept_gps_lng'],
    df_filtered['lat'], df_filtered['lng']
)

# 过滤合理距离（100米到10公里，因为路网范围约5km）
distance_mask = (df_filtered['distance'] >= 100) & (df_filtered['distance'] <= 10000)
df_filtered = df_filtered[distance_mask]
print(f"合理距离订单: {len(df_filtered):,}")

# 统计
print(f"\n数据统计:")
print(f"  配送距离: 平均 {df_filtered['distance'].mean()/1000:.2f}km, 中位数 {df_filtered['distance'].median()/1000:.2f}km")
print(f"  配送时间: 平均 {df_filtered['actual_delivery_time'].mean()/60:.0f}分钟, 中位数 {df_filtered['actual_delivery_time'].median()/60:.0f}分钟")

# 转换为项目格式
def convert_to_project_format(df_src, num_orders, start_time=0):
    df_sorted = df_src.sort_values('arrival_seconds').head(num_orders).copy()
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['new_order_id'] = range(1, len(df_sorted) + 1)
    
    min_arrival = df_sorted['arrival_seconds'].min()
    df_sorted['adjusted_arrival'] = df_sorted['arrival_seconds'] - min_arrival + start_time
    
    preparation_time = 300  # 5分钟备餐
    delivery_window = 3600  # 1小时配送窗口
    
    orders = pd.DataFrame({
        'order_id': df_sorted['new_order_id'],
        'arrival_time': df_sorted['adjusted_arrival'],
        'merchant_lat': df_sorted['accept_gps_lat'],
        'merchant_lng': df_sorted['accept_gps_lng'],
        'customer_lat': df_sorted['lat'],
        'customer_lng': df_sorted['lng'],
        'preparation_time': preparation_time,
        'delivery_window': delivery_window,
        'earliest_pickup_time': df_sorted['adjusted_arrival'] + preparation_time,  # 最早取货时间
        'latest_delivery_time': df_sorted['adjusted_arrival'] + delivery_window,   # 最晚送达时间
        'actual_delivery_time': df_sorted['actual_delivery_time'],
        'distance': df_sorted['distance'],
    })
    return orders

# 生成订单文件
print("\n生成订单文件...")

orders_100 = convert_to_project_format(df_filtered, 100)
orders_100.to_csv(OUTPUT_DIR / "lade_shanghai_matched_100.csv", index=False)
print(f"  lade_shanghai_matched_100.csv: {len(orders_100)}条")

orders_500 = convert_to_project_format(df_filtered, 500)
orders_500.to_csv(OUTPUT_DIR / "lade_shanghai_matched_500.csv", index=False)
print(f"  lade_shanghai_matched_500.csv: {len(orders_500)}条")

orders_1000 = convert_to_project_format(df_filtered, 1000)
orders_1000.to_csv(OUTPUT_DIR / "lade_shanghai_matched_1000.csv", index=False)
print(f"  lade_shanghai_matched_1000.csv: {len(orders_1000)}条")

# 验证
print(f"\n坐标范围验证:")
print(f"  商家 lat: {orders_100['merchant_lat'].min():.4f} - {orders_100['merchant_lat'].max():.4f}")
print(f"  商家 lng: {orders_100['merchant_lng'].min():.4f} - {orders_100['merchant_lng'].max():.4f}")
print(f"  客户 lat: {orders_100['customer_lat'].min():.4f} - {orders_100['customer_lat'].max():.4f}")
print(f"  客户 lng: {orders_100['customer_lng'].min():.4f} - {orders_100['customer_lng'].max():.4f}")

print("\n完成! 订单数据已与现有上海路网匹配")

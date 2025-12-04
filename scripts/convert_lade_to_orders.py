"""
将菜鸟LaDe上海数据转换为项目订单格式

LaDe数据字段:
- order_id: 订单ID
- courier_id: 骑手ID  
- lng, lat: 配送目的地GPS（客户位置）
- accept_gps_lng, accept_gps_lat: 接单位置GPS（商家/取货位置）
- accept_time: 接单时间（订单到达时间）
- delivery_time: 送达时间

项目订单格式:
- order_id: 订单ID
- arrival_time: 订单到达时间（秒）
- merchant_lat, merchant_lng: 商家GPS
- customer_lat, customer_lng: 客户GPS
- preparation_time: 备餐时间
- delivery_window: 配送时间窗口
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# 配置
INPUT_FILE = "D:/0On-demand Delivery/data/lade/delivery/delivery_sh.csv"
OUTPUT_DIR = Path("D:/0On-demand Delivery/data/orders")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 读取数据
print("读取LaDe上海数据...")
df = pd.read_csv(INPUT_FILE)
print(f"总记录数: {len(df):,}")

# 检查坐标范围
print(f"\n原始坐标范围:")
print(f"  配送位置 lng: {df['lng'].min():.4f} - {df['lng'].max():.4f}")
print(f"  配送位置 lat: {df['lat'].min():.4f} - {df['lat'].max():.4f}")
print(f"  接单位置 lng: {df['accept_gps_lng'].min():.4f} - {df['accept_gps_lng'].max():.4f}")
print(f"  接单位置 lat: {df['accept_gps_lat'].min():.4f} - {df['accept_gps_lat'].max():.4f}")

# 过滤有效数据
# 1. 去除GPS异常值
valid_mask = (
    (df['lng'] > 120) & (df['lng'] < 123) &
    (df['lat'] > 30) & (df['lat'] < 32) &
    (df['accept_gps_lng'] > 120) & (df['accept_gps_lng'] < 123) &
    (df['accept_gps_lat'] > 30) & (df['accept_gps_lat'] < 32)
)
df_valid = df[valid_mask].copy()
print(f"\n过滤后有效记录: {len(df_valid):,}")

# 2. 解析时间
def parse_time(time_str):
    """解析时间字符串为秒数（从当天0点开始）"""
    try:
        # 格式: "06-04 11:05:00"
        parts = time_str.split(' ')
        time_part = parts[1] if len(parts) > 1 else parts[0]
        h, m, s = map(int, time_part.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return None

df_valid['arrival_seconds'] = df_valid['accept_time'].apply(parse_time)
df_valid['delivery_seconds'] = df_valid['delivery_time'].apply(parse_time)

# 去除时间解析失败的记录
df_valid = df_valid.dropna(subset=['arrival_seconds', 'delivery_seconds'])
print(f"时间解析后有效记录: {len(df_valid):,}")

# 3. 计算实际配送时间
df_valid['actual_delivery_time'] = df_valid['delivery_seconds'] - df_valid['arrival_seconds']
# 处理跨天情况
df_valid.loc[df_valid['actual_delivery_time'] < 0, 'actual_delivery_time'] += 86400

# 过滤合理的配送时间（5分钟到4小时）
reasonable_mask = (df_valid['actual_delivery_time'] >= 300) & (df_valid['actual_delivery_time'] <= 14400)
df_valid = df_valid[reasonable_mask]
print(f"合理配送时间记录: {len(df_valid):,}")

# 4. 计算配送距离
def haversine_distance(lat1, lng1, lat2, lng2):
    """计算两点间的Haversine距离（米）"""
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

df_valid['distance'] = haversine_distance(
    df_valid['accept_gps_lat'], df_valid['accept_gps_lng'],
    df_valid['lat'], df_valid['lng']
)

# 过滤合理距离（100米到30公里）
distance_mask = (df_valid['distance'] >= 100) & (df_valid['distance'] <= 30000)
df_valid = df_valid[distance_mask]
print(f"合理距离记录: {len(df_valid):,}")

# 统计
print(f"\n数据统计:")
print(f"  配送距离: 平均 {df_valid['distance'].mean()/1000:.1f}km, 中位数 {df_valid['distance'].median()/1000:.1f}km")
print(f"  配送时间: 平均 {df_valid['actual_delivery_time'].mean()/60:.0f}分钟, 中位数 {df_valid['actual_delivery_time'].median()/60:.0f}分钟")

# 转换为项目格式
def convert_to_project_format(df_src, num_orders, start_time=0):
    """转换为项目订单格式"""
    # 按时间排序并选取
    df_sorted = df_src.sort_values('arrival_seconds').head(num_orders).copy()
    
    # 重新编号
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['new_order_id'] = range(1, len(df_sorted) + 1)
    
    # 调整到达时间（从start_time开始）
    min_arrival = df_sorted['arrival_seconds'].min()
    df_sorted['adjusted_arrival'] = df_sorted['arrival_seconds'] - min_arrival + start_time
    
    # 创建输出DataFrame
    orders = pd.DataFrame({
        'order_id': df_sorted['new_order_id'],
        'arrival_time': df_sorted['adjusted_arrival'],
        'merchant_lat': df_sorted['accept_gps_lat'],
        'merchant_lng': df_sorted['accept_gps_lng'],
        'customer_lat': df_sorted['lat'],
        'customer_lng': df_sorted['lng'],
        'preparation_time': 300,  # 默认5分钟备餐
        'delivery_window': 3600,  # 默认1小时配送窗口
        'actual_delivery_time': df_sorted['actual_delivery_time'],  # 保留实际配送时间用于验证
        'distance': df_sorted['distance'],  # 保留距离用于验证
    })
    
    return orders

# 生成不同规模的订单文件
print("\n生成订单文件...")

# 100订单
orders_100 = convert_to_project_format(df_valid, 100)
orders_100.to_csv(OUTPUT_DIR / "lade_shanghai_100.csv", index=False)
print(f"  lade_shanghai_100.csv: {len(orders_100)}条")

# 500订单
orders_500 = convert_to_project_format(df_valid, 500)
orders_500.to_csv(OUTPUT_DIR / "lade_shanghai_500.csv", index=False)
print(f"  lade_shanghai_500.csv: {len(orders_500)}条")

# 1000订单
orders_1000 = convert_to_project_format(df_valid, 1000)
orders_1000.to_csv(OUTPUT_DIR / "lade_shanghai_1000.csv", index=False)
print(f"  lade_shanghai_1000.csv: {len(orders_1000)}条")

# 验证坐标范围
print(f"\n转换后坐标范围:")
print(f"  商家 lat: {orders_100['merchant_lat'].min():.4f} - {orders_100['merchant_lat'].max():.4f}")
print(f"  商家 lng: {orders_100['merchant_lng'].min():.4f} - {orders_100['merchant_lng'].max():.4f}")
print(f"  客户 lat: {orders_100['customer_lat'].min():.4f} - {orders_100['customer_lat'].max():.4f}")
print(f"  客户 lng: {orders_100['customer_lng'].min():.4f} - {orders_100['customer_lng'].max():.4f}")

# 检查与现有上海路网的匹配度
shanghai_bounds = {
    'lat_min': 31.20, 'lat_max': 31.25,
    'lng_min': 121.42, 'lng_max': 121.47
}

in_bounds = (
    (orders_100['merchant_lat'] >= shanghai_bounds['lat_min']) &
    (orders_100['merchant_lat'] <= shanghai_bounds['lat_max']) &
    (orders_100['merchant_lng'] >= shanghai_bounds['lng_min']) &
    (orders_100['merchant_lng'] <= shanghai_bounds['lng_max'])
)
print(f"\n与现有上海路网匹配: {in_bounds.sum()}/{len(orders_100)} ({in_bounds.sum()/len(orders_100)*100:.1f}%)")

if in_bounds.sum() < len(orders_100) * 0.5:
    print("警告: 大部分订单不在现有上海路网范围内，需要下载更大范围的路网")

print("\n完成!")

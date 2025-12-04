"""
分析 LaDe 数据集中取货点（商家）的出现频次
检验是否满足 M/M/c 排队系统建模的数据要求

M/M/c 排队系统建模要求：
1. 每个商家需要有多个订单到达，才能估计到达率 λ
2. 需要足够的样本量来估计服务率 μ（备餐时间）
3. 理想情况下，每个商家至少需要 5-10 个订单才能进行可靠的参数估计
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# 项目根目录
project_root = Path(__file__).parent.parent

# 数据文件
ORDERS_FILE_100 = project_root / "data" / "orders" / "lade_shanghai_matched_100.csv"
ORDERS_FILE_500 = project_root / "data" / "orders" / "lade_shanghai_matched_500.csv"
ORDERS_FILE_1000 = project_root / "data" / "orders" / "lade_shanghai_matched_1000.csv"
RAW_FILE = project_root / "data" / "lade" / "delivery" / "delivery_sh.csv"


def analyze_merchant_frequency(orders_file: Path, precision: int = 4):
    """
    分析取货点出现频次
    
    Args:
        orders_file: 订单文件路径
        precision: GPS坐标精度（小数位数），用于判定是否为同一商家
    
    Returns:
        分析结果字典
    """
    print(f"\n{'='*60}")
    print(f"分析文件: {orders_file.name}")
    print(f"坐标精度: {precision} 位小数")
    print(f"{'='*60}")
    
    # 读取数据
    df = pd.read_csv(orders_file)
    print(f"总订单数: {len(df)}")
    
    # 创建商家位置标识（四舍五入到指定精度）
    df['merchant_key'] = df.apply(
        lambda row: f"{round(row['merchant_lat'], precision)},{round(row['merchant_lng'], precision)}",
        axis=1
    )
    
    # 统计每个取货点的订单数
    merchant_counts = df['merchant_key'].value_counts()
    
    # 基本统计
    total_merchants = len(merchant_counts)
    single_order_merchants = (merchant_counts == 1).sum()
    multi_order_merchants = (merchant_counts > 1).sum()
    
    print(f"\n--- 取货点统计 ---")
    print(f"唯一取货点数量: {total_merchants}")
    print(f"仅出现1次的取货点: {single_order_merchants} ({single_order_merchants/total_merchants*100:.1f}%)")
    print(f"出现多次的取货点: {multi_order_merchants} ({multi_order_merchants/total_merchants*100:.1f}%)")
    
    # 频次分布
    print(f"\n--- 订单频次分布 ---")
    freq_distribution = merchant_counts.value_counts().sort_index()
    for freq, count in freq_distribution.head(10).items():
        pct = count / total_merchants * 100
        print(f"  {freq}次订单: {count} 个商家 ({pct:.1f}%)")
    
    if len(freq_distribution) > 10:
        remaining = freq_distribution.iloc[10:].sum()
        print(f"  >10次订单: {remaining} 个商家")
    
    # M/M/c 建模可行性分析
    print(f"\n--- M/M/c 排队建模可行性 ---")
    
    # 标准1：至少5个订单才能估计参数
    min_orders_for_modeling = 5
    viable_merchants_5 = (merchant_counts >= min_orders_for_modeling).sum()
    orders_from_viable_5 = df[df['merchant_key'].isin(
        merchant_counts[merchant_counts >= min_orders_for_modeling].index
    )].shape[0]
    
    print(f"标准1（≥5单）: {viable_merchants_5} 个商家可建模")
    print(f"  覆盖订单数: {orders_from_viable_5} ({orders_from_viable_5/len(df)*100:.1f}%)")
    
    # 标准2：至少3个订单
    min_orders_for_modeling = 3
    viable_merchants_3 = (merchant_counts >= min_orders_for_modeling).sum()
    orders_from_viable_3 = df[df['merchant_key'].isin(
        merchant_counts[merchant_counts >= min_orders_for_modeling].index
    )].shape[0]
    
    print(f"标准2（≥3单）: {viable_merchants_3} 个商家可建模")
    print(f"  覆盖订单数: {orders_from_viable_3} ({orders_from_viable_3/len(df)*100:.1f}%)")
    
    # 标准3：至少2个订单
    viable_merchants_2 = (merchant_counts >= 2).sum()
    orders_from_viable_2 = df[df['merchant_key'].isin(
        merchant_counts[merchant_counts >= 2].index
    )].shape[0]
    
    print(f"标准3（≥2单）: {viable_merchants_2} 个商家可建模")
    print(f"  覆盖订单数: {orders_from_viable_2} ({orders_from_viable_2/len(df)*100:.1f}%)")
    
    # Top商家
    print(f"\n--- Top 10 高频商家 ---")
    for i, (merchant, count) in enumerate(merchant_counts.head(10).items()):
        print(f"  {i+1}. {merchant}: {count} 单")
    
    return {
        'file': orders_file.name,
        'total_orders': len(df),
        'unique_merchants': total_merchants,
        'single_order_merchants': single_order_merchants,
        'multi_order_merchants': multi_order_merchants,
        'viable_merchants_5': viable_merchants_5,
        'viable_merchants_3': viable_merchants_3,
        'viable_merchants_2': viable_merchants_2,
        'max_orders_per_merchant': merchant_counts.max(),
        'merchant_counts': merchant_counts
    }


def analyze_raw_data_in_area(raw_file: Path, bounds: dict, precision: int = 4):
    """
    分析原始数据中实验区域内的商家频次
    
    Args:
        raw_file: 原始LaDe数据文件
        bounds: 区域边界 {lat_min, lat_max, lng_min, lng_max}
        precision: GPS坐标精度
    """
    print(f"\n{'='*60}")
    print(f"分析原始数据中实验区域内的商家频次")
    print(f"区域范围: lat[{bounds['lat_min']}, {bounds['lat_max']}], lng[{bounds['lng_min']}, {bounds['lng_max']}]")
    print(f"{'='*60}")
    
    # 读取原始数据
    print("读取原始数据...")
    df = pd.read_csv(raw_file)
    print(f"原始数据总量: {len(df):,}")
    
    # 筛选实验区域内的订单
    in_area_mask = (
        (df['accept_gps_lat'] >= bounds['lat_min']) &
        (df['accept_gps_lat'] <= bounds['lat_max']) &
        (df['accept_gps_lng'] >= bounds['lng_min']) &
        (df['accept_gps_lng'] <= bounds['lng_max']) &
        (df['lat'] >= bounds['lat_min']) &
        (df['lat'] <= bounds['lat_max']) &
        (df['lng'] >= bounds['lng_min']) &
        (df['lng'] <= bounds['lng_max'])
    )
    
    df_area = df[in_area_mask].copy()
    print(f"实验区域内订单: {len(df_area):,} ({len(df_area)/len(df)*100:.2f}%)")
    
    if len(df_area) == 0:
        print("警告: 实验区域内没有订单!")
        return None
    
    # 创建商家位置标识
    df_area['merchant_key'] = df_area.apply(
        lambda row: f"{round(row['accept_gps_lat'], precision)},{round(row['accept_gps_lng'], precision)}",
        axis=1
    )
    
    # 统计
    merchant_counts = df_area['merchant_key'].value_counts()
    total_merchants = len(merchant_counts)
    
    print(f"\n--- 区域内商家统计 ---")
    print(f"唯一商家数: {total_merchants}")
    print(f"平均每商家订单数: {len(df_area)/total_merchants:.1f}")
    print(f"最多订单商家: {merchant_counts.max()} 单")
    
    # 频次分布
    print(f"\n--- 订单频次分布 ---")
    for threshold in [1, 2, 3, 5, 10, 20, 50]:
        count = (merchant_counts >= threshold).sum()
        orders = df_area[df_area['merchant_key'].isin(
            merchant_counts[merchant_counts >= threshold].index
        )].shape[0]
        print(f"  ≥{threshold}单: {count} 商家, 覆盖 {orders} 订单 ({orders/len(df_area)*100:.1f}%)")
    
    # M/M/c建模结论
    print(f"\n--- M/M/c 排队建模结论 ---")
    viable_5 = (merchant_counts >= 5).sum()
    if viable_5 >= 10:
        print(f"✓ 有 {viable_5} 个商家订单量≥5，可进行 M/M/c 建模")
    else:
        print(f"✗ 仅 {viable_5} 个商家订单量≥5，样本不足")
    
    # 建议
    avg_orders = len(df_area) / total_merchants
    if avg_orders < 3:
        print(f"\n建议: 平均每商家仅 {avg_orders:.1f} 单，考虑：")
        print("  1. 扩大实验区域范围")
        print("  2. 使用更多天的数据")
        print("  3. 降低坐标精度以合并相邻位置")
    
    return {
        'total_orders_in_area': len(df_area),
        'unique_merchants': total_merchants,
        'merchant_counts': merchant_counts
    }


def main():
    """主函数"""
    print("="*60)
    print("LaDe 数据集商家频次分析")
    print("检验 M/M/c 排队系统建模可行性")
    print("="*60)
    
    # 分析已处理的订单文件
    results = []
    
    for orders_file in [ORDERS_FILE_100, ORDERS_FILE_500, ORDERS_FILE_1000]:
        if orders_file.exists():
            result = analyze_merchant_frequency(orders_file, precision=4)
            results.append(result)
    
    # 分析原始数据中实验区域
    if RAW_FILE.exists():
        # 上海实验区域边界（与config.yaml中的路网范围一致）
        shanghai_bounds = {
            'lat_min': 31.20,
            'lat_max': 31.25,
            'lng_min': 121.42,
            'lng_max': 121.47
        }
        
        raw_result = analyze_raw_data_in_area(RAW_FILE, shanghai_bounds, precision=4)
        
        # 尝试更宽松的坐标精度
        print("\n" + "="*60)
        print("尝试降低坐标精度（合并相邻位置）")
        print("="*60)
        
        for precision in [3, 2]:
            print(f"\n--- 精度: {precision} 位小数 ---")
            raw_result_lower = analyze_raw_data_in_area(RAW_FILE, shanghai_bounds, precision=precision)
    
    # 总结
    print("\n" + "="*60)
    print("总结与建议")
    print("="*60)
    
    if results:
        for r in results:
            print(f"\n{r['file']}:")
            print(f"  - 唯一商家: {r['unique_merchants']}")
            print(f"  - 可建模商家(≥5单): {r['viable_merchants_5']}")
            print(f"  - 单商家最大订单数: {r['max_orders_per_merchant']}")
    
    print("\n建模建议:")
    print("1. 如果大部分取货点仅出现1次，说明数据中每个订单来自不同商家")
    print("2. 此时不适合对单个商家进行M/M/c排队建模")
    print("3. 替代方案：使用整体统计的备餐时间分布（Gamma分布）来模拟商家行为")
    print("4. 或者将相邻位置的商家合并为一个'虚拟商家'进行建模")


if __name__ == "__main__":
    main()

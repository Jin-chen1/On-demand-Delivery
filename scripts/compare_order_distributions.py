"""
对比原始订单数据与多中心订单数据的分布差异
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ORDER_DIR = Path("D:/0On-demand Delivery/data/orders")
OUTPUT_DIR = Path("D:/0On-demand Delivery/docs/figures")


def main():
    # 加载数据
    old_100 = pd.read_csv(ORDER_DIR / "lade_shanghai_matched_100.csv")
    new_100 = pd.read_csv(ORDER_DIR / "lade_shanghai_diverse_100.csv")
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 图1：原始100单商家分布
    axes[0, 0].scatter(
        old_100['merchant_lng'], old_100['merchant_lat'],
        c='red', s=60, marker='^', alpha=0.8, edgecolors='white'
    )
    axes[0, 0].set_xlabel('Longitude', fontsize=11)
    axes[0, 0].set_ylabel('Latitude', fontsize=11)
    axes[0, 0].set_title('Original 100 Orders - Merchants\n(Single-Center)', 
                         fontsize=13, fontweight='bold', color='red')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 标注范围
    lat_range = old_100['merchant_lat'].max() - old_100['merchant_lat'].min()
    lng_range = old_100['merchant_lng'].max() - old_100['merchant_lng'].min()
    axes[0, 0].text(0.02, 0.98, f'Lat spread: {lat_range*111:.2f} km\nLng spread: {lng_range*95:.2f} km',
                    transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 图2：多中心100单商家分布
    scatter = axes[0, 1].scatter(
        new_100['merchant_lng'], new_100['merchant_lat'],
        c=new_100['cluster'], cmap='tab10', s=60, marker='^', alpha=0.8, edgecolors='white'
    )
    axes[0, 1].set_xlabel('Longitude', fontsize=11)
    axes[0, 1].set_ylabel('Latitude', fontsize=11)
    axes[0, 1].set_title('Diverse 100 Orders - Merchants\n(Multi-Center, 10 Zones)', 
                         fontsize=13, fontweight='bold', color='green')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Zone ID')
    
    lat_range = new_100['merchant_lat'].max() - new_100['merchant_lat'].min()
    lng_range = new_100['merchant_lng'].max() - new_100['merchant_lng'].min()
    axes[0, 1].text(0.02, 0.98, f'Lat spread: {lat_range*111:.2f} km\nLng spread: {lng_range*95:.2f} km',
                    transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 图3：原始100单完整分布（商家+客户+连线）
    for i in range(len(old_100)):
        row = old_100.iloc[i]
        axes[1, 0].plot(
            [row['merchant_lng'], row['customer_lng']],
            [row['merchant_lat'], row['customer_lat']],
            color='purple', linewidth=0.5, alpha=0.3
        )
    axes[1, 0].scatter(old_100['merchant_lng'], old_100['merchant_lat'],
                       c='red', s=40, marker='^', label='Merchants')
    axes[1, 0].scatter(old_100['customer_lng'], old_100['customer_lat'],
                       c='blue', s=20, marker='o', alpha=0.6, label='Customers')
    axes[1, 0].set_xlabel('Longitude', fontsize=11)
    axes[1, 0].set_ylabel('Latitude', fontsize=11)
    axes[1, 0].set_title('Original Data: Star-Shaped Pattern\n(All merchants at one location)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图4：多中心100单完整分布
    for i in range(len(new_100)):
        row = new_100.iloc[i]
        axes[1, 1].plot(
            [row['merchant_lng'], row['customer_lng']],
            [row['merchant_lat'], row['customer_lat']],
            color='purple', linewidth=0.5, alpha=0.3
        )
    axes[1, 1].scatter(new_100['merchant_lng'], new_100['merchant_lat'],
                       c='red', s=40, marker='^', label='Merchants')
    axes[1, 1].scatter(new_100['customer_lng'], new_100['customer_lat'],
                       c='blue', s=20, marker='o', alpha=0.6, label='Customers')
    axes[1, 1].set_xlabel('Longitude', fontsize=11)
    axes[1, 1].set_ylabel('Latitude', fontsize=11)
    axes[1, 1].set_title('Diverse Data: Multi-Center Pattern\n(Merchants across 10 zones)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Order Distribution Comparison: Single-Center vs Multi-Center\n'
                 '(Cainiao-AI LaDe Shanghai Dataset)',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "order_distribution_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Comparison figure saved: {output_path}")
    
    # 打印统计对比
    print("\n" + "=" * 60)
    print("Statistical Comparison")
    print("=" * 60)
    
    print("\n| Metric | Original | Diverse | Improvement |")
    print("|--------|----------|---------|-------------|")
    
    old_lat_spread = old_100['merchant_lat'].max() - old_100['merchant_lat'].min()
    new_lat_spread = new_100['merchant_lat'].max() - new_100['merchant_lat'].min()
    print(f"| Lat spread (km) | {old_lat_spread*111:.2f} | {new_lat_spread*111:.1f} | {new_lat_spread/old_lat_spread:.0f}x |")
    
    old_lng_spread = old_100['merchant_lng'].max() - old_100['merchant_lng'].min()
    new_lng_spread = new_100['merchant_lng'].max() - new_100['merchant_lng'].min()
    print(f"| Lng spread (km) | {old_lng_spread*95:.2f} | {new_lng_spread*95:.1f} | {new_lng_spread/old_lng_spread:.0f}x |")
    
    old_unique = len(old_100.groupby(['merchant_lat', 'merchant_lng']))
    new_unique = len(new_100.groupby(['merchant_lat', 'merchant_lng']))
    print(f"| Unique merchants | {old_unique} | {new_unique} | {new_unique/old_unique:.1f}x |")
    
    print(f"| Zones covered | 1 | {new_100['cluster'].nunique()} | {new_100['cluster'].nunique()}x |")
    
    print(f"| Avg distance (km) | {old_100['distance'].mean()/1000:.2f} | {new_100['distance'].mean()/1000:.2f} | - |")


if __name__ == "__main__":
    main()

"""
Day 2 结果可视化
展示订单到达时间分布
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def visualize_order_arrivals(result_dir: Path):
    """可视化订单到达"""
    
    # 读取事件数据
    events_file = result_dir / "events.csv"
    if not events_file.exists():
        print(f"事件文件不存在: {events_file}")
        return
    
    events_df = pd.read_csv(events_file)
    
    # 筛选订单到达事件
    arrival_events = events_df[events_df['event_type'] == 'order_arrival']
    
    print(f"订单到达事件数: {len(arrival_events)}")
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1: 订单到达时间序列
    ax1 = axes[0]
    ax1.scatter(arrival_events['timestamp'], arrival_events['entity_id'], 
                alpha=0.6, s=20, color='steelblue')
    ax1.set_xlabel('Simulation Time (seconds)', fontsize=12)
    ax1.set_ylabel('Order ID', fontsize=12)
    ax1.set_title('Order Arrival Timeline (Day 2 Test)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 订单到达率分布（时间窗口统计）
    ax2 = axes[1]
    bin_width = 300  # 5分钟时间窗
    bins = range(0, int(arrival_events['timestamp'].max()) + bin_width, bin_width)
    ax2.hist(arrival_events['timestamp'], bins=bins, 
             color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Simulation Time (seconds)', fontsize=12)
    ax2.set_ylabel('Number of Orders', fontsize=12)
    ax2.set_title('Order Arrival Rate Distribution (5-minute bins)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = result_dir / "order_arrival_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化图表已保存: {output_file}")
    
    plt.close()
    
    # 打印统计信息
    print("\n订单到达统计:")
    print(f"  首个订单到达时间: {arrival_events['timestamp'].min():.2f}秒")
    print(f"  最后订单到达时间: {arrival_events['timestamp'].max():.2f}秒")
    print(f"  时间跨度: {(arrival_events['timestamp'].max() - arrival_events['timestamp'].min())/60:.2f}分钟")
    print(f"  平均到达间隔: {arrival_events['timestamp'].diff().mean():.2f}秒")


if __name__ == "__main__":
    # 找到最新的测试结果目录
    results_base = project_root / "data" / "simulation_results"
    
    if not results_base.exists():
        print("结果目录不存在")
        sys.exit(1)
    
    # 获取所有day2测试目录
    day2_dirs = sorted([d for d in results_base.glob("day2_test_*") if d.is_dir()])
    
    if not day2_dirs:
        print("未找到Day 2测试结果")
        sys.exit(1)
    
    # 使用最新的结果
    latest_result = day2_dirs[-1]
    print(f"使用测试结果: {latest_result.name}")
    
    visualize_order_arrivals(latest_result)
    
    print("\n✓ 可视化完成")

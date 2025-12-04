"""
数据准备模块测试脚本
测试OSM路网提取、距离矩阵计算和订单生成
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.data_preparation.order_generator import generate_orders
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_network_extraction():
    """测试路网提取"""
    print("\n" + "="*60)
    print("测试1: OSM路网提取")
    print("="*60)
    
    config = get_config()
    network_config = config.get_network_config()
    output_dir = config.get_data_dir("processed")
    
    try:
        graph, files = extract_osm_network(
            network_config,
            output_dir=output_dir,
            force_download=False
        )
        
        print(f"✓ 路网提取成功")
        print(f"  节点数: {len(graph.nodes())}")
        print(f"  边数: {len(graph.edges())}")
        print(f"  保存文件:")
        for key, path in files.items():
            print(f"    {key}: {path.name}")
        
        return graph, True
        
    except Exception as e:
        print(f"✗ 路网提取失败: {str(e)}")
        logger.exception("路网提取错误")
        return None, False


def test_distance_matrix(graph):
    """测试距离矩阵计算"""
    print("\n" + "="*60)
    print("测试2: 距离矩阵计算")
    print("="*60)
    
    if graph is None:
        print("✗ 跳过测试（路网未加载）")
        return None, False
    
    config = get_config()
    matrix_config = config.get_distance_matrix_config()
    output_dir = config.get_data_dir("processed")
    
    try:
        dist_matrix, time_matrix, mapping = compute_distance_matrices(
            graph,
            matrix_config,
            output_dir=output_dir,
            force_recalculate=False
        )
        
        print(f"✓ 距离矩阵计算成功")
        print(f"  矩阵大小: {dist_matrix.shape}")
        print(f"  节点数量: {len(mapping['node_list'])}")
        print(f"  平均距离: {dist_matrix[dist_matrix < float('inf')].mean():.2f} 米")
        print(f"  平均时间: {time_matrix[time_matrix < float('inf')].mean():.2f} 秒")
        
        return mapping, True
        
    except Exception as e:
        print(f"✗ 距离矩阵计算失败: {str(e)}")
        logger.exception("距离矩阵计算错误")
        return None, False


def test_order_generation(graph, mapping):
    """测试订单生成"""
    print("\n" + "="*60)
    print("测试3: 订单生成")
    print("="*60)
    
    if graph is None or mapping is None:
        print("✗ 跳过测试（依赖模块未完成）")
        return False
    
    config = get_config()
    order_config = config.get_order_generation_config()
    random_seed = config.get_random_seed()
    orders_dir = config.get_data_dir("orders")
    
    try:
        orders, files = generate_orders(
            graph,
            order_config,
            node_list=mapping['node_list'],
            output_dir=orders_dir,
            random_seed=random_seed
        )
        
        print(f"✓ 订单生成成功")
        print(f"  订单数量: {len(orders)}")
        print(f"  保存文件:")
        for key, path in files.items():
            print(f"    {key}: {path.name}")
        
        # 显示第一个订单的详细信息
        if orders:
            first_order = orders[0]
            print(f"\n  示例订单 #{first_order.order_id}:")
            print(f"    到达时间: {first_order.arrival_time:.2f}s ({first_order.arrival_time/3600:.2f}h)")
            print(f"    商家节点: {first_order.merchant_node}")
            print(f"    客户节点: {first_order.customer_node}")
            print(f"    准备时间: {first_order.preparation_time/60:.2f} 分钟")
            print(f"    配送时间窗: {first_order.delivery_window/60:.2f} 分钟")
        
        return True
        
    except Exception as e:
        print(f"✗ 订单生成失败: {str(e)}")
        logger.exception("订单生成错误")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("数据准备模块完整测试")
    print("="*60)
    
    results = {}
    
    # 测试1: 路网提取
    graph, success = test_network_extraction()
    results['network'] = success
    
    # 测试2: 距离矩阵
    mapping, success = test_distance_matrix(graph)
    results['distance_matrix'] = success
    
    # 测试3: 订单生成
    success = test_order_generation(graph, mapping)
    results['order_generation'] = success
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:20s}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！")
        print("Day 1任务完成：")
        print("  - OSM路网已提取")
        print("  - 距离矩阵已预计算")
        print("  - 订单生成器已就绪")
    else:
        print("✗ 部分测试失败，请检查错误日志")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

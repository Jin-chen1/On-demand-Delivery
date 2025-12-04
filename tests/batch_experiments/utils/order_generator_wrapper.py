"""
订单生成器封装 - 用于批量实验
动态调整参数并生成订单数据
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.order_generator import OrderGenerator

logger = logging.getLogger(__name__)


class OrderGeneratorWrapper:
    """订单生成器封装类"""
    
    def __init__(self, graph, node_list: list):
        """
        初始化封装器
        
        Args:
            graph: 路网图对象
            node_list: 可用节点列表
        """
        self.graph = graph
        self.node_list = node_list
        logger.info("订单生成器封装初始化完成")
    
    def generate_orders_for_experiment(
        self, 
        order_config: Dict[str, Any], 
        random_seed: int,
        output_dir: Path = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        为单个实验生成订单
        
        Args:
            order_config: 订单生成配置
            random_seed: 随机种子
            output_dir: 输出目录（如果为None则使用临时目录）
        
        Returns:
            (订单文件路径, 订单统计信息)
        """
        logger.info(f"生成实验订单（订单量: {order_config['total_orders']}, 种子: {random_seed}）")
        
        # 使用临时目录或指定目录
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="orders_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建订单生成器
        generator = OrderGenerator(
            graph=self.graph,
            config=order_config,
            node_list=self.node_list,
            random_seed=random_seed
        )
        
        # 生成订单
        orders = generator.generate_orders()
        
        # 保存订单
        saved_files = generator.save_orders(orders, output_dir)
        
        # 获取统计信息
        stats = generator.get_statistics(orders)
        
        # 返回订单文件路径
        orders_file = saved_files['orders']
        
        logger.info(f"订单文件已生成: {orders_file}")
        logger.info(f"实际生成订单数: {stats['total_orders']}")
        
        return orders_file, stats
    
    def validate_order_config(self, config: Dict[str, Any]) -> bool:
        """
        验证订单配置的有效性
        
        Args:
            config: 订单配置
        
        Returns:
            是否有效
        """
        required_keys = ['total_orders', 'simulation_duration', 'arrival_process']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"订单配置缺少必需字段: {key}")
                return False
        
        # 检查到达率合理性
        if 'rate' in config['arrival_process']:
            rate = config['arrival_process']['rate']
            expected_orders = rate * config['simulation_duration']
            
            if abs(expected_orders - config['total_orders']) > config['total_orders'] * 0.2:
                logger.warning(
                    f"到达率与目标订单数不匹配: "
                    f"预期 {expected_orders:.0f} 订单, 目标 {config['total_orders']} 订单"
                )
        
        return True
    
    def get_order_summary(self, orders_file: Path) -> Dict[str, Any]:
        """
        获取订单文件摘要信息
        
        Args:
            orders_file: 订单CSV文件路径
        
        Returns:
            摘要信息字典
        """
        if not orders_file.exists():
            raise FileNotFoundError(f"订单文件不存在: {orders_file}")
        
        df = pd.read_csv(orders_file)
        
        summary = {
            'total_orders': len(df),
            'first_arrival': df['arrival_time'].min(),
            'last_arrival': df['arrival_time'].max(),
            'time_span': df['arrival_time'].max() - df['arrival_time'].min(),
            'avg_preparation_time': df['preparation_time'].mean(),
            'avg_delivery_window': df['delivery_window'].mean(),
            'unique_merchants': df['merchant_node'].nunique(),
            'unique_customers': df['customer_node'].nunique()
        }
        
        return summary


def create_order_generator_wrapper(graph, node_list: list) -> OrderGeneratorWrapper:
    """
    创建订单生成器封装（便捷函数）
    
    Args:
        graph: 路网图对象
        node_list: 可用节点列表
    
    Returns:
        OrderGeneratorWrapper实例
    """
    return OrderGeneratorWrapper(graph, node_list)

"""
RL模块公共工具函数

避免循环导入，为多个模块提供共享的工具函数
"""

from typing import Dict, Any


def extract_simulation_metrics(sim_env) -> Dict[str, Any]:
    """
    从仿真环境中提取关键指标（公共工具函数）
    
    Args:
        sim_env: 仿真环境实例 (SimulationEnvironment)
    
    Returns:
        包含以下指标的字典:
        - total_orders: 总订单数
        - completed_orders: 已完成订单数
        - timeout_orders: 超时订单数
        - completion_rate: 完成率
        - timeout_rate: 超时率
        - avg_service_time: 平均服务时间（秒）
        - total_distance: 总配送距离（米）
    
    Example:
        >>> metrics = extract_simulation_metrics(env.sim_env)
        >>> print(f"完成率: {metrics['completion_rate']:.1%}")
    """
    if sim_env is None:
        return {
            'total_orders': 0,
            'completed_orders': 0,
            'timeout_orders': 0,
            'completion_rate': 0.0,
            'timeout_rate': 0.0,
            'avg_service_time': 0.0,
            'total_distance': 0.0
        }
    
    # 统计订单状态
    total_orders = len(sim_env.orders)
    completed_orders_list = [
        o for o in sim_env.orders.values() 
        if hasattr(o, 'status') and 'DELIVERED' in str(o.status)
    ]
    completed_orders = len(completed_orders_list)
    timeout_orders = len([
        o for o in sim_env.orders.values() 
        if hasattr(o, 'status') and 'TIMEOUT' in str(o.status)
    ])
    
    # 计算比率
    completion_rate = completed_orders / max(total_orders, 1)
    timeout_rate = timeout_orders / max(total_orders, 1)
    
    # 计算平均服务时间
    avg_service_time = 0.0
    if completed_orders > 0:
        service_times = []
        for o in completed_orders_list:
            if hasattr(o, 'completion_time') and hasattr(o, 'creation_time'):
                service_times.append(o.completion_time - o.creation_time)
        if service_times:
            avg_service_time = sum(service_times) / len(service_times)
    
    # 计算总距离
    total_distance = 0.0
    if hasattr(sim_env, 'couriers'):
        for courier in sim_env.couriers.values():
            if hasattr(courier, 'total_distance'):
                total_distance += courier.total_distance
    
    return {
        'total_orders': total_orders,
        'completed_orders': completed_orders,
        'timeout_orders': timeout_orders,
        'completion_rate': completion_rate,
        'timeout_rate': timeout_rate,
        'avg_service_time': avg_service_time,
        'total_distance': total_distance
    }

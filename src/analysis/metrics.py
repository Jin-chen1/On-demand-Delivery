"""
评估指标计算模块 - Day 5
计算超时率、平均配送时间、骑手利用率等关键指标
用于生成论文 Fig 4（压力测试曲线）的数据
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import csv

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 订单相关指标
    total_orders: int = 0
    completed_orders: int = 0
    timeout_orders: int = 0
    pending_orders: int = 0
    assigned_orders: int = 0
    
    # 时间相关指标
    avg_delivery_time: float = 0.0  # 平均配送时间（秒）
    avg_waiting_time: float = 0.0   # 平均等待时间（秒）
    max_delivery_time: float = 0.0  # 最大配送时间（秒）
    min_delivery_time: float = float('inf')  # 最小配送时间（秒）
    
    # 超时相关
    timeout_rate: float = 0.0  # 超时率（0-1）
    avg_delay: float = 0.0     # 平均延迟时间（秒）
    
    # 骑手相关指标
    total_couriers: int = 0
    avg_utilization: float = 0.0  # 平均利用率（0-1）
    total_distance: float = 0.0   # 总行驶距离（米）
    avg_distance_per_courier: float = 0.0  # 每个骑手平均距离
    orders_per_km: float = 0.0    # 单位里程配送订单数
    
    # 调度效率
    avg_orders_per_courier: float = 0.0
    
    # 详细数据（用于绘图）
    delivery_times: List[float] = field(default_factory=list)
    waiting_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_orders': self.total_orders,
            'completed_orders': self.completed_orders,
            'timeout_orders': self.timeout_orders,
            'pending_orders': self.pending_orders,
            'assigned_orders': self.assigned_orders,
            'avg_delivery_time': self.avg_delivery_time,
            'avg_waiting_time': self.avg_waiting_time,
            'max_delivery_time': self.max_delivery_time,
            'min_delivery_time': self.min_delivery_time if self.min_delivery_time != float('inf') else 0,
            'timeout_rate': self.timeout_rate,
            'avg_delay': self.avg_delay,
            'total_couriers': self.total_couriers,
            'avg_utilization': self.avg_utilization,
            'total_distance': self.total_distance,
            'avg_distance_per_courier': self.avg_distance_per_courier,
            'orders_per_km': self.orders_per_km,
            'avg_orders_per_courier': self.avg_orders_per_courier
        }


class MetricsCalculator:
    """
    评估指标计算器
    从仿真环境或日志中提取数据，计算论文所需的关键指标
    """
    
    def __init__(self):
        """初始化计算器"""
        self.logger = logging.getLogger(__name__)
    
    def calculate_from_environment(self, env) -> PerformanceMetrics:
        """
        从仿真环境对象中计算指标
        
        Args:
            env: SimulationEnvironment 实例
        
        Returns:
            PerformanceMetrics 对象
        """
        self.logger.info("开始从仿真环境计算指标...")
        
        metrics = PerformanceMetrics()
        
        # 订单统计
        metrics.total_orders = len(env.orders)
        metrics.pending_orders = len(env.pending_orders)
        metrics.assigned_orders = len(env.assigned_orders)
        
        # 分析已完成和超时订单
        completed_times = []
        waiting_times = []
        timeout_count = 0
        total_delay = 0.0
        
        for order_id, order in env.orders.items():
            if order.status.value == 'delivered':
                metrics.completed_orders += 1
                
                # 配送时间 = 实际送达时间 - 订单到达时间
                if order.delivery_complete_time is not None:
                    delivery_time = order.delivery_complete_time - order.arrival_time
                    completed_times.append(delivery_time)
                    
                    # 等待时间 = 分配时间 - 到达时间
                    if order.assignment_time is not None:
                        waiting_time = order.assignment_time - order.arrival_time
                        waiting_times.append(waiting_time)
                    
                    # 检查是否超时
                    if order.delivery_complete_time > order.latest_delivery_time:
                        timeout_count += 1
                        delay = order.delivery_complete_time - order.latest_delivery_time
                        total_delay += delay
            
            elif order.status.value == 'timeout':
                timeout_count += 1
                metrics.timeout_orders += 1
        
        # 计算时间指标
        if completed_times:
            metrics.avg_delivery_time = np.mean(completed_times)
            metrics.max_delivery_time = np.max(completed_times)
            metrics.min_delivery_time = np.min(completed_times)
            metrics.delivery_times = completed_times
        
        if waiting_times:
            metrics.avg_waiting_time = np.mean(waiting_times)
            metrics.waiting_times = waiting_times
        
        # 超时率
        if metrics.total_orders > 0:
            metrics.timeout_rate = timeout_count / metrics.total_orders
        
        # 平均延迟
        if timeout_count > 0:
            metrics.avg_delay = total_delay / timeout_count
        
        # 骑手统计
        metrics.total_couriers = len(env.couriers)
        
        total_utilization = 0.0
        total_distance = 0.0
        total_completed_by_couriers = 0
        
        for courier_id, courier in env.couriers.items():
            total_utilization += courier.get_utilization()
            total_distance += courier.total_distance
            total_completed_by_couriers += courier.completed_orders
        
        if metrics.total_couriers > 0:
            metrics.avg_utilization = total_utilization / metrics.total_couriers
            metrics.avg_distance_per_courier = total_distance / metrics.total_couriers
            metrics.avg_orders_per_courier = total_completed_by_couriers / metrics.total_couriers
        
        metrics.total_distance = total_distance
        
        # 单位里程配送订单数（公里）
        if total_distance > 0:
            metrics.orders_per_km = (metrics.completed_orders / (total_distance / 1000.0))
        
        self.logger.info("指标计算完成")
        self._log_metrics_summary(metrics)
        
        return metrics
    
    def calculate_from_events(
        self, 
        events: List[Any], 
        orders: Dict[int, Any],
        couriers: Dict[int, Any]
    ) -> PerformanceMetrics:
        """
        从事件日志中计算指标（适用于已保存的仿真结果）
        
        Args:
            events: SimulationEvent 列表
            orders: 订单字典
            couriers: 骑手字典
        
        Returns:
            PerformanceMetrics 对象
        """
        self.logger.info("开始从事件日志计算指标...")
        
        metrics = PerformanceMetrics()
        
        # 基本统计
        metrics.total_orders = len(orders)
        metrics.total_couriers = len(couriers)
        
        # 从订单数据计算
        completed_times = []
        waiting_times = []
        timeout_count = 0
        
        for order in orders.values():
            if order.status.value == 'delivered':
                metrics.completed_orders += 1
                if order.delivery_complete_time:
                    delivery_time = order.delivery_complete_time - order.arrival_time
                    completed_times.append(delivery_time)
                    
                    if order.assignment_time:
                        waiting_time = order.assignment_time - order.arrival_time
                        waiting_times.append(waiting_time)
                    
                    if order.delivery_complete_time > order.latest_delivery_time:
                        timeout_count += 1
            
            elif order.status.value == 'pending':
                metrics.pending_orders += 1
            elif order.status.value == 'assigned':
                metrics.assigned_orders += 1
        
        # 计算时间指标
        if completed_times:
            metrics.avg_delivery_time = np.mean(completed_times)
            metrics.max_delivery_time = np.max(completed_times)
            metrics.min_delivery_time = np.min(completed_times)
            metrics.delivery_times = completed_times
        
        if waiting_times:
            metrics.avg_waiting_time = np.mean(waiting_times)
            metrics.waiting_times = waiting_times
        
        # 超时率
        metrics.timeout_orders = timeout_count
        if metrics.total_orders > 0:
            metrics.timeout_rate = timeout_count / metrics.total_orders
        
        # 骑手统计
        total_utilization = 0.0
        total_distance = 0.0
        
        for courier in couriers.values():
            total_utilization += courier.get_utilization()
            total_distance += courier.total_distance
        
        if metrics.total_couriers > 0:
            metrics.avg_utilization = total_utilization / metrics.total_couriers
            metrics.avg_distance_per_courier = total_distance / metrics.total_couriers
            metrics.avg_orders_per_courier = metrics.completed_orders / metrics.total_couriers
        
        metrics.total_distance = total_distance
        
        if total_distance > 0:
            metrics.orders_per_km = (metrics.completed_orders / (total_distance / 1000.0))
        
        self.logger.info("指标计算完成")
        self._log_metrics_summary(metrics)
        
        return metrics
    
    def _log_metrics_summary(self, metrics: PerformanceMetrics) -> None:
        """记录指标摘要到日志"""
        self.logger.info("=" * 60)
        self.logger.info("性能指标摘要")
        self.logger.info("=" * 60)
        self.logger.info(f"订单总数: {metrics.total_orders}")
        self.logger.info(f"  已完成: {metrics.completed_orders} ({metrics.completed_orders/max(metrics.total_orders, 1)*100:.1f}%)")
        self.logger.info(f"  超时: {metrics.timeout_orders} (超时率: {metrics.timeout_rate*100:.2f}%)")
        self.logger.info(f"  待分配: {metrics.pending_orders}")
        self.logger.info(f"  已分配: {metrics.assigned_orders}")
        self.logger.info(f"平均配送时间: {metrics.avg_delivery_time:.1f}秒 ({metrics.avg_delivery_time/60:.1f}分钟)")
        self.logger.info(f"平均等待时间: {metrics.avg_waiting_time:.1f}秒 ({metrics.avg_waiting_time/60:.1f}分钟)")
        self.logger.info(f"骑手总数: {metrics.total_couriers}")
        self.logger.info(f"  平均利用率: {metrics.avg_utilization*100:.1f}%")
        self.logger.info(f"  总行驶距离: {metrics.total_distance/1000:.2f}公里")
        self.logger.info(f"  单位里程配送: {metrics.orders_per_km:.2f}单/公里")
        self.logger.info("=" * 60)
    
    def save_metrics(self, metrics: PerformanceMetrics, output_path: Path) -> None:
        """
        保存指标到文件
        
        Args:
            metrics: 性能指标对象
            output_path: 输出路径（支持 .json 或 .csv）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
            self.logger.info(f"指标已保存到 JSON: {output_path}")
        
        elif output_path.suffix == '.csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.to_dict().keys())
                writer.writeheader()
                writer.writerow(metrics.to_dict())
            self.logger.info(f"指标已保存到 CSV: {output_path}")
        
        else:
            raise ValueError(f"不支持的文件格式: {output_path.suffix}，仅支持 .json 或 .csv")
    
    def compare_metrics(
        self, 
        baseline_metrics: PerformanceMetrics, 
        improved_metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """
        对比两组指标，计算改进百分比
        用于生成论文中的对比表格
        
        Args:
            baseline_metrics: 基线方法的指标
            improved_metrics: 改进方法的指标
        
        Returns:
            改进百分比字典
        """
        improvements = {}
        
        # 超时率改进（越低越好）
        if baseline_metrics.timeout_rate > 0:
            improvements['timeout_rate'] = (
                (baseline_metrics.timeout_rate - improved_metrics.timeout_rate) 
                / baseline_metrics.timeout_rate * 100
            )
        
        # 平均配送时间改进（越低越好）
        if baseline_metrics.avg_delivery_time > 0:
            improvements['avg_delivery_time'] = (
                (baseline_metrics.avg_delivery_time - improved_metrics.avg_delivery_time)
                / baseline_metrics.avg_delivery_time * 100
            )
        
        # 骑手利用率改进（越高越好）
        if baseline_metrics.avg_utilization > 0:
            improvements['avg_utilization'] = (
                (improved_metrics.avg_utilization - baseline_metrics.avg_utilization)
                / baseline_metrics.avg_utilization * 100
            )
        
        # 单位里程配送订单数改进（越高越好）
        if baseline_metrics.orders_per_km > 0:
            improvements['orders_per_km'] = (
                (improved_metrics.orders_per_km - baseline_metrics.orders_per_km)
                / baseline_metrics.orders_per_km * 100
            )
        
        # 完成率改进（越高越好）
        baseline_completion = baseline_metrics.completed_orders / max(baseline_metrics.total_orders, 1)
        improved_completion = improved_metrics.completed_orders / max(improved_metrics.total_orders, 1)
        if baseline_completion > 0:
            improvements['completion_rate'] = (
                (improved_completion - baseline_completion) / baseline_completion * 100
            )
        
        self.logger.info("\n对比改进:")
        for metric, improvement in improvements.items():
            self.logger.info(f"  {metric}: {improvement:+.2f}%")
        
        return improvements

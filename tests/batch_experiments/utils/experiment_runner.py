"""
实验执行引擎 - Day 8核心组件
负责运行批量实验，管理进度，处理错误
"""

import sys
from pathlib import Path
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.osm_network import extract_osm_network
from src.data_preparation.distance_matrix import compute_distance_matrices
from src.simulation import SimulationEnvironment
from .experiment_task import ExperimentTask
from .order_generator_wrapper import OrderGeneratorWrapper

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验执行引擎"""
    
    def __init__(self, 
                 tasks: List[ExperimentTask],
                 output_dir: Path,
                 graph,
                 distance_matrix,
                 time_matrix,
                 node_mapping,
                 execution_config: Dict[str, Any]):
        """
        初始化实验执行引擎
        
        Args:
            tasks: 实验任务列表
            output_dir: 输出目录
            graph: 路网图
            distance_matrix: 距离矩阵
            time_matrix: 时间矩阵
            node_mapping: 节点映射
            execution_config: 执行配置
        """
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.graph = graph
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.node_mapping = node_mapping
        
        # 执行配置
        self.max_retries = execution_config.get('max_retries', 2)
        self.continue_on_error = execution_config.get('continue_on_error', True)
        self.save_progress_interval = execution_config.get('save_progress_interval', 5)
        self.enable_checkpoint = execution_config.get('enable_checkpoint', True)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_logs_dir = self.output_dir / "detailed_logs"
        self.detailed_logs_dir.mkdir(exist_ok=True)
        self.orders_dir = self.output_dir / "generated_orders"
        self.orders_dir.mkdir(exist_ok=True)
        
        # 进度文件
        self.progress_file = self.output_dir / "progress.json"
        
        # 订单生成器封装
        self.order_wrapper = OrderGeneratorWrapper(
            graph=self.graph,
            node_list=self.node_mapping['node_list']
        )
        
        # 统计信息
        self.stats = {
            'total_experiments': len(tasks),
            'completed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'total_duration': 0
        }
        
        logger.info(f"实验执行引擎初始化完成，共 {len(tasks)} 个任务")
    
    def run_all_experiments(self) -> List[Dict[str, Any]]:
        """
        运行所有实验
        
        Returns:
            所有实验结果列表
        """
        logger.info("="*80)
        logger.info("开始批量实验执行")
        logger.info("="*80)
        
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()
        
        results = []
        
        # 加载已完成的任务（断点续传）
        completed_task_ids = self._load_completed_tasks()
        
        for i, task in enumerate(self.tasks, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"实验进度: {i}/{len(self.tasks)} ({i/len(self.tasks)*100:.1f}%)")
            logger.info(f"任务ID: {task.task_id}")
            logger.info(f"配置: {task.num_orders}订单, {task.num_couriers}骑手, {task.dispatcher_type}调度器, 种子{task.random_seed}")
            logger.info(f"{'='*80}")
            
            # 检查是否已完成（断点续传）
            if self.enable_checkpoint and task.task_id in completed_task_ids:
                logger.info(f"⏭️  任务已完成，跳过: {task.task_id}")
                self.stats['skipped'] += 1
                
                # 加载已有结果
                result = self._load_task_result(task.task_id)
                if result:
                    results.append(result)
                
                continue
            
            # 运行单个实验（带重试）
            result = self._run_single_experiment_with_retry(task)
            
            if result:
                results.append(result)
                self.stats['completed'] += 1
            else:
                self.stats['failed'] += 1
                
                if not self.continue_on_error:
                    logger.error("实验失败且配置为不继续，停止执行")
                    break
            
            # 定期保存进度
            if i % self.save_progress_interval == 0:
                self._save_progress(results)
            
            # 显示预估剩余时间
            self._log_time_estimate(i, len(self.tasks), start_time)
        
        # 最终保存
        self._save_progress(results)
        
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['total_duration'] = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("批量实验执行完成")
        logger.info("="*80)
        self._log_final_summary()
        
        return results
    
    def _run_single_experiment_with_retry(self, task: ExperimentTask) -> Optional[Dict[str, Any]]:
        """
        运行单个实验（带重试机制）
        
        Args:
            task: 实验任务
        
        Returns:
            实验结果字典，失败返回None
        """
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.warning(f"重试实验 (第 {attempt}/{self.max_retries} 次): {task.task_id}")
            
            try:
                result = self._run_single_experiment(task)
                return result
                
            except Exception as e:
                error_msg = f"实验失败: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                task.error_message = error_msg
                
                if attempt >= self.max_retries:
                    logger.error(f"实验最终失败（已重试{self.max_retries}次）: {task.task_id}")
                    self._save_failed_task_info(task)
                    return None
                
                # 等待后重试
                time.sleep(2)
        
        return None
    
    def _run_single_experiment(self, task: ExperimentTask) -> Dict[str, Any]:
        """
        运行单个实验
        
        Args:
            task: 实验任务
        
        Returns:
            实验结果字典
        """
        task.status = "running"
        task.start_time = datetime.now().isoformat()
        exp_start = time.time()
        
        logger.info(f"步骤1: 生成订单数据...")
        
        # 生成订单
        order_output_dir = self.orders_dir / task.task_id
        orders_file, order_stats = self.order_wrapper.generate_orders_for_experiment(
            order_config=task.order_generation_config,
            random_seed=task.random_seed,
            output_dir=order_output_dir
        )
        
        logger.info(f"✓ 订单生成完成: {order_stats['total_orders']}个订单")
        
        # 运行仿真
        logger.info(f"步骤2: 运行仿真...")
        sim_result = self._run_simulation(task, orders_file)
        
        # 记录结果
        task.end_time = datetime.now().isoformat()
        task.duration_seconds = time.time() - exp_start
        task.status = "completed"
        
        # 整合结果
        result = {
            'task_id': task.task_id,
            'experiment_index': task.experiment_index,
            'num_orders': task.num_orders,
            'num_couriers': task.num_couriers,
            'dispatcher_type': task.dispatcher_type,
            'random_seed': task.random_seed,
            'start_time': task.start_time,
            'end_time': task.end_time,
            'duration_seconds': task.duration_seconds,
            **sim_result  # 仿真结果
        }
        
        task.result = result
        
        logger.info(f"✓ 实验完成，耗时 {task.duration_seconds:.1f}秒")
        logger.info(f"  完成率: {sim_result.get('completion_rate', 0)*100:.1f}%")
        logger.info(f"  超时率: {sim_result.get('timeout_rate', 0)*100:.1f}%")
        
        return result
    
    def _run_simulation(self, task: ExperimentTask, orders_file: Path) -> Dict[str, Any]:
        """
        运行仿真并收集结果
        
        Args:
            task: 实验任务
            orders_file: 订单文件路径
        
        Returns:
            仿真结果字典
        """
        # 创建仿真环境
        sim_env = SimulationEnvironment(
            graph=self.graph,
            distance_matrix=self.distance_matrix,
            time_matrix=self.time_matrix,
            node_mapping=self.node_mapping,
            config=task.simulation_config
        )
        
        # 加载订单
        sim_env.load_orders_from_csv(orders_file)
        
        # 初始化骑手
        sim_env.initialize_couriers(task.num_couriers, task.courier_config)
        
        # 运行仿真
        sim_env.run(until=task.simulation_config['simulation_duration'])
        
        # 收集统计数据
        result = self._collect_simulation_results(sim_env, task.dispatcher_type)
        
        return result
    
    def _collect_simulation_results(self, sim_env, dispatcher_type: str) -> Dict[str, Any]:
        """收集仿真结果"""
        import numpy as np
        
        # 基础事件统计
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        assigned_events = [e for e in sim_env.events if e.event_type == 'order_assigned']
        delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
        
        results = {
            'total_orders': len(arrival_events),
            'assigned_orders': len(assigned_events),
            'completed_orders': len(delivery_events),
            'pending_orders': len(sim_env.pending_orders),
            'completion_rate': len(delivery_events) / len(arrival_events) if arrival_events else 0
        }
        
        # 订单性能指标
        if delivery_events:
            completed_order_ids = [e.entity_id for e in delivery_events]
            
            # 超时统计
            timeout_count = sum(
                1 for oid in completed_order_ids
                if sim_env.orders[oid].is_timeout(sim_env.env.now)
            )
            results['timeout_count'] = timeout_count
            results['timeout_rate'] = timeout_count / len(delivery_events)
            
            # 服务时长
            service_times = [
                sim_env.orders[oid].get_total_service_time()
                for oid in completed_order_ids
                if sim_env.orders[oid].get_total_service_time() is not None
            ]
            
            if service_times:
                results['avg_service_time'] = np.mean(service_times)
                results['median_service_time'] = np.median(service_times)
                results['std_service_time'] = np.std(service_times)
            else:
                results['avg_service_time'] = 0
                results['median_service_time'] = 0
                results['std_service_time'] = 0
            
            # 骑手统计
            total_distance = sum(c.total_distance for c in sim_env.couriers.values())
            results['total_distance'] = total_distance
            results['avg_distance_per_order'] = total_distance / len(delivery_events)
            
            total_utilization = sum(c.get_utilization() for c in sim_env.couriers.values())
            results['avg_courier_utilization'] = total_utilization / len(sim_env.couriers)
        else:
            results.update({
                'timeout_count': 0,
                'timeout_rate': 0,
                'avg_service_time': 0,
                'median_service_time': 0,
                'std_service_time': 0,
                'total_distance': 0,
                'avg_distance_per_order': 0,
                'avg_courier_utilization': 0
            })
        
        # 调度器特定统计
        if hasattr(sim_env.dispatcher, 'get_statistics'):
            dispatcher_stats = sim_env.dispatcher.get_statistics()
            results['dispatch_count'] = dispatcher_stats.get('dispatch_count', 0)
            results['solve_success_count'] = dispatcher_stats.get('solve_success_count', 0)
            results['solve_failure_count'] = dispatcher_stats.get('solve_failure_count', 0)
            results['avg_solve_time'] = dispatcher_stats.get('average_solve_time', 0)
            results['solve_success_rate'] = dispatcher_stats.get('solve_success_rate', 0)
        else:
            results.update({
                'dispatch_count': 0,
                'solve_success_count': 0,
                'solve_failure_count': 0,
                'avg_solve_time': 0,
                'solve_success_rate': 0
            })
        
        return results
    
    def _save_progress(self, results: List[Dict[str, Any]]) -> None:
        """保存进度"""
        progress_data = {
            'stats': self.stats,
            'completed_task_ids': [r['task_id'] for r in results],
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"进度已保存: {self.progress_file}")
    
    def _load_completed_tasks(self) -> set:
        """加载已完成的任务ID"""
        if not self.progress_file.exists():
            return set()
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            completed = set(progress_data.get('completed_task_ids', []))
            logger.info(f"检测到 {len(completed)} 个已完成任务（断点续传）")
            return completed
            
        except Exception as e:
            logger.warning(f"加载进度文件失败: {str(e)}")
            return set()
    
    def _load_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """加载任务结果"""
        # 从已保存的结果文件中加载
        # 这里简化处理，实际可从CSV或JSON加载
        return None
    
    def _save_failed_task_info(self, task: ExperimentTask) -> None:
        """保存失败任务信息"""
        failed_file = self.output_dir / "failed_tasks.json"
        
        failed_tasks = []
        if failed_file.exists():
            with open(failed_file, 'r', encoding='utf-8') as f:
                failed_tasks = json.load(f)
        
        failed_tasks.append({
            'task_id': task.task_id,
            'error_message': task.error_message,
            'time': datetime.now().isoformat()
        })
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, indent=2, ensure_ascii=False)
    
    def _log_time_estimate(self, current: int, total: int, start_time: float) -> None:
        """记录时间预估"""
        elapsed = time.time() - start_time
        avg_time_per_exp = elapsed / current
        remaining = total - current
        estimated_remaining = avg_time_per_exp * remaining
        
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        
        logger.info(f"\n⏱️  时间统计:")
        logger.info(f"  已用时间: {elapsed/3600:.2f}小时")
        logger.info(f"  平均每个实验: {avg_time_per_exp/60:.1f}分钟")
        logger.info(f"  预计剩余: {estimated_remaining/3600:.2f}小时")
        logger.info(f"  预计完成: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _log_final_summary(self) -> None:
        """记录最终摘要"""
        logger.info(f"\n最终统计:")
        logger.info(f"  总实验数: {self.stats['total_experiments']}")
        logger.info(f"  完成数: {self.stats['completed']}")
        logger.info(f"  失败数: {self.stats['failed']}")
        logger.info(f"  跳过数: {self.stats['skipped']}")
        logger.info(f"  总耗时: {self.stats['total_duration']/3600:.2f}小时")
        
        if self.stats['completed'] > 0:
            success_rate = self.stats['completed'] / (self.stats['completed'] + self.stats['failed']) * 100
            logger.info(f"  成功率: {success_rate:.1f}%")

"""
实验任务数据结构和配置管理器
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import logging
from itertools import product

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTask:
    """单个实验任务"""
    # 任务标识
    task_id: str
    experiment_index: int  # 实验序号（1-72）
    
    # 实验参数
    num_orders: int
    num_couriers: int
    dispatcher_type: str
    dispatcher_config: Dict[str, Any]
    random_seed: int
    
    # 仿真配置
    simulation_config: Dict[str, Any]
    courier_config: Dict[str, Any]
    order_generation_config: Dict[str, Any]
    
    # 状态信息
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_identifier(self) -> str:
        """获取简短标识符"""
        return f"orders{self.num_orders}_couriers{self.num_couriers}_{self.dispatcher_type}_seed{self.random_seed}"


class ExperimentConfigManager:
    """实验配置管理器"""
    
    def __init__(self, config_file: Path):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径（YAML格式）
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        logger.info(f"加载实验配置: {config_file}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def generate_tasks(self) -> List[ExperimentTask]:
        """
        生成所有实验任务
        
        Returns:
            实验任务列表
        """
        logger.info("生成实验任务列表...")
        
        dimensions = self.config['dimensions']
        simulation_base = self.config['simulation']
        courier_config = self.config['courier']
        
        # 提取维度参数
        num_orders_list = dimensions['num_orders']
        num_couriers_list = dimensions['num_couriers']
        dispatchers = dimensions['dispatchers']
        random_seeds = dimensions['random_seeds']
        
        # 生成所有组合
        tasks = []
        task_index = 1
        
        for num_orders, num_couriers, dispatcher, seed in product(
            num_orders_list, num_couriers_list, dispatchers, random_seeds
        ):
            # 生成任务ID
            task_id = f"{task_index:03d}_orders{num_orders}_couriers{num_couriers}_{dispatcher['type']}_seed{seed}"
            
            # 动态调整订单生成配置
            order_gen_config = self._adjust_order_config(
                simulation_base['order_generation'],
                num_orders,
                simulation_base['duration']
            )
            
            # 创建仿真配置
            sim_config = {
                'simulation_duration': simulation_base['duration'],
                'dispatch_interval': simulation_base['dispatch_interval'],
                'dispatcher_type': dispatcher['type'],
                'dispatcher_config': dispatcher['config']
            }
            
            # 创建任务对象
            task = ExperimentTask(
                task_id=task_id,
                experiment_index=task_index,
                num_orders=num_orders,
                num_couriers=num_couriers,
                dispatcher_type=dispatcher['type'],
                dispatcher_config=dispatcher['config'],
                random_seed=seed,
                simulation_config=sim_config,
                courier_config=courier_config,
                order_generation_config=order_gen_config
            )
            
            tasks.append(task)
            task_index += 1
        
        logger.info(f"生成了 {len(tasks)} 个实验任务")
        self._log_task_summary(tasks)
        
        return tasks
    
    def _adjust_order_config(self, base_config: Dict[str, Any], 
                            num_orders: int, 
                            duration: int) -> Dict[str, Any]:
        """
        根据订单量动态调整订单生成配置
        
        Args:
            base_config: 基础配置
            num_orders: 目标订单数
            duration: 仿真时长（秒）
        
        Returns:
            调整后的配置
        """
        config = base_config.copy()
        
        # 计算到达率（订单/秒）
        arrival_rate = num_orders / duration
        
        # 更新配置
        config['total_orders'] = num_orders
        config['simulation_duration'] = duration
        config['arrival_process']['rate'] = arrival_rate
        
        return config
    
    def _log_task_summary(self, tasks: List[ExperimentTask]) -> None:
        """记录任务摘要"""
        # 按调度器统计
        dispatcher_counts = {}
        for task in tasks:
            dispatcher_counts[task.dispatcher_type] = dispatcher_counts.get(task.dispatcher_type, 0) + 1
        
        logger.info("实验任务摘要:")
        logger.info(f"  总任务数: {len(tasks)}")
        logger.info(f"  订单量维度: {sorted(set(t.num_orders for t in tasks))}")
        logger.info(f"  骑手数维度: {sorted(set(t.num_couriers for t in tasks))}")
        logger.info(f"  调度器分布: {dispatcher_counts}")
        logger.info(f"  重复次数: {len(set(t.random_seed for t in tasks))}")
    
    def get_output_dir(self, timestamp: str) -> Path:
        """
        获取输出目录路径
        
        Args:
            timestamp: 时间戳字符串
        
        Returns:
            输出目录路径
        """
        base_dir = Path(self.config['experiment']['output_base_dir'])
        output_dir = base_dir / timestamp
        return output_dir
    
    def get_execution_config(self) -> Dict[str, Any]:
        """获取执行配置"""
        return self.config.get('execution', {})
    
    def get_results_config(self) -> Dict[str, Any]:
        """获取结果配置"""
        return self.config.get('results', {})

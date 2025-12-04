"""
Day 27: 鲁棒性实验脚本
测试不同算法在极端场景下的表现

鲁棒性测试场景：
1. 爆单场景 - 订单量突然+50%
2. 暴雨场景 - 骑手速度-30%
3. 运力不足 - 骑手数量-30%
4. 商家延迟 - 备餐时间+50%
5. 组合压力 - 多重压力叠加
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# 上海路网和订单数据路径
SHANGHAI_DATA_DIR = project_root / "data" / "processed" / "shanghai"
SHANGHAI_ROAD_NETWORK = SHANGHAI_DATA_DIR / "road_network.graphml"
SHANGHAI_DISTANCE_MATRIX = SHANGHAI_DATA_DIR / "distance_matrix.npy"
SHANGHAI_TIME_MATRIX = SHANGHAI_DATA_DIR / "time_matrix.npy"
SHANGHAI_NODE_MAPPING = SHANGHAI_DATA_DIR / "node_id_mapping.json"
SHANGHAI_ORDERS_FILE = project_root / "data" / "orders" / "uniform_grid_100.csv"  # 均匀网格采样数据

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustnessExperiment:
    """鲁棒性实验类"""
    
    # 算法列表
    ALGORITHMS = ['OR-Tools', 'ALNS', 'Hybrid-HRL']
    
    def __init__(self, config_path: str = None):
        """
        初始化鲁棒性实验
        
        Args:
            config_path: 场景配置文件路径
        """
        self.config_path = config_path or str(project_root / "config" / "scenarios.yaml")
        self.load_config()
        
        # 实验结果存储
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # 输出目录
        self.output_dir = project_root / "outputs" / "robustness_experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_config(self):
        """加载场景配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.robustness_configs = self.config.get('robustness_scenarios', {})
        self.weather_configs = self.config.get('weather_scenarios', {})
        self.stress_configs = self.config.get('stress_test_scenarios', {})
        
        logger.info(f"加载鲁棒性场景配置，共{len(self.robustness_configs)}个场景")
        
        # 加载上海路网数据
        self._load_shanghai_data()
    
    def _load_shanghai_data(self):
        """加载上海路网和相关数据"""
        logger.info("加载上海路网数据...")
        
        # 加载路网图
        self.graph = nx.read_graphml(str(SHANGHAI_ROAD_NETWORK))
        self.graph = nx.relabel_nodes(self.graph, {n: int(n) for n in self.graph.nodes()})
        logger.info(f"  路网节点: {self.graph.number_of_nodes()}")
        
        # 加载距离矩阵和时间矩阵
        self.distance_matrix = np.load(str(SHANGHAI_DISTANCE_MATRIX))
        self.time_matrix = np.load(str(SHANGHAI_TIME_MATRIX))
        logger.info(f"  距离矩阵: {self.distance_matrix.shape}")
        
        # 加载节点映射
        with open(str(SHANGHAI_NODE_MAPPING), 'r') as f:
            self.node_mapping = json.load(f)
        
        # 如果没有 node_list，从 node_to_idx 生成（关键修复）
        # 同时将节点ID转为整数以匹配graph节点类型
        if 'node_list' not in self.node_mapping:
            self.node_mapping['node_list'] = [int(n) for n in self.node_mapping['node_to_idx'].keys()]
        logger.info(f"  节点映射: {len(self.node_mapping['node_list'])} 个节点")
        
        # 订单文件路径
        self.orders_file = SHANGHAI_ORDERS_FILE
        logger.info(f"  订单文件: {self.orders_file}")
        
        # RL模型路径 - 使用最新训练的模型
        self.rl_model_path = self._find_latest_rl_model()
        logger.info(f"  RL模型: {self.rl_model_path}")
    
    def _find_latest_rl_model(self) -> str:
        """查找最新的RL模型文件"""
        models_dir = project_root / "outputs" / "rl_training" / "models"
        if not models_dir.exists():
            logger.warning(f"模型目录不存在: {models_dir}")
            return ""
        
        # 查找所有模型文件（支持两种命名：final_model.zip 和 final_curriculum_model.zip）
        model_files = list(models_dir.glob("*/final_model.zip"))
        model_files.extend(models_dir.glob("*/final_curriculum_model.zip"))
        
        if not model_files:
            logger.warning("未找到训练好的RL模型")
            return ""
        
        # 按修改时间排序，返回最新的
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _get_rl_model_path(self) -> str:
        """获取RL模型路径"""
        return getattr(self, 'rl_model_path', '')
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """获取场景配置"""
        # 先检查鲁棒性场景
        if scenario_name in self.robustness_configs:
            return self._build_robustness_config(scenario_name)
        # 检查天气场景
        elif scenario_name in self.weather_configs:
            return self._build_weather_config(scenario_name)
        # 检查压力测试场景
        elif scenario_name in self.stress_configs:
            return self.stress_configs[scenario_name].copy()
        else:
            raise ValueError(f"未知场景: {scenario_name}")
    
    def _build_robustness_config(self, scenario_name: str) -> Dict[str, Any]:
        """构建鲁棒性测试配置"""
        config = self.robustness_configs[scenario_name].copy()
        
        # 获取基础场景配置
        base_name = config.get('base_scenario', 'medium_load')
        if base_name in self.stress_configs:
            base_config = self.stress_configs[base_name].copy()
            base_config.update(config)
            return base_config
        
        return config
    
    def _build_weather_config(self, scenario_name: str) -> Dict[str, Any]:
        """构建天气场景配置"""
        weather = self.weather_configs[scenario_name].copy()
        
        # 合并基础配置
        base_config = {
            'simulation_duration': 7200,
            'total_orders': 1000,
            'num_couriers': 20,
            'scenario_name': scenario_name
        }
        base_config.update(weather)
        
        return base_config
    
    def _adjust_order_arrival_times(self, sim_env, simulation_duration: float):
        """
        调整订单到达时间到仿真时间范围内
        
        如果订单的原始到达时间超出仿真时长，则将所有订单的到达时间
        线性映射到[0, simulation_duration * 0.7]范围内
        """
        if not sim_env.orders:
            return
        
        # 收集所有订单的到达时间
        arrival_times = [order.arrival_time for order in sim_env.orders.values()]
        min_arrival = min(arrival_times)
        max_arrival = max(arrival_times)
        
        # 检查是否需要调整
        needs_adjustment = min_arrival > simulation_duration * 0.1 or max_arrival > simulation_duration
        
        if not needs_adjustment:
            logger.info(f"订单到达时间在仿真范围内，无需调整 (范围: {min_arrival:.0f}s - {max_arrival:.0f}s)")
            return
        
        logger.warning(f"订单到达时间超出仿真范围! 原始范围: {min_arrival:.0f}s - {max_arrival:.0f}s, 仿真时长: {simulation_duration}s")
        
        # 目标时间范围：[0, simulation_duration * 0.7]，留30%时间给配送
        target_start = 0
        target_end = simulation_duration * 0.7
        
        # 计算时间偏移和缩放
        original_range = max_arrival - min_arrival if max_arrival > min_arrival else 1.0
        target_range = target_end - target_start
        scale_factor = target_range / original_range
        
        # 调整每个订单的时间
        for order in sim_env.orders.values():
            old_arrival = order.arrival_time
            new_arrival = target_start + (old_arrival - min_arrival) * scale_factor
            time_shift = new_arrival - old_arrival
            
            # 更新所有时间字段
            order.arrival_time = new_arrival
            order.earliest_pickup_time += time_shift
            order.latest_delivery_time += time_shift
            
            # 更新deadline（如果存在）
            if hasattr(order, 'deadline'):
                order.deadline += time_shift
        
        # 更新到达时间统计
        new_arrival_times = [order.arrival_time for order in sim_env.orders.values()]
        logger.info(f"订单到达时间已调整: {min(new_arrival_times):.0f}s - {max(new_arrival_times):.0f}s")
    
    def run_scenario_comparison(self, scenario_name: str, 
                                algorithms: List[str] = None,
                                num_episodes: int = 5) -> Dict[str, Any]:
        """
        在指定场景下对比所有算法
        
        Args:
            scenario_name: 场景名称
            algorithms: 算法列表
            num_episodes: episode数
            
        Returns:
            对比结果
        """
        algorithms = algorithms or self.ALGORITHMS
        config = self.get_scenario_config(scenario_name)
        
        logger.info(f"=" * 60)
        logger.info(f"鲁棒性测试场景: {scenario_name}")
        logger.info(f"配置: {json.dumps(config, indent=2, default=str)}")
        logger.info(f"=" * 60)
        
        scenario_results = {}
        
        for algo in algorithms:
            logger.info(f"\n运行算法: {algo}")
            algo_results = []
            
            for episode in range(num_episodes):
                np.random.seed(42 + episode)
                result = self._run_algorithm_episode(algo, config, episode)
                algo_results.append(result)
            
            # 汇总结果
            summary = self._summarize_algorithm_results(algo_results)
            scenario_results[algo] = summary
            
            logger.info(f"  {algo} - 超时率: {summary['mean_timeout_rate']:.2%} ± {summary['std_timeout_rate']:.2%}")
        
        self.results[scenario_name] = {
            'config': config,
            'algorithm_results': scenario_results
        }
        
        return scenario_results
    
    def _run_algorithm_episode(self, algorithm: str, 
                               config: Dict[str, Any], 
                               episode_idx: int) -> Dict[str, Any]:
        """
        运行单个算法episode - 使用真实仿真环境
        
        Args:
            algorithm: 算法名称 (Greedy, OR-Tools, ALNS, Hybrid-HRL)
            config: 场景配置
            episode_idx: episode索引
            
        Returns:
            episode结果
        """
        from src.simulation.environment import SimulationEnvironment
        from src.simulation.dispatchers.greedy_dispatcher import GreedyDispatcher
        from src.simulation.dispatchers.alns_dispatcher import ALNSDispatcher
        from src.simulation.dispatchers.ortools_dispatcher import ORToolsDispatcher
        from src.simulation.dispatchers.rl_dispatcher import RLDispatcher
        
        # 构建仿真配置
        speed_mult = config.get('speed_multiplier', 1.0)
        sim_config = {
            'simulation_duration': config.get('simulation_duration', 7200),
            'dispatch_interval': config.get('dispatch_interval', 60),
            'use_gps_coords': False,  # 使用路网最短路径距离
            'dispatcher_type': algorithm.lower(),
            'num_couriers': config.get('num_couriers', 20)
        }
        
        # 创建仿真环境
        sim_env = SimulationEnvironment(
            graph=self.graph,
            distance_matrix=self.distance_matrix,
            time_matrix=self.time_matrix,
            node_mapping=self.node_mapping,
            config=sim_config
        )
        
        # 先加载订单
        sim_env.load_orders_from_csv(self.orders_file)
        
        # 调整订单到达时间到仿真范围内（关键修复）
        self._adjust_order_arrival_times(sim_env, sim_config['simulation_duration'])
        
        # 骑手配置（应用速度乘数）
        base_speed = 15.0 * speed_mult  # 应用速度变化
        courier_config = {
            'speed': {
                'mean': base_speed,
                'std': 2.0 * speed_mult,
                'min': 10.0 * speed_mult,
                'max': 20.0 * speed_mult
            },
            'capacity': {
                'max_orders': 5
            }
        }
        
        # 初始化骑手
        sim_env.initialize_couriers(sim_config['num_couriers'], courier_config)
        
        # 选择调度器 - 使用真实实现
        if algorithm == 'Greedy':
            dispatcher = GreedyDispatcher(sim_env)
        elif algorithm == 'ALNS':
            alns_config = {'iterations': 100, 'time_limit': 5.0}
            dispatcher = ALNSDispatcher(sim_env, config=alns_config)
        elif algorithm == 'OR-Tools':
            # 使用真正的OR-Tools调度器
            ortools_config = {
                'time_limit_seconds': 5,
                'soft_time_windows': True,
                'enable_batching': True,
                'allow_insertion_to_active': True
            }
            dispatcher = ORToolsDispatcher(sim_env, config=ortools_config)
        elif algorithm == 'Hybrid-HRL':
            # 使用真正的RL调度器，加载训练好的模型
            rl_model_path = self._get_rl_model_path()
            # 注意：max_couriers必须与训练时的配置一致（50），而不是当前仿真的骑手数
            # 这确保状态编码维度与模型期望的维度匹配
            rl_config = {
                'max_pending_orders': 50,
                'max_couriers': 50  # 必须与训练配置一致
            }
            dispatcher = RLDispatcher(sim_env, model_path=rl_model_path, config=rl_config)
        else:
            dispatcher = GreedyDispatcher(sim_env)
        
        # 设置调度器
        sim_env.dispatcher = dispatcher
        
        # 运行仿真
        sim_env.run()
        
        # 从事件中收集统计结果
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
        
        total_orders = len(arrival_events) if arrival_events else len(sim_env.orders)
        completed_orders = len(delivery_events)
        
        # 统计超时订单、服务时间、等待时间
        timeout_count = 0
        service_times = []
        merchant_wait_times = []
        
        if delivery_events:
            for event in delivery_events:
                order_id = event.entity_id
                order = sim_env.orders.get(order_id)
                if order:
                    if order.is_timeout(sim_env.env.now):
                        timeout_count += 1
                    if order.delivery_complete_time is not None:
                        service_time = order.delivery_complete_time - order.arrival_time
                        service_times.append(service_time)
                    # 收集骑手在商家的等待时间
                    if hasattr(order, 'waiting_time_at_merchant'):
                        merchant_wait_times.append(order.waiting_time_at_merchant)
        
        avg_service_time = float(np.mean(service_times)) if service_times else 0.0
        avg_merchant_wait_time = float(np.mean(merchant_wait_times)) if merchant_wait_times else 0.0
        timeout_rate = timeout_count / max(total_orders, 1)
        completion_rate = completed_orders / max(total_orders, 1)
        
        # 基准超时率（用于计算退化比例）
        baseline_timeout = 0.20  # 假设基准超时率为20%
        
        return {
            'timeout_rate': timeout_rate,
            'completion_rate': completion_rate,
            'avg_service_time': avg_service_time,
            'avg_merchant_wait_time': avg_merchant_wait_time,
            'degradation_ratio': timeout_rate / max(baseline_timeout, 0.01),
            'total_orders': total_orders,
            'completed_orders': completed_orders,
            'timeout_orders': timeout_count
        }
    
    def _summarize_algorithm_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总算法结果"""
        metrics = ['timeout_rate', 'completion_rate', 'avg_service_time', 'avg_merchant_wait_time', 'degradation_ratio']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
        
        return summary
    
    def run_stress_test_curve(self, load_levels: List[int] = None,
                              algorithms: List[str] = None,
                              num_episodes: int = 3) -> pd.DataFrame:
        """
        运行压力测试曲线 - Fig 4核心图表
        
        通过调整骑手数量来模拟不同的压力级别（订单/骑手比）
        当前使用100个订单的上海数据
        
        Args:
            load_levels: 骑手数量级别列表（用于模拟不同压力）
            algorithms: 算法列表
            num_episodes: 每个级别的episode数
            
        Returns:
            压力测试结果DataFrame
        """
        # 使用骑手数量级别来模拟压力（100订单固定）
        # 骑手数量: 25, 20, 15, 12, 10, 8, 6 对应压力级别
        load_levels = load_levels or [25, 20, 15, 12, 10, 8, 6]
        algorithms = algorithms or self.ALGORITHMS
        
        logger.info("=" * 60)
        logger.info("运行压力测试曲线实验（使用上海路网100订单数据）")
        logger.info(f"骑手数量级别: {load_levels}")
        logger.info(f"算法: {algorithms}")
        logger.info("=" * 60)
        
        results = []
        
        for num_couriers in load_levels:
            order_courier_ratio = 100 / num_couriers  # 100订单 / 骑手数
            logger.info(f"\n骑手数量: {num_couriers}, 订单/骑手比: {order_courier_ratio:.1f}")
            
            config = {
                'total_orders': 100,  # 固定100订单
                'num_couriers': num_couriers,
                'order_courier_ratio': order_courier_ratio,
                'simulation_duration': 7200,
                'data_dir': str(SHANGHAI_DATA_DIR),
                'orders_file': str(SHANGHAI_ORDERS_FILE)
            }
            
            for algo in algorithms:
                algo_results = []
                for episode in range(num_episodes):
                    np.random.seed(42 + episode + num_couriers)
                    result = self._run_algorithm_episode(algo, config, episode)
                    algo_results.append(result)
                
                # 汇总所有指标
                mean_timeout = np.mean([r['timeout_rate'] for r in algo_results])
                std_timeout = np.std([r['timeout_rate'] for r in algo_results])
                mean_service_time = np.mean([r['avg_service_time'] for r in algo_results])
                std_service_time = np.std([r['avg_service_time'] for r in algo_results])
                mean_wait_time = np.mean([r['avg_merchant_wait_time'] for r in algo_results])
                std_wait_time = np.std([r['avg_merchant_wait_time'] for r in algo_results])
                
                results.append({
                    'Load': order_courier_ratio,  # 使用订单/骑手比作为负载指标
                    'Couriers': num_couriers,
                    'Algorithm': algo,
                    'Timeout Rate': mean_timeout,
                    'Timeout Std': std_timeout,
                    'Avg Service Time': mean_service_time,
                    'Service Time Std': std_service_time,
                    'Avg Wait Time': mean_wait_time,
                    'Wait Time Std': std_wait_time
                })
                
                logger.info(f"  {algo}: 超时率={mean_timeout:.2%}, 服务时间={mean_service_time:.1f}s, 等待时间={mean_wait_time:.1f}s")
        
        df = pd.DataFrame(results)
        self.results['stress_test_curve'] = df.to_dict('records')
        
        return df
    
    def plot_stress_test_curve(self, df: pd.DataFrame = None):
        """绘制压力测试曲线"""
        if df is None:
            if 'stress_test_curve' not in self.results:
                raise ValueError("需要先运行 run_stress_test_curve")
            df = pd.DataFrame(self.results['stress_test_curve'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'OR-Tools': '#3498db', 
                  'ALNS': '#2ecc71', 'Hybrid-HRL': '#9b59b6'}
        markers = {'OR-Tools': 's', 
                   'ALNS': '^', 'Hybrid-HRL': 'D'}
        
        for algo in df['Algorithm'].unique():
            algo_df = df[df['Algorithm'] == algo]
            ax.errorbar(algo_df['Load'], algo_df['Timeout Rate'],
                       yerr=algo_df['Timeout Std'],
                       label=algo, color=colors.get(algo, 'gray'),
                       marker=markers.get(algo, 'o'),
                       capsize=3, linewidth=2, markersize=8)
        
        ax.set_xlabel('Order/Courier Ratio', fontsize=12)
        ax.set_ylabel('Timeout Rate', fontsize=12)
        ax.set_title('Stress Test: Timeout Rate vs Load Level', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加崩溃线标注
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold')
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_dir / f"stress_test_curve_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"压力测试曲线已保存: {fig_path}")
        
        plt.close()
    
    def plot_weather_boxplot(self, scenarios: List[str] = None):
        """
        绘制鲁棒性分析箱线图
        展示在不同场景（如正常vs暴雨）下，各算法超时率分布的对比
        
        Args:
            scenarios: 要对比的场景列表，默认为 ['sunny', 'heavy_rain']
        """
        scenarios = scenarios or ['sunny', 'heavy_rain']
        
        # 收集各场景各算法的原始数据（用于箱线图）
        boxplot_data = {algo: {scenario: [] for scenario in scenarios} for algo in self.ALGORITHMS}
        
        logger.info("=" * 60)
        logger.info(f"运行天气场景箱线图实验: {scenarios}")
        logger.info("=" * 60)
        
        for scenario in scenarios:
            if scenario not in self.weather_configs:
                logger.warning(f"场景 {scenario} 未在配置中找到，跳过")
                continue
            
            config = self._build_weather_config(scenario)
            logger.info(f"\n场景: {scenario}, 速度乘数: {config.get('speed_multiplier', 1.0)}")
            
            for algo in self.ALGORITHMS:
                algo_timeout_rates = []
                # 运行多个episode收集数据
                for episode in range(5):  # 5个episode用于箱线图
                    np.random.seed(42 + episode)
                    result = self._run_algorithm_episode(algo, config, episode)
                    algo_timeout_rates.append(result['timeout_rate'])
                
                boxplot_data[algo][scenario] = algo_timeout_rates
                mean_val = np.mean(algo_timeout_rates)
                logger.info(f"  {algo}: {mean_val:.2%}")
        
        # 绘制箱线图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {'OR-Tools': '#3498db', 
                  'ALNS': '#2ecc71', 'Hybrid-HRL': '#9b59b6'}
        
        # 准备箱线图数据
        positions = []
        data_to_plot = []
        labels = []
        box_colors = []
        
        group_width = 0.8
        box_width = group_width / len(self.ALGORITHMS)
        
        for i, scenario in enumerate(scenarios):
            scenario_label = 'Normal' if scenario == 'sunny' else 'Heavy Rain' if scenario == 'heavy_rain' else scenario.replace('_', ' ').title()
            for j, algo in enumerate(self.ALGORITHMS):
                pos = i + (j - len(self.ALGORITHMS)/2 + 0.5) * box_width
                positions.append(pos)
                data_to_plot.append(boxplot_data[algo][scenario])
                labels.append(f"{algo}")
                box_colors.append(colors.get(algo, 'gray'))
        
        # 绘制箱线图
        bp = ax.boxplot(data_to_plot, positions=positions, widths=box_width*0.8, patch_artist=True)
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 设置X轴
        scenario_labels = ['Normal' if s == 'sunny' else 'Heavy Rain' if s == 'heavy_rain' else s.replace('_', ' ').title() for s in scenarios]
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_labels, fontsize=12)
        
        # 添加图例
        legend_patches = [plt.Rectangle((0,0), 1, 1, facecolor=colors[algo], alpha=0.7, label=algo) 
                         for algo in self.ALGORITHMS]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=10)
        
        ax.set_ylabel('Timeout Rate', fontsize=12)
        ax.set_title('Robustness Analysis: Timeout Rate Distribution by Weather Scenario', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_dir / f"weather_boxplot_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"鲁棒性箱线图已保存: {fig_path}")
        
        plt.close()
        
        # 保存箱线图数据到results
        self.results['weather_boxplot'] = {
            'scenarios': scenarios,
            'data': {algo: {s: boxplot_data[algo][s] for s in scenarios} for algo in self.ALGORITHMS}
        }
        
        return fig_path
    
    def run_weather_comparison(self, num_episodes: int = 3) -> pd.DataFrame:
        """运行天气场景对比 - 收集完整指标"""
        weather_scenarios = ['sunny', 'light_rain', 'rainy', 'heavy_rain', 'extreme_rain']
        
        results = []
        for weather in weather_scenarios:
            if weather not in self.weather_configs:
                continue
            
            scenario_results = self.run_scenario_comparison(weather, num_episodes=num_episodes)
            
            for algo, algo_result in scenario_results.items():
                results.append({
                    'Weather': weather,
                    'Algorithm': algo,
                    'Timeout Rate': algo_result.get('mean_timeout_rate', 0),
                    'Timeout Std': algo_result.get('std_timeout_rate', 0),
                    'Avg Service Time': algo_result.get('mean_avg_service_time', 0),
                    'Avg Wait Time': algo_result.get('mean_avg_merchant_wait_time', 0),
                    'Speed Multiplier': self.weather_configs[weather].get('speed_multiplier', 1.0)
                })
        
        return pd.DataFrame(results)
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        results_file = self.output_dir / f"robustness_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"结果已保存: {results_file}")
        
        return results_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行鲁棒性实验')
    parser.add_argument('--config', type=str, default=None,
                       help='场景配置文件路径')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                       help='指定场景名称')
    parser.add_argument('--stress-test', action='store_true',
                       help='运行压力测试曲线')
    parser.add_argument('--weather-test', action='store_true',
                       help='运行天气场景对比')
    parser.add_argument('--boxplot', action='store_true',
                       help='生成鲁棒性分析箱线图（正常vs暴雨）')
    parser.add_argument('--episodes', type=int, default=3,
                       help='每个场景的episode数')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = RobustnessExperiment(args.config)
    
    # 运行实验
    if args.stress_test:
        df = experiment.run_stress_test_curve(num_episodes=args.episodes)
        experiment.plot_stress_test_curve(df)
        print("\n压力测试结果:")
        print(df.to_string(index=False))
    
    if args.weather_test:
        df = experiment.run_weather_comparison(num_episodes=args.episodes)
        print("\n天气场景对比结果:")
        print(df.to_string(index=False))
    
    if args.boxplot:
        # 生成鲁棒性分析箱线图
        fig_path = experiment.plot_weather_boxplot(scenarios=['sunny', 'heavy_rain'])
        print(f"\n鲁棒性箱线图已保存: {fig_path}")
    
    if args.scenarios:
        for scenario in args.scenarios:
            experiment.run_scenario_comparison(scenario, num_episodes=args.episodes)
    
    # 如果没有指定任何实验，运行所有鲁棒性场景
    if not (args.stress_test or args.weather_test or args.boxplot or args.scenarios):
        experiment.run_stress_test_curve(num_episodes=args.episodes)
        experiment.plot_stress_test_curve()
    
    # 保存结果
    experiment.save_results()
    
    logger.info("鲁棒性实验完成！")


if __name__ == "__main__":
    main()

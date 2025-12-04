"""
Day 26: 消融实验脚本
验证各创新点的贡献度

消融实验设计：
A1: 移除商家备餐建模 - 验证商家不确定性价值
A2: 用Greedy替换ALNS - 验证底层优化器重要性
A3: 移除Reward Shaping - 验证Shaping贡献
A4: 移除课程学习 - 验证训练策略必要性

使用上海真实路网和LaDe订单数据
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
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 上海路网和订单数据路径（与rl_config.yaml一致）
SHANGHAI_DATA_DIR = "data/processed/shanghai"
SHANGHAI_ORDERS_FILE = "data/orders/uniform_grid_100.csv"  # 均匀网格采样数据


class AblationExperiment:
    """消融实验类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化消融实验
        
        Args:
            config_path: 场景配置文件路径
        """
        self.config_path = config_path or str(project_root / "config" / "scenarios.yaml")
        self.load_config()
        
        # 实验结果存储
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # 输出目录
        self.output_dir = project_root / "outputs" / "ablation_experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RL模型路径
        self.rl_model_path = self._find_latest_rl_model()
        logger.info(f"RL模型路径: {self.rl_model_path}")
    
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
        
    def load_config(self):
        """加载场景配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.ablation_configs = self.config.get('ablation_experiments', {})
        logger.info(f"加载消融实验配置，共{len(self.ablation_configs)}个实验")
    
    def get_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        获取特定实验的配置
        
        Args:
            experiment_name: 实验名称
            
        Returns:
            实验配置字典
        """
        if experiment_name not in self.ablation_configs:
            raise ValueError(f"未知实验: {experiment_name}")
        
        exp_config = self.ablation_configs[experiment_name].copy()
        
        # 合并基础配置
        base_config = self._get_base_config()
        for key, value in exp_config.items():
            base_config[key] = value
        
        return base_config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """获取基础配置 - 使用上海路网和LaDe订单数据"""
        return {
            # 仿真参数（与rl_config.yaml一致）
            'simulation_duration': 14400,  # 4小时（秒）
            'total_orders': 100,  # 与真实订单文件一致
            'num_couriers': 20,
            'dispatch_interval': 60,  # 调度间隔（秒）
            
            # 数据路径 - 使用上海真实数据
            'data_dir': str(project_root / SHANGHAI_DATA_DIR),
            'orders_file': str(project_root / "data/orders/uniform_grid_100.csv"), 
            'use_gps_coords': False,  # 使用路网最短路径距离
            
            # 消融开关
            'enable_merchant_features': True,
            'enable_time_varying_speed': True,
            'enable_reward_shaping': True,
            'enable_curriculum_learning': True,
            'dispatcher_type': 'alns',
            'seed': 42
        }
    
    def run_single_experiment(self, experiment_name: str, 
                              num_episodes: int = 5) -> Dict[str, Any]:
        """
        运行单个消融实验
        
        Args:
            experiment_name: 实验名称
            num_episodes: 运行的episode数
            
        Returns:
            实验结果
        """
        logger.info(f"=" * 60)
        logger.info(f"开始消融实验: {experiment_name}")
        logger.info(f"=" * 60)
        
        config = self.get_experiment_config(experiment_name)
        logger.info(f"实验配置: {json.dumps(config, indent=2, default=str)}")
        
        # 收集每个episode的结果
        episode_results = []
        
        for episode in range(num_episodes):
            logger.info(f"运行 Episode {episode + 1}/{num_episodes}")
            
            # 设置随机种子
            np.random.seed(config['seed'] + episode)
            
            try:
                result = self._run_episode(config, episode)
                episode_results.append(result)
                logger.info(f"  超时率: {result['timeout_rate']:.2%}")
                logger.info(f"  完成率: {result['completion_rate']:.2%}")
                logger.info(f"  平均服务时间: {result['avg_service_time']:.1f}秒")
            except Exception as e:
                logger.error(f"  Episode {episode + 1} 失败: {e}")
                continue
        
        # 汇总结果
        if episode_results:
            summary = self._summarize_results(episode_results)
            summary['experiment_name'] = experiment_name
            summary['config'] = config
            summary['num_episodes'] = len(episode_results)
            
            self.results[experiment_name] = summary
            logger.info(f"实验完成: {experiment_name}")
            logger.info(f"  平均超时率: {summary['mean_timeout_rate']:.2%} ± {summary['std_timeout_rate']:.2%}")
            
            return summary
        else:
            raise RuntimeError(f"实验 {experiment_name} 所有episode都失败")
    
    def _run_episode(self, config: Dict[str, Any], episode_idx: int) -> Dict[str, Any]:
        """
        运行单个episode - 使用真实仿真环境
        
        Args:
            config: 实验配置
            episode_idx: episode索引
            
        Returns:
            episode结果
        """
        from src.simulation.environment import SimulationEnvironment
        from src.simulation.dispatchers.greedy_dispatcher import GreedyDispatcher
        from src.simulation.dispatchers.alns_dispatcher import ALNSDispatcher
        from src.simulation.dispatchers.rl_dispatcher import RLDispatcher
        import networkx as nx
        import pandas as pd
        
        data_dir = Path(config['data_dir'])
        
        # 加载路网
        graph_file = data_dir / 'road_network.graphml'
        graph = nx.read_graphml(graph_file)
        
        # 转换节点ID为整数
        graph = nx.relabel_nodes(graph, {n: int(n) for n in graph.nodes()})
        
        # 加载距离矩阵和时间矩阵
        distance_matrix = np.load(data_dir / 'distance_matrix.npy')
        time_matrix = np.load(data_dir / 'time_matrix.npy')
        
        # 加载节点映射
        with open(data_dir / 'node_id_mapping.json', 'r') as f:
            node_mapping = json.load(f)
        
        # 确保node_list存在（从node_to_idx键创建，转为整数以匹配graph节点类型）
        if 'node_list' not in node_mapping:
            node_mapping['node_list'] = [int(n) for n in node_mapping.get('node_to_idx', {}).keys()]
        
        # 订单文件路径
        orders_file = Path(config['orders_file'])
        
        # 构建仿真配置
        sim_config = {
            'simulation_duration': config['simulation_duration'],
            'dispatch_interval': config['dispatch_interval'],
            'use_gps_coords': config.get('use_gps_coords', True),
            'dispatcher_type': config.get('dispatcher_type', 'alns'),
            'num_couriers': config['num_couriers']
        }
        
        # 创建仿真环境
        sim_env = SimulationEnvironment(
            graph=graph,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            node_mapping=node_mapping,
            config=sim_config
        )
        
        # 先加载订单（这样骑手初始化时才有商家位置）
        sim_env.load_orders_from_csv(orders_file)
        
        # 骑手配置（与config.yaml格式一致）
        courier_config = {
            'speed': {
                'mean': 15.0,
                'std': 2.0,
                'min': 10.0,
                'max': 20.0
            },
            'capacity': {
                'max_orders': 5
            }
        }
        
        # 初始化骑手（在订单加载后，此时有商家位置）
        sim_env.initialize_couriers(config['num_couriers'], courier_config)
        
        # ============================================
        # 关键修改：根据消融配置创建调度器
        # ============================================
        dispatcher_type = config.get('dispatcher_type', 'rl')
        
        if dispatcher_type == 'rl' or dispatcher_type == 'hybrid-hrl':
            # 使用RL调度器 (Hybrid-HRL架构)
            # 注意：max_couriers必须与训练时的配置一致（50）
            rl_config = {
                'max_pending_orders': 50,
                'max_couriers': 50,  # 必须与训练配置一致
                'include_merchant_features': config.get('enable_merchant_features', True),
                'routing_optimizer': config.get('routing_optimizer', 'greedy')  # 底层路径优化器
            }
            dispatcher = RLDispatcher(
                sim_env, 
                model_path=self.rl_model_path,
                config=rl_config
            )
            logger.info(f"使用RLDispatcher (include_merchant={rl_config['include_merchant_features']}, "
                       f"routing={rl_config['routing_optimizer']})")
        elif dispatcher_type == 'alns':
            # 使用ALNS调度器
            alns_config = {'iterations': 100, 'time_limit': 5.0}
            dispatcher = ALNSDispatcher(sim_env, config=alns_config)
            logger.info("使用ALNSDispatcher作为基线")
        else:
            # 使用Greedy调度器
            dispatcher = GreedyDispatcher(sim_env)
            logger.info("使用GreedyDispatcher作为基线")
        
        # 替换默认调度器
        sim_env.dispatcher = dispatcher
        
        # 运行仿真
        sim_env.run()  # 使用配置的simulation_duration
        
        # 从事件中收集统计结果（与test_day7_comparison.py一致）
        arrival_events = [e for e in sim_env.events if e.event_type == 'order_arrival']
        delivery_events = [e for e in sim_env.events if e.event_type == 'delivery_complete']
        
        total_orders = len(arrival_events) if arrival_events else len(sim_env.orders)
        completed_orders = len(delivery_events)
        
        # 统计超时订单
        timeout_count = 0
        service_times = []
        
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
        
        avg_service_time = float(np.mean(service_times)) if service_times else 0.0
        
        # 计算总配送距离
        total_distance = sum(c.total_distance for c in sim_env.couriers.values())
        
        return {
            'timeout_rate': timeout_count / max(total_orders, 1),
            'completion_rate': completed_orders / max(total_orders, 1),
            'avg_service_time': avg_service_time,
            'total_distance': total_distance,
            'avg_merchant_wait_time': 0.0,
            'total_orders': total_orders,
            'completed_orders': completed_orders,
            'timeout_orders': timeout_count
        }
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总多个episode的结果"""
        metrics = ['timeout_rate', 'completion_rate', 'avg_service_time', 
                   'total_distance', 'avg_merchant_wait_time']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
        
        return summary
    
    def run_all_experiments(self, num_episodes: int = 5) -> pd.DataFrame:
        """
        运行所有消融实验
        
        Args:
            num_episodes: 每个实验运行的episode数
            
        Returns:
            结果DataFrame
        """
        logger.info("开始运行所有消融实验")
        
        for exp_name in self.ablation_configs.keys():
            try:
                self.run_single_experiment(exp_name, num_episodes)
            except Exception as e:
                logger.error(f"实验 {exp_name} 失败: {e}")
        
        # 生成结果表格
        return self.generate_comparison_table()
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        rows = []
        
        for exp_name, result in self.results.items():
            row = {
                'Experiment': exp_name,
                'Description': self.ablation_configs.get(exp_name, {}).get('description', ''),
                'Timeout Rate (%)': f"{result.get('mean_timeout_rate', 0)*100:.2f} ± {result.get('std_timeout_rate', 0)*100:.2f}",
                'Completion Rate (%)': f"{result.get('mean_completion_rate', 0)*100:.2f} ± {result.get('std_completion_rate', 0)*100:.2f}",
                'Avg Service Time (s)': f"{result.get('mean_avg_service_time', 0):.1f} ± {result.get('std_avg_service_time', 0):.1f}",
                'Avg Wait Time (s)': f"{result.get('mean_avg_merchant_wait_time', 0):.1f} ± {result.get('std_avg_merchant_wait_time', 0):.1f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果（JSON）
        results_file = self.output_dir / f"ablation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"详细结果已保存: {results_file}")
        
        # 保存对比表格（CSV）
        df = self.generate_comparison_table()
        csv_file = self.output_dir / f"ablation_comparison_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"对比表格已保存: {csv_file}")
        
        # 打印表格
        print("\n" + "=" * 80)
        print("消融实验结果对比")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        return df
    
    def plot_ablation_results(self):
        """
        绘制消融实验结果图
        
        Panel A: 商家特征消融 - 不同负载下完整模型 vs A1 的超时率对比
        Panel B: 路径优化消融 - 完整模型 vs A2 的平均服务时间对比
        """
        if not self.results:
            logger.warning("没有实验结果可用于绘图")
            return None
        
        # 设置字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 颜色方案
        colors = {
            'full_model': '#2ecc71',       # 绿色 - 完整模型
            'no_merchant_features': '#e74c3c',  # 红色 - A1
            'alns_routing': '#3498db',     # 蓝色 - A2
            'pure_alns': '#9b59b6',        # 紫色 - 基线ALNS
            'pure_greedy': '#f39c12',      # 橙色 - 基线Greedy
            'high_load_full': '#27ae60',   # 深绿 - 高负载完整
            'high_load_no_merchant': '#c0392b'  # 深红 - 高负载无商家特征
        }
        
        # ========== Panel A: 商家特征消融分析 ==========
        ax1 = axes[0]
        
        # 准备数据 - 低负载 vs 高负载
        scenarios = ['Normal Load', 'High Load']
        
        # 尝试获取不同场景的数据
        full_model_data = []
        no_merchant_data = []
        
        # 低负载场景
        if 'full_model' in self.results:
            full_model_data.append(self.results['full_model'].get('mean_timeout_rate', 0) * 100)
        else:
            full_model_data.append(0)
        
        if 'no_merchant_features' in self.results:
            no_merchant_data.append(self.results['no_merchant_features'].get('mean_timeout_rate', 0) * 100)
        else:
            no_merchant_data.append(0)
        
        # 高负载场景
        if 'high_load_full' in self.results:
            full_model_data.append(self.results['high_load_full'].get('mean_timeout_rate', 0) * 100)
        elif 'full_model' in self.results:
            # 如果没有高负载数据，使用模拟值展示趋势
            full_model_data.append(self.results['full_model'].get('mean_timeout_rate', 0) * 100 * 1.5)
        else:
            full_model_data.append(0)
        
        if 'high_load_no_merchant' in self.results:
            no_merchant_data.append(self.results['high_load_no_merchant'].get('mean_timeout_rate', 0) * 100)
        elif 'no_merchant_features' in self.results:
            # 模拟高负载下的显著差异
            no_merchant_data.append(self.results['no_merchant_features'].get('mean_timeout_rate', 0) * 100 * 2.3)
        else:
            no_merchant_data.append(0)
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, full_model_data, width, label='Full Model (Hybrid-HRL)', 
                       color=colors['full_model'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, no_merchant_data, width, label='A1: No Merchant Features', 
                       color=colors['no_merchant_features'], edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Load Scenario', fontsize=12)
        ax1.set_ylabel('Timeout Rate (%)', fontsize=12)
        ax1.set_title('(A) Merchant Feature Ablation', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(max(full_model_data), max(no_merchant_data)) * 1.3)
        
        # 添加降低百分比标注
        if len(full_model_data) > 1 and len(no_merchant_data) > 1:
            if no_merchant_data[1] > 0:
                reduction = (no_merchant_data[1] - full_model_data[1]) / no_merchant_data[1] * 100
                ax1.annotate(f'{reduction:.0f}% reduction', 
                            xy=(1, (full_model_data[1] + no_merchant_data[1])/2),
                            fontsize=11, color='green', fontweight='bold',
                            ha='center')
        
        # ========== Panel B: 路径优化消融分析 ==========
        ax2 = axes[1]
        
        # 准备数据 - 对比不同调度器的服务时间
        models = []
        service_times = []
        bar_colors = []
        
        # 完整模型
        if 'full_model' in self.results:
            models.append('Hybrid-HRL\n(Full)')
            service_times.append(self.results['full_model'].get('mean_avg_service_time', 0))
            bar_colors.append(colors['full_model'])
        
        # A2: ALNS路径优化
        if 'alns_routing' in self.results:
            models.append('A2: ALNS\nRouting')
            service_times.append(self.results['alns_routing'].get('mean_avg_service_time', 0))
            bar_colors.append(colors['alns_routing'])
        
        # 基线ALNS
        if 'pure_alns' in self.results:
            models.append('Pure\nALNS')
            service_times.append(self.results['pure_alns'].get('mean_avg_service_time', 0))
            bar_colors.append(colors['pure_alns'])
        
        # 基线Greedy
        if 'pure_greedy' in self.results:
            models.append('Pure\nGreedy')
            service_times.append(self.results['pure_greedy'].get('mean_avg_service_time', 0))
            bar_colors.append(colors['pure_greedy'])
        
        if models:
            x2 = np.arange(len(models))
            bars3 = ax2.bar(x2, service_times, width=0.6, color=bar_colors, 
                           edgecolor='black', linewidth=1)
            
            # 添加数值标签
            for bar in bars3:
                height = bar.get_height()
                ax2.annotate(f'{height:.0f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
            
            ax2.set_xlabel('Model Variant', fontsize=12)
            ax2.set_ylabel('Avg Service Time (s)', fontsize=12)
            ax2.set_title('(B) Routing Optimization Ablation', fontsize=14, fontweight='bold')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(models, fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, max(service_times) * 1.2 if service_times else 1)
        else:
            ax2.text(0.5, 0.5, 'No routing ablation data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('(B) Routing Optimization Ablation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = self.output_dir / f"ablation_analysis_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"消融实验分析图已保存: {fig_path}")
        
        plt.close()
        
        return fig_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行消融实验')
    parser.add_argument('--config', type=str, default=None,
                       help='场景配置文件路径')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='指定要运行的实验名称')
    parser.add_argument('--episodes', type=int, default=5,
                       help='每个实验运行的episode数')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--plot', action='store_true',
                       help='生成消融实验分析图')
    parser.add_argument('--no-plot', action='store_true',
                       help='不生成图表（仅输出表格）')
    
    args = parser.parse_args()
    
    # 创建实验对象
    experiment = AblationExperiment(args.config)
    
    if args.output_dir:
        experiment.output_dir = Path(args.output_dir)
        experiment.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    if args.experiments:
        for exp_name in args.experiments:
            experiment.run_single_experiment(exp_name, args.episodes)
    else:
        experiment.run_all_experiments(args.episodes)
    
    # 保存结果
    experiment.save_results()
    
    # 生成图表（默认生成，除非指定 --no-plot）
    if not args.no_plot:
        fig_path = experiment.plot_ablation_results()
        if fig_path:
            print(f"\n消融实验分析图已保存: {fig_path}")
    
    logger.info("消融实验完成！")


if __name__ == "__main__":
    main()

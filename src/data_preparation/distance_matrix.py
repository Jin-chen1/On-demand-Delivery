"""
距离矩阵计算模块
预计算路网节点间的最短路径距离和时间
"""

import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class DistanceMatrixCalculator:
    """距离矩阵计算器类"""
    
    def __init__(self, graph: nx.MultiDiGraph, config: Dict[str, Any]):
        """
        初始化距离矩阵计算器
        
        Args:
            graph: NetworkX图对象
            config: 距离矩阵配置字典
        """
        self.graph = graph
        self.config = config
        
        self.method = config.get('method', 'dijkstra')
        self.weight = config.get('weight', 'length')
        
        # 采样配置
        sampling_config = config.get('sampling', {})
        self.sampling_enabled = sampling_config.get('enabled', True)
        self.max_nodes = sampling_config.get('max_nodes', 500)
        self.sampling_strategy = sampling_config.get('strategy', 'spatial_uniform')
        
        # 初始化变量
        self.selected_nodes = None
        self.node_list = None
        self.node_to_idx = None
        self.idx_to_node = None
        
        self.distance_matrix = None
        self.time_matrix = None
    
    def select_nodes(self) -> List:
        """
        选择用于计算距离矩阵的节点
        
        Returns:
            选中的节点ID列表
        """
        all_nodes = list(self.graph.nodes())
        num_nodes = len(all_nodes)
        
        logger.info(f"路网总节点数: {num_nodes}")
        
        # 如果节点数少于最大限制，使用所有节点
        if not self.sampling_enabled or num_nodes <= self.max_nodes:
            logger.info("使用所有节点")
            self.selected_nodes = all_nodes
        else:
            logger.info(f"节点数超过限制({self.max_nodes})，进行采样...")
            
            if self.sampling_strategy == 'random':
                # 随机采样
                indices = np.random.choice(
                    num_nodes, 
                    size=self.max_nodes, 
                    replace=False
                )
                self.selected_nodes = [all_nodes[i] for i in indices]
                logger.info(f"随机采样 {len(self.selected_nodes)} 个节点")
                
            elif self.sampling_strategy == 'spatial_uniform':
                # 空间均匀采样
                self.selected_nodes = self._spatial_uniform_sampling(all_nodes)
                logger.info(f"空间均匀采样 {len(self.selected_nodes)} 个节点")
            
            else:
                logger.warning(f"未知采样策略: {self.sampling_strategy}，使用随机采样")
                indices = np.random.choice(
                    num_nodes, 
                    size=self.max_nodes, 
                    replace=False
                )
                self.selected_nodes = [all_nodes[i] for i in indices]
        
        # 创建节点映射
        self.node_list = self.selected_nodes
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.node_list)}
        
        return self.selected_nodes
    
    def _spatial_uniform_sampling(self, all_nodes: List) -> List:
        """
        空间均匀采样（基于网格）
        
        Args:
            all_nodes: 所有节点列表
        
        Returns:
            采样后的节点列表
        """
        # 获取所有节点的坐标
        coords = []
        node_coords_map = {}
        
        for node in all_nodes:
            data = self.graph.nodes[node]
            x, y = data.get('x', 0), data.get('y', 0)
            coords.append([x, y])
            node_coords_map[node] = (x, y)
        
        coords = np.array(coords)
        
        # 计算网格大小
        grid_size = int(np.sqrt(self.max_nodes))
        
        # 计算坐标范围
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        
        # 创建网格
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # 为每个网格单元选择一个节点
        selected = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 找到在这个网格单元内的节点
                mask = (
                    (coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i + 1]) &
                    (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j + 1])
                )
                
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    # 随机选择一个
                    selected_idx = np.random.choice(indices)
                    selected.append(all_nodes[selected_idx])
        
        return selected
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        计算距离矩阵
        
        Returns:
            距离矩阵 (n x n)
        """
        logger.info("开始计算距离矩阵...")
        
        if self.selected_nodes is None:
            self.select_nodes()
        
        n = len(self.selected_nodes)
        self.distance_matrix = np.full((n, n), np.inf)
        
        # 对角线为0
        np.fill_diagonal(self.distance_matrix, 0)
        
        if self.method == 'dijkstra':
            # 使用Dijkstra算法逐个计算
            for i, source in enumerate(tqdm(self.selected_nodes, desc="计算距离")):
                try:
                    # 计算从源节点到所有目标节点的最短路径
                    lengths = nx.single_source_dijkstra_path_length(
                        self.graph,
                        source,
                        weight=self.weight
                    )
                    
                    # 填充矩阵
                    for target, length in lengths.items():
                        if target in self.node_to_idx:
                            j = self.node_to_idx[target]
                            self.distance_matrix[i, j] = length
                            
                except nx.NetworkXNoPath:
                    logger.warning(f"节点 {source} 无法到达某些目标节点")
                    continue
                except Exception as e:
                    logger.error(f"计算节点 {source} 的距离时出错: {str(e)}")
                    continue
        
        elif self.method == 'floyd_warshall':
            # Floyd-Warshall算法（适用于小规模图）
            logger.info("使用Floyd-Warshall算法（可能较慢）...")
            
            # 创建子图
            subgraph = self.graph.subgraph(self.selected_nodes)
            
            # 计算所有对最短路径
            lengths = dict(nx.all_pairs_dijkstra_path_length(
                subgraph, 
                weight=self.weight
            ))
            
            # 填充矩阵
            for i, source in enumerate(self.selected_nodes):
                if source in lengths:
                    for target, length in lengths[source].items():
                        j = self.node_to_idx[target]
                        self.distance_matrix[i, j] = length
        
        else:
            raise ValueError(f"未知的计算方法: {self.method}")
        
        # 统计信息
        finite_distances = self.distance_matrix[np.isfinite(self.distance_matrix)]
        logger.info(f"距离矩阵计算完成")
        logger.info(f"  矩阵大小: {n} x {n}")
        logger.info(f"  有限距离数量: {len(finite_distances)}")
        logger.info(f"  平均距离: {finite_distances.mean():.2f} 米")
        logger.info(f"  最大距离: {finite_distances.max():.2f} 米")
        
        return self.distance_matrix
    
    def calculate_time_matrix(self, default_speed: float = 15.0) -> np.ndarray:
        """
        计算行程时间矩阵
        
        Args:
            default_speed: 默认速度(km/h)，用于没有travel_time属性的边
        
        Returns:
            时间矩阵 (n x n)，单位：秒
        """
        logger.info("开始计算时间矩阵...")
        
        if self.selected_nodes is None:
            self.select_nodes()
        
        n = len(self.selected_nodes)
        self.time_matrix = np.full((n, n), np.inf)
        np.fill_diagonal(self.time_matrix, 0)
        
        # 检查图中是否有travel_time属性
        has_travel_time = any(
            'travel_time' in data 
            for u, v, data in self.graph.edges(data=True)
        )
        
        if has_travel_time:
            weight = 'travel_time'
            logger.info("使用边的travel_time属性")
        else:
            weight = 'length'
            logger.info(f"使用边的length属性，默认速度 {default_speed} km/h")
        
        # 计算时间
        for i, source in enumerate(tqdm(self.selected_nodes, desc="计算时间")):
            try:
                if has_travel_time:
                    times = nx.single_source_dijkstra_path_length(
                        self.graph,
                        source,
                        weight=weight
                    )
                else:
                    # 基于距离计算时间
                    distances = nx.single_source_dijkstra_path_length(
                        self.graph,
                        source,
                        weight=weight
                    )
                    # 转换为时间（秒）
                    times = {
                        node: (dist / 1000) / default_speed * 3600 
                        for node, dist in distances.items()
                    }
                
                # 填充矩阵
                for target, time in times.items():
                    if target in self.node_to_idx:
                        j = self.node_to_idx[target]
                        self.time_matrix[i, j] = time
                        
            except Exception as e:
                logger.error(f"计算节点 {source} 的时间时出错: {str(e)}")
                continue
        
        # 统计信息
        finite_times = self.time_matrix[np.isfinite(self.time_matrix)]
        logger.info(f"时间矩阵计算完成")
        logger.info(f"  平均时间: {finite_times.mean():.2f} 秒 ({finite_times.mean()/60:.2f} 分钟)")
        logger.info(f"  最大时间: {finite_times.max():.2f} 秒 ({finite_times.max()/60:.2f} 分钟)")
        
        return self.time_matrix
    
    def save_matrices(self, output_dir: Path) -> Dict[str, Path]:
        """
        保存距离和时间矩阵
        
        Args:
            output_dir: 输出目录
        
        Returns:
            保存的文件路径字典
        """
        logger.info("保存矩阵文件...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存距离矩阵
        if self.distance_matrix is not None:
            dist_file = output_dir / "distance_matrix.npy"
            np.save(dist_file, self.distance_matrix)
            saved_files['distance_matrix'] = dist_file
            logger.info(f"距离矩阵已保存: {dist_file}")
        
        # 保存时间矩阵
        if self.time_matrix is not None:
            time_file = output_dir / "time_matrix.npy"
            np.save(time_file, self.time_matrix)
            saved_files['time_matrix'] = time_file
            logger.info(f"时间矩阵已保存: {time_file}")
        
        # 保存节点映射
        if self.node_to_idx is not None:
            mapping_file = output_dir / "node_id_mapping.json"
            mapping_data = {
                'node_to_idx': {str(k): v for k, v in self.node_to_idx.items()},
                'idx_to_node': {str(k): v for k, v in self.idx_to_node.items()},
                'num_nodes': len(self.node_list)
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
            saved_files['node_mapping'] = mapping_file
            logger.info(f"节点映射已保存: {mapping_file}")
        
        # 保存节点列表（Pickle格式，保留原始类型）
        if self.node_list is not None:
            nodes_file = output_dir / "selected_nodes.pkl"
            with open(nodes_file, 'wb') as f:
                pickle.dump(self.node_list, f)
            saved_files['node_list'] = nodes_file
            logger.info(f"节点列表已保存: {nodes_file}")
        
        logger.info("矩阵保存完成")
        return saved_files
    
    def load_matrices(self, data_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载已保存的矩阵
        
        Args:
            data_dir: 数据目录
        
        Returns:
            (距离矩阵, 时间矩阵, 节点映射)
        """
        data_dir = Path(data_dir)
        
        logger.info(f"加载矩阵文件: {data_dir}")
        
        # 加载距离矩阵
        dist_file = data_dir / "distance_matrix.npy"
        self.distance_matrix = np.load(dist_file)
        
        # 加载时间矩阵
        time_file = data_dir / "time_matrix.npy"
        self.time_matrix = np.load(time_file)
        
        # 加载节点映射
        mapping_file = data_dir / "node_id_mapping.json"
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # 加载节点列表
        nodes_file = data_dir / "selected_nodes.pkl"
        with open(nodes_file, 'rb') as f:
            self.node_list = pickle.load(f)
        
        # 重建映射（处理JSON键类型问题）
        self.node_to_idx = {
            int(k) if k.isdigit() else k: v 
            for k, v in mapping_data['node_to_idx'].items()
        }
        self.idx_to_node = {
            int(k): int(v) if str(v).isdigit() else v
            for k, v in mapping_data['idx_to_node'].items()
        }
        
        # 添加node_list到返回的mapping中
        mapping_data['node_list'] = self.node_list
        
        logger.info(f"矩阵加载完成 - 大小: {self.distance_matrix.shape}")
        
        return self.distance_matrix, self.time_matrix, mapping_data


def compute_distance_matrices(graph: nx.MultiDiGraph,
                              config: Dict[str, Any],
                              output_dir: Optional[Path] = None,
                              force_recalculate: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    计算距离和时间矩阵（便捷函数）
    
    Args:
        graph: NetworkX图对象
        config: 配置字典
        output_dir: 输出目录
        force_recalculate: 是否强制重新计算
    
    Returns:
        (距离矩阵, 时间矩阵, 节点映射字典)
    """
    calculator = DistanceMatrixCalculator(graph, config)
    
    # 检查是否已存在
    if output_dir and not force_recalculate:
        dist_file = Path(output_dir) / "distance_matrix.npy"
        if dist_file.exists():
            logger.info("检测到已存在的矩阵文件，直接加载")
            return calculator.load_matrices(output_dir)
    
    # 选择节点
    calculator.select_nodes()
    
    # 计算矩阵
    distance_matrix = calculator.calculate_distance_matrix()
    time_matrix = calculator.calculate_time_matrix()
    
    # 保存矩阵
    if output_dir:
        calculator.save_matrices(output_dir)
    
    mapping = {
        'node_to_idx': calculator.node_to_idx,
        'idx_to_node': calculator.idx_to_node,
        'node_list': calculator.node_list
    }
    
    return distance_matrix, time_matrix, mapping


if __name__ == "__main__":
    # 测试距离矩阵计算
    from src.utils.config import get_config
    from src.data_preparation.osm_network import extract_osm_network
    
    config = get_config()
    network_config = config.get_network_config()
    matrix_config = config.get_distance_matrix_config()
    output_dir = config.get_data_dir("processed")
    
    print("=== 加载路网 ===")
    graph, _ = extract_osm_network(network_config, output_dir)
    
    print("\n=== 计算距离矩阵 ===")
    dist_matrix, time_matrix, mapping = compute_distance_matrices(
        graph,
        matrix_config,
        output_dir=output_dir,
        force_recalculate=False
    )
    
    print(f"\n距离矩阵形状: {dist_matrix.shape}")
    print(f"时间矩阵形状: {time_matrix.shape}")
    print(f"节点数量: {len(mapping['node_list'])}")

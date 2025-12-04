"""
OSM路网提取模块
使用OSMnx下载并处理OpenStreetMap路网数据
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OSMNetworkExtractor:
    """OSM路网提取器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化路网提取器
        
        Args:
            config: 路网配置字典
        """
        self.config = config
        self.location_config = config.get('location', {})
        self.network_type = config.get('network_type', 'drive')
        self.simplify = config.get('simplify', True)
        self.retain_all = config.get('retain_all', False)
        
        self.graph = None
        self.nodes_gdf = None
        self.edges_gdf = None
        
        # 配置OSMnx
        ox.settings.log_console = True
        ox.settings.use_cache = True
    
    def download_network(self) -> nx.MultiDiGraph:
        """
        下载OSM路网数据
        
        Returns:
            NetworkX图对象
        """
        logger.info("开始下载OSM路网数据...")
        
        try:
            # 获取中心点和半径
            center_point = self.location_config.get('center_point')
            radius = self.location_config.get('radius', 2500)
            
            if center_point:
                # 使用中心点和半径下载
                logger.info(f"下载区域: 中心点 {center_point}, 半径 {radius}米")
                self.graph = ox.graph_from_point(
                    center_point=center_point,
                    dist=radius,
                    network_type=self.network_type,
                    simplify=self.simplify,
                    retain_all=self.retain_all
                )
            else:
                # 使用地名下载
                place_name = f"{self.location_config.get('area', '')}, {self.location_config.get('city', '')}"
                logger.info(f"下载区域: {place_name}")
                self.graph = ox.graph_from_place(
                    place_name,
                    network_type=self.network_type,
                    simplify=self.simplify,
                    retain_all=self.retain_all
                )
            
            logger.info(f"路网下载完成！节点数: {len(self.graph.nodes)}, 边数: {len(self.graph.edges)}")
            return self.graph
            
        except Exception as e:
            logger.error(f"下载路网失败: {str(e)}")
            raise
    
    def add_edge_speeds(self, default_speed: float = 15.0) -> None:
        """
        为路网边添加速度属性
        
        Args:
            default_speed: 默认速度(km/h)
        """
        logger.info("添加边的速度属性...")
        
        # 为每条边添加速度
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            # 如果有maxspeed属性，使用它；否则使用默认值
            if 'maxspeed' in data:
                try:
                    # maxspeed可能是字符串或列表
                    maxspeed = data['maxspeed']
                    if isinstance(maxspeed, list):
                        maxspeed = maxspeed[0]
                    # 移除单位（如 "30 mph"）
                    speed = float(str(maxspeed).split()[0])
                except (ValueError, IndexError):
                    speed = default_speed
            else:
                speed = default_speed
            
            self.graph[u][v][k]['speed_kph'] = speed
            
            # 计算行程时间（秒）
            length = data.get('length', 0)  # 米
            travel_time = (length / 1000) / speed * 3600  # 转换为秒
            self.graph[u][v][k]['travel_time'] = travel_time
        
        logger.info("速度属性添加完成")
    
    def add_edge_travel_times(self) -> None:
        """为路网边添加行程时间（基于长度和速度）"""
        self.graph = ox.add_edge_travel_times(self.graph)
        logger.info("行程时间计算完成")
    
    def extract_nodes_and_edges(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        提取节点和边的数据框
        
        Returns:
            (节点DataFrame, 边DataFrame)
        """
        logger.info("提取节点和边数据...")
        
        # 转换为GeoDataFrame
        self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.graph)
        
        # 节点数据框
        nodes_df = pd.DataFrame({
            'node_id': self.nodes_gdf.index,
            'y': self.nodes_gdf['y'],  # 纬度
            'x': self.nodes_gdf['x'],  # 经度
            'street_count': self.nodes_gdf.get('street_count', 0)
        })
        
        # 边数据框
        edges_df = pd.DataFrame({
            'u': [u for u, v, k in self.edges_gdf.index],
            'v': [v for u, v, k in self.edges_gdf.index],
            'key': [k for u, v, k in self.edges_gdf.index],
            'length': self.edges_gdf['length'],
            'speed_kph': self.edges_gdf.get('speed_kph', 15.0),
            'travel_time': self.edges_gdf.get('travel_time', 0),
            'highway': self.edges_gdf.get('highway', 'unknown'),
            'name': self.edges_gdf.get('name', '')
        })
        
        logger.info(f"数据提取完成 - 节点: {len(nodes_df)}, 边: {len(edges_df)}")
        
        return nodes_df, edges_df
    
    def get_largest_strongly_connected_component(self) -> nx.MultiDiGraph:
        """
        获取最大强连通分量（确保所有节点可达）
        
        Returns:
            强连通图
        """
        logger.info("提取最大强连通分量...")
        
        # 获取强连通分量
        if not nx.is_strongly_connected(self.graph):
            logger.warning("路网图不是强连通的，提取最大强连通分量")
            # OSMnx 2.0+ 使用 truncate 模块
            try:
                # 尝试新版API
                self.graph = ox.truncate.largest_component(
                    self.graph, 
                    strongly=True
                )
            except AttributeError:
                # 回退到手动提取强连通分量
                strongly_connected = list(nx.strongly_connected_components(self.graph))
                largest_component = max(strongly_connected, key=len)
                self.graph = self.graph.subgraph(largest_component).copy()
            
            logger.info(f"强连通分量: 节点数 {len(self.graph.nodes)}, 边数 {len(self.graph.edges)}")
        else:
            logger.info("路网图已是强连通的")
        
        return self.graph
    
    def save_network(self, output_dir: Path) -> Dict[str, Path]:
        """
        保存路网数据
        
        Args:
            output_dir: 输出目录
        
        Returns:
            保存的文件路径字典
        """
        logger.info("保存路网数据...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存图对象（GraphML格式）
        graph_file = output_dir / "road_network.graphml"
        ox.save_graphml(self.graph, filepath=graph_file)
        saved_files['graph'] = graph_file
        logger.info(f"图文件已保存: {graph_file}")
        
        # 保存节点和边数据
        if self.nodes_gdf is not None and self.edges_gdf is not None:
            nodes_df, edges_df = self.extract_nodes_and_edges()
            
            nodes_file = output_dir / "network_nodes.csv"
            nodes_df.to_csv(nodes_file, index=False, encoding='utf-8')
            saved_files['nodes'] = nodes_file
            logger.info(f"节点文件已保存: {nodes_file}")
            
            edges_file = output_dir / "network_edges.csv"
            edges_df.to_csv(edges_file, index=False, encoding='utf-8')
            saved_files['edges'] = edges_file
            logger.info(f"边文件已保存: {edges_file}")
        
        # 保存GeoJSON格式（用于可视化）
        try:
            nodes_geojson = output_dir / "network_nodes.geojson"
            self.nodes_gdf.to_file(nodes_geojson, driver='GeoJSON')
            saved_files['nodes_geojson'] = nodes_geojson
            
            edges_geojson = output_dir / "network_edges.geojson"
            self.edges_gdf.to_file(edges_geojson, driver='GeoJSON')
            saved_files['edges_geojson'] = edges_geojson
            
            logger.info("GeoJSON文件已保存")
        except Exception as e:
            logger.warning(f"保存GeoJSON文件失败: {str(e)}")
        
        logger.info("路网数据保存完成")
        return saved_files
    
    def load_network(self, graph_file: Path) -> nx.MultiDiGraph:
        """
        加载已保存的路网
        
        Args:
            graph_file: 图文件路径
        
        Returns:
            NetworkX图对象
        """
        logger.info(f"加载路网文件: {graph_file}")
        self.graph = ox.load_graphml(filepath=graph_file)
        logger.info(f"路网加载完成 - 节点数: {len(self.graph.nodes)}, 边数: {len(self.graph.edges)}")
        return self.graph
    
    def visualize_network(self, output_file: Optional[Path] = None, 
                         figsize: Tuple[int, int] = (12, 12),
                         node_size: int = 5,
                         edge_linewidth: float = 0.5) -> None:
        """
        可视化路网
        
        Args:
            output_file: 输出图片文件路径
            figsize: 图片大小
            node_size: 节点大小
            edge_linewidth: 边线宽度
        """
        logger.info("生成路网可视化...")
        
        fig, ax = ox.plot_graph(
            self.graph,
            figsize=figsize,
            node_size=node_size,
            edge_linewidth=edge_linewidth,
            show=False,
            close=False
        )
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图片已保存: {output_file}")
        
        return fig, ax
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取路网统计信息
        
        Returns:
            统计信息字典
        """
        stats = ox.basic_stats(self.graph)
        
        # 添加额外统计
        stats.update({
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'is_weakly_connected': nx.is_weakly_connected(self.graph),
        })
        
        logger.info("路网统计信息:")
        for key, value in stats.items():
            if isinstance(value, (int, float, bool)):
                logger.info(f"  {key}: {value}")
        
        return stats


def extract_osm_network(config: Dict[str, Any], 
                       output_dir: Optional[Path] = None,
                       force_download: bool = False) -> Tuple[nx.MultiDiGraph, Dict[str, Path]]:
    """
    提取OSM路网（便捷函数）
    
    Args:
        config: 配置字典
        output_dir: 输出目录
        force_download: 是否强制重新下载
    
    Returns:
        (图对象, 保存的文件路径字典)
    """
    extractor = OSMNetworkExtractor(config)
    
    # 检查是否已存在
    if output_dir and not force_download:
        graph_file = Path(output_dir) / "road_network.graphml"
        if graph_file.exists():
            logger.info("检测到已存在的路网文件，直接加载")
            graph = extractor.load_network(graph_file)
            return graph, {'graph': graph_file}
    
    # 下载路网
    graph = extractor.download_network()
    
    # 获取强连通分量
    graph = extractor.get_largest_strongly_connected_component()
    
    # 添加速度和行程时间
    extractor.add_edge_speeds()
    
    # 提取节点和边数据
    extractor.extract_nodes_and_edges()
    
    # 获取统计信息
    stats = extractor.get_network_stats()
    
    # 保存路网
    if output_dir:
        saved_files = extractor.save_network(output_dir)
        
        # 可视化并保存
        try:
            vis_file = Path(output_dir) / "network_visualization.png"
            extractor.visualize_network(output_file=vis_file)
        except Exception as e:
            logger.warning(f"可视化失败: {str(e)}")
        
        return graph, saved_files
    
    return graph, {}


if __name__ == "__main__":
    # 测试路网提取
    from src.utils.config import get_config
    
    config = get_config()
    network_config = config.get_network_config()
    output_dir = config.get_data_dir("processed")
    
    print("=== 开始提取OSM路网 ===")
    graph, files = extract_osm_network(
        network_config, 
        output_dir=output_dir,
        force_download=False
    )
    
    print("\n=== 路网提取完成 ===")
    print(f"节点数: {len(graph.nodes)}")
    print(f"边数: {len(graph.edges)}")
    print(f"\n保存的文件:")
    for key, path in files.items():
        print(f"  {key}: {path}")

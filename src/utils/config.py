"""
配置管理工具
用于加载和管理系统配置
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config/config.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项（支持点号分隔的嵌套路径）
        
        Args:
            key_path: 配置项路径，如 "network.location.city"
            default: 默认值
        
        Returns:
            配置值
        
        Examples:
            >>> config = ConfigManager()
            >>> city = config.get("network.location.city")
            >>> radius = config.get("network.location.radius", 2000)
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_network_config(self) -> Dict[str, Any]:
        """获取路网配置"""
        return self.config.get('network', {})
    
    def get_distance_matrix_config(self) -> Dict[str, Any]:
        """获取距离矩阵配置"""
        return self.config.get('distance_matrix', {})
    
    def get_order_generation_config(self) -> Dict[str, Any]:
        """获取订单生成配置"""
        return self.config.get('order_generation', {})
    
    def get_courier_config(self) -> Dict[str, Any]:
        """获取骑手配置"""
        return self.config.get('courier', {})
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        return self.config.get('simulation', {})
    
    def get_random_seed(self) -> int:
        """获取随机种子"""
        return self.config.get('random_seed', 42)
    
    def get_project_root(self) -> Path:
        """获取项目根目录"""
        return Path(__file__).parent.parent.parent
    
    def get_data_dir(self, subdir: str = "") -> Path:
        """
        获取数据目录路径
        
        Args:
            subdir: 子目录名称（raw, processed, orders等）
        
        Returns:
            数据目录路径
        """
        data_dir = self.get_project_root() / "data"
        if subdir:
            data_dir = data_dir / subdir
        
        # 确保目录存在
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def update(self, key_path: str, value: Any) -> None:
        """
        更新配置项
        
        Args:
            key_path: 配置项路径
            value: 新值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，默认覆盖原配置文件
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


# 全局配置实例（单例模式）
_global_config: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置管理器实例
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    
    return _global_config


if __name__ == "__main__":
    # 测试配置管理器
    config = ConfigManager()
    
    print("=== 配置管理器测试 ===")
    print(f"项目根目录: {config.get_project_root()}")
    print(f"城市: {config.get('network.location.city')}")
    print(f"半径: {config.get('network.location.radius')} 米")
    print(f"总订单数: {config.get('order_generation.total_orders')}")
    print(f"骑手数量: {config.get('courier.num_couriers')}")
    print(f"随机种子: {config.get_random_seed()}")

"""
调度器模块
实现各种订单分配和路径规划策略

注意：使用懒加载避免可选依赖问题（如alns、stable-baselines3）
"""

# 懒加载：不在此处导入，在需要时由各模块直接导入
# 这样可以避免alns、stable-baselines3等可选依赖导致的导入错误

__all__ = ['GreedyDispatcher', 'ORToolsDispatcher', 'ALNSDispatcher', 'RLDispatcher']

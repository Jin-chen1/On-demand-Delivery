"""
Day 4 快速测试脚本
运行 OR-Tools VRP 调度器测试并验证功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from test_day4_ortools import test_ortools_dispatcher

if __name__ == "__main__":
    print("="*70)
    print("Day 4: OR-Tools VRP Dispatcher 测试")
    print("="*70)
    print("\n开始测试...\n")
    
    success = test_ortools_dispatcher()
    
    print("\n" + "="*70)
    if success:
        print("✅ 测试通过！Day 4 OR-Tools 调度器工作正常")
        print("\n核心功能验证:")
        print("  ✓ VRP 模型构建成功")
        print("  ✓ OR-Tools 求解成功")
        print("  ✓ 订单分配和路径规划成功")
        print("  ✓ 滚动时域调度机制工作正常")
    else:
        print("⚠️  测试未完全通过，请查看日志了解详情")
    print("="*70)
    
    sys.exit(0 if success else 1)

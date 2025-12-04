"""
Day 3 快速测试脚本
运行Greedy调度器测试并验证功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from tests.test_day3_greedy import test_greedy_dispatcher

if __name__ == "__main__":
    print("="*70)
    print("Day 3: Greedy Dispatcher 测试")
    print("="*70)
    print("\n开始测试...\n")
    
    success = test_greedy_dispatcher()
    
    print("\n" + "="*70)
    if success:
        print("✅ 测试通过！Day 3 调度器工作正常")
    else:
        print("⚠️  测试未完全通过，请查看日志了解详情")
    print("="*70)
    
    sys.exit(0 if success else 1)

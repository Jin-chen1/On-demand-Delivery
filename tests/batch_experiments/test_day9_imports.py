"""
快速测试Day 9模块导入是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有必要的导入"""
    print("Testing Day 9 module imports...")
    print("-" * 50)
    
    try:
        # 测试核心库
        print("[OK] Testing standard libraries...")
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("  - pandas, numpy, matplotlib, seaborn: OK")
        
        # 测试项目模块
        print("\n[OK] Testing project modules...")
        from utils.result_analyzer import ResultCollector, DataAnalyzer
        print("  - result_analyzer: OK")
        
        from utils.visualization import ExperimentVisualizer
        print("  - visualization: OK")
        
        from utils.experiment_task import ExperimentTask, ExperimentConfigManager
        print("  - experiment_task: OK")
        
        from utils.experiment_runner import ExperimentRunner
        print("  - experiment_runner: OK")
        
        print("\n" + "=" * 50)
        print("[SUCCESS] All import tests passed! Day 9 modules are ready.")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("\nPlease check:")
        print("  1. Install all dependencies: pip install -r requirements.txt")
        print("  2. Run from project root directory")
        return False
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

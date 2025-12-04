"""
Day 7 一键运行脚本
自动完成对比测试和可视化生成
"""

import sys
from pathlib import Path
import subprocess
import time

project_root = Path(__file__).parent.parent  # 项目根目录
tests_dir = Path(__file__).parent  # tests目录
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str):
    """运行命令并显示进度"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} 完成！(耗时: {elapsed:.1f}秒)")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} 失败！(耗时: {elapsed:.1f}秒)")
        print(f"错误信息: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} 出错！(耗时: {elapsed:.1f}秒)")
        print(f"错误信息: {e}")
        return False


def main():
    """主函数"""
    print("="*70)
    print("Day 7: OR-Tools vs ALNS 完整测试流程")
    print("="*70)
    print("\n本脚本将自动执行以下步骤：")
    print("  1. 运行对比测试 (test_day7_comparison.py)")
    print("  2. 生成可视化图表 (visualize_day7_results.py)")
    print("\n预计总耗时：10-15分钟（取决于硬件性能）")
    
    input("\n按Enter键开始...")
    
    total_start = time.time()
    
    # 步骤1：运行对比测试
    test_script = tests_dir / "test_day7_comparison.py"
    if not test_script.exists():
        print(f"\n错误：测试脚本不存在: {test_script}")
        return False
    
    success = run_command(
        [sys.executable, str(test_script)],
        "步骤1: 运行OR-Tools vs ALNS对比测试"
    )
    
    if not success:
        print("\n⚠️  对比测试失败，跳过可视化步骤")
        return False
    
    # 步骤2：生成可视化图表
    vis_script = tests_dir / "visualize_day7_results.py"
    if not vis_script.exists():
        print(f"\n错误：可视化脚本不存在: {vis_script}")
        return False
    
    success = run_command(
        [sys.executable, str(vis_script)],
        "步骤2: 生成可视化图表"
    )
    
    if not success:
        print("\n⚠️  可视化生成失败")
        return False
    
    # 总结
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*70)
    print("🎉 Day 7 完整测试流程执行完成！")
    print("="*70)
    print(f"\n总耗时: {total_elapsed/60:.1f}分钟")
    
    # 显示输出位置
    outputs_dir = project_root / "outputs" / "day7_comparison"
    if outputs_dir.exists():
        result_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                            key=lambda x: x.name, reverse=True)
        if result_dirs:
            latest_dir = result_dirs[0]
            print(f"\n📊 结果保存位置:")
            print(f"  {latest_dir}")
            print(f"\n📈 可视化图表位置:")
            print(f"  {latest_dir / 'visualizations'}")
            print(f"\n生成的文件:")
            print(f"  - comparison_results.json (详细数据)")
            print(f"  - comparison_table.csv (对比表格)")
            print(f"  - visualizations/fig1_solve_time_comparison.png")
            print(f"  - visualizations/fig2_performance_comparison.png")
            print(f"  - visualizations/fig3_radar_comparison.png")
            print(f"  - visualizations/fig4_summary_table.png")
    
    print("\n" + "="*70)
    print("📝 查看详细总结: docs/day7_completion_summary.md")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
下载菜鸟LaDe（Last-mile Delivery）数据集
来源: https://huggingface.co/datasets/Cainiao-AI/LaDe

LaDe数据集包含:
- 中国杭州和上海的真实配送数据
- 包含GPS轨迹、配送时间等信息
- 适合即时配送场景研究
"""
from datasets import load_dataset
from pathlib import Path
import pandas as pd
import os

# 设置下载目录
output_dir = Path("D:/0On-demand Delivery/data/lade")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("下载菜鸟LaDe数据集")
print("来源: https://huggingface.co/datasets/Cainiao-AI/LaDe")
print("=" * 60)

# 下载数据集
print("\n正在下载数据集...")
try:
    dataset = load_dataset("Cainiao-AI/LaDe")
    print(f"数据集加载成功!")
    print(f"可用分割: {list(dataset.keys())}")
    
    # 查看数据集信息
    for split_name, split_data in dataset.items():
        print(f"\n{split_name}:")
        print(f"  样本数: {len(split_data)}")
        print(f"  列: {split_data.column_names}")
        
        # 保存为CSV
        df = split_data.to_pandas()
        csv_path = output_dir / f"lade_{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  已保存: {csv_path}")
        
        # 显示前几行
        print(f"  前3行预览:")
        print(df.head(3).to_string())
        
except Exception as e:
    print(f"下载失败: {e}")
    print("\n尝试直接下载...")
    
    # 备选方案：使用huggingface_hub直接下载
    from huggingface_hub import hf_hub_download, list_repo_files
    
    repo_id = "Cainiao-AI/LaDe"
    print(f"\n列出仓库文件: {repo_id}")
    
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        print(f"文件列表: {files}")
        
        for file in files:
            if file.endswith(('.csv', '.parquet', '.json')):
                print(f"\n下载: {file}")
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type="dataset",
                    local_dir=output_dir
                )
                print(f"已保存: {local_path}")
    except Exception as e2:
        print(f"备选方案也失败: {e2}")

print("\n" + "=" * 60)
print("下载完成!")
print(f"数据保存在: {output_dir}")
print("=" * 60)

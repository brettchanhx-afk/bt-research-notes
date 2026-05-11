import pandas as pd
import numpy as np

file_path = 'data/多资产行情序列.csv'

print("=== 检查CSV文件结构 ===")

try:
    # 尝试不同编码
    for enc in ['gbk', 'gb18030', 'utf-8', 'utf-8-sig']:
        try:
            df = pd.read_csv(file_path, index_col=0, encoding=enc, nrows=5)
            print(f"\n成功使用编码: {enc}")
            print(f"列名: {df.columns.tolist()}")
            print(f"前5行数据:\n{df.head()}")
            break
        except Exception as e:
            print(f"编码 {enc} 失败: {e}")
except Exception as e:
    print(f"读取错误: {e}")

print("\n=== 跳过前3行 ===")
try:
    df = pd.read_csv(file_path, index_col=0, encoding='gbk', header=0)
    print(f"形状: {df.shape}")
    print(f"索引类型: {type(df.index)}")
    print(f"第一行: {df.iloc[0].tolist()}")
    print(f"第二行: {df.iloc[1].tolist()}")
except Exception as e:
    print(f"错误: {e}")
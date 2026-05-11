import sys
print("测试输出", file=sys.stdout)
print("测试输出到标准输出")

import os
os.makedirs('output', exist_ok=True)

import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df.to_csv('output/test.csv', index=False)
print("文件已保存")
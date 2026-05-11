import sys
print("=== 测试输出 ===")
print(f"Python版本: {sys.version}")
print(f"标准输出编码: {sys.stdout.encoding}")
print("测试成功！")

# 写入文件测试
with open('output/test_output.txt', 'w', encoding='utf-8') as f:
    f.write("测试输出内容\n")
print("文件写入成功")

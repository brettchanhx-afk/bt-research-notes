import os
import sys

try:
    import pdfplumber
except ImportError:
    print("正在安装pdfplumber...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
    import pdfplumber

pdf_path = r"d:\Documents\trae_projects\asset_alloc_practical_method\参考研报\国泰君安_大类资产配置量化模型研究系列之六：大类资产配置策略落地方法研究-2024-01-11.pdf"
txt_path = r"d:\Documents\trae_projects\asset_alloc_practical_method\国泰君安_大类资产配置策略落地方法研究.txt"

print(f"正在解析PDF文件: {pdf_path}")

all_text = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"正在处理第 {i+1} 页...")
        text = page.extract_text()
        if text:
            all_text.append(f"=== 第 {i+1} 页 ===\n")
            all_text.append(text)
            all_text.append("\n\n")

full_text = "\n".join(all_text)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"解析完成！文本已保存到: {txt_path}")
print(f"总页数: {len(pdf.pages)}")
print(f"总字符数: {len(full_text)}")

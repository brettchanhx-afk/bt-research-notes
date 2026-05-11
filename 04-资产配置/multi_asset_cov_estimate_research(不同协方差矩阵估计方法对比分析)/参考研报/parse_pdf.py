import pdfplumber
import os

pdf_path = r"d:\Documents\trae_projects\multi_asset_cov_estimate_research\参考研报\不同协方差矩阵估计方法对比分析-大类资产配置量化模型研究系列之五.pdf"
output_path = r"d:\Documents\trae_projects\multi_asset_cov_estimate_research\参考研报\不同协方差矩阵估计方法对比分析-大类资产配置量化模型研究系列之五.txt"

all_text = []

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text:
            all_text.append(f"=== 第 {page_num} 页 ===\n{text}\n")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_text))

print(f"PDF解析完成！共处理 {len(pdf.pages)} 页")
print(f"文本已保存到: {output_path}")
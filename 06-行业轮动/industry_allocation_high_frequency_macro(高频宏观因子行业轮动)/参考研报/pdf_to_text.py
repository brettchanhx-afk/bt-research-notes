import pdfplumber
import os

pdf_path = r"d:\Documents\trae_projects\industry_allocation_high_frequency_macro\参考研报\2023-06-10_华泰证券_行业配置策略：高频宏观因子.pdf"
output_path = r"d:\Documents\trae_projects\industry_allocation_high_frequency_macro\参考研报\2023-06-10_华泰证券_行业配置策略：高频宏观因子.txt"

with pdfplumber.open(pdf_path) as pdf:
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                f.write(f"\n--- 第 {page_num} 页 ---\n\n")
                f.write(text)
                f.write("\n")

print(f"PDF已成功解析为文本文件: {output_path}")
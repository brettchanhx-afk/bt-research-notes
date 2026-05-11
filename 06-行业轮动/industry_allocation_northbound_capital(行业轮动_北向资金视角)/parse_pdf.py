import pdfplumber
import sys
from pathlib import Path

pdf_path = r"d:\Documents\trae_projects\industry_allocation_northbound_capital\参考研报\2022-10-27_华泰证券_金工深度研究：析精剖微-机构拆解看北向资金.pdf"
output_path = r"d:\Documents\trae_projects\industry_allocation_northbound_capital\参考研报\2022-10-27_华泰证券_金工深度研究：析精剖微-机构拆解看北向资金.txt"

try:
    print(f"正在解析PDF文件: {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                all_text.append(f"=== 第 {i+1} 页 ===\n{text}\n")
                print(f"  已提取第 {i+1}/{len(pdf.pages)} 页")

    full_text = "\n".join(all_text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"\nPDF解析完成！文本已保存至: {output_path}")
    print(f"总字符数: {len(full_text)}")

except Exception as e:
    print(f"解析失败: {e}")
    sys.exit(1)

import pdfplumber
import os

pdf_path = r"d:\Documents\trae_projects\hierarchical_cluster_risk_parity\参考研报\国泰君安_基于层次聚类改进风险平价策略_2024-10-31.pdf"
output_path = r"d:\Documents\trae_projects\hierarchical_cluster_risk_parity\参考研报\pdf_content.txt"

all_text = []

with pdfplumber.open(pdf_path) as pdf:
    print(f"PDF总页�? {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text.append(f"\n{'='*60}\n�?{i+1} 页\n{'='*60}\n")
            all_text.append(text)
            print(f"�?{i+1} 页已提取")

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(''.join(all_text))

print(f"\nPDF内容已保存到: {output_path}")
print(f"总字符数: {sum(len(t) for t in all_text)}")

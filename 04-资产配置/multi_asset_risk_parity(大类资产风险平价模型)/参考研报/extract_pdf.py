import pdfplumber
import os

pdf_path = r"d:\Documents\trae_projects\multi_asset_risk_parity\参考研报\桥水全天候策略和风险平价模型全解析-大类资产配置量化模型研究系列之三.pdf"
output_path = r"d:\Documents\trae_projects\multi_asset_risk_parity\参考研报\桥水全天候策略和风险平价模型全解析-大类资产配置量化模型研究系列之三.txt"

all_text = []

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        text = page.extract_text()
        if text:
            all_text.append(f"===== 第 {page_num} 页 =====\n{text}")

full_text = "\n\n".join(all_text)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

print(f"已提取 {len(pdf.pages)} 页内容到: {output_path}")
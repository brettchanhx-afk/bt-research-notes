import pdfplumber

pdf_path = r'd:\Documents\trae_projects\decay_weighted_pca_rp\参考研报\天风证券_2017-11-17_资产配置策略研究之二：引入衰减加权和趋势跟踪的主成分风险平价模型研究.pdf'
txt_path = pdf_path.replace('.pdf', '.txt')

with pdfplumber.open(pdf_path) as pdf:
    all_text = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text.append(f'=== 第 {i+1} 页 ===\n{text}\n')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

print(f'完成！文本已保存到: {txt_path}')
print(f'共提取 {len(pdf.pages)} 页')
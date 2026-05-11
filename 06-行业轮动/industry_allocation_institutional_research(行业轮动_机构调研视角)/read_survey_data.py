import pandas as pd
import os

file_path = r'd:\Documents\trae_projects\industry_allocation_institutional_research\data\全部A股_当月被调研总次数2010_2026.xlsx'
output_path = r'd:\Documents\trae_projects\industry_allocation_institutional_research\output\survey_data_preview.txt'

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"Testing read_excel with nrows=5...\n")
    f.flush()

    df = pd.read_excel(file_path, nrows=5)
    f.write(f'Shape: {df.shape}\n')
    f.write(f'Columns: {list(df.columns)}\n')
    f.write(df.to_string())
    f.write('\n')

print(f"Preview saved to: {output_path}")
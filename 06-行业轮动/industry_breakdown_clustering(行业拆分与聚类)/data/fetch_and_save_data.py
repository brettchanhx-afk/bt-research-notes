"""
数据获取与保存脚本
使用akshare获取申万行业数据并保存到data文件夹
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

SW_INDUSTRY_CODES = {
    '农林牧渔': '801010',
    '基础化工': '801030',
    '钢铁': '801040',
    '有色金属': '801050',
    '电子': '801080',
    '汽车': '801880',
    '家用电器': '801110',
    '食品饮料': '801120',
    '纺织服饰': '801130',
    '轻工制造': '801140',
    '医药生物': '801150',
    '公用事业': '801160',
    '交通运输': '801170',
    '房地产': '801180',
    '商贸零售': '801200',
    '社会服务': '801210',
    '银行': '801780',
    '非银金融': '801790',
    '综合': '801230',
    '建筑材料': '801710',
    '建筑装饰': '801720',
    '电力设备': '801730',
    '机械设备': '801890',
    '国防军工': '801740',
    '计算机': '801750',
    '传媒': '801760',
    '通信': '801770',
    '煤炭': '801950',
    '石油石化': '801960',
    '环保': '801970'
}

def get_sw_industry_history(symbol_code, start_year=2014, end_year=2020):
    """获取申万行业指数历史数据"""
    try:
        df = ak.index_hist_sw(symbol=symbol_code)
        if df is None or df.empty:
            return pd.DataFrame()

        df.columns = df.columns.str.strip()

        if '日期' not in df.columns:
            return pd.DataFrame()

        df['日期'] = pd.to_datetime(df['日期'])
        df = df[(df['日期'].dt.year >= start_year) & (df['日期'].dt.year <= end_year)]

        if df.empty:
            return pd.DataFrame()

        df = df.set_index('日期')
        df = df.sort_index()

        df = df.rename(columns={
            '代码': 'index_code',
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount'
        })

        cols_to_keep = ['index_code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        df = df[cols_to_keep]

        return df
    except Exception as e:
        print(f"获取{symbol_code}失败: {e}")
        return pd.DataFrame()

def main():
    print("=" * 70)
    print("开始获取申万行业数据")
    print("=" * 70)

    START_YEAR = 2014
    END_YEAR = 2020

    all_industry_data = {}
    industry_list = []
    failed_industries = []

    print(f"\n数据时间范围: {START_YEAR} - {END_YEAR}")
    print(f"共 {len(SW_INDUSTRY_CODES)} 个申万一级行业\n")

    for i, (industry, code) in enumerate(SW_INDUSTRY_CODES.items(), 1):
        print(f"[{i}/{len(SW_INDUSTRY_CODES)}] 获取 {industry} ({code})...", end=" ")

        df = get_sw_industry_history(code, START_YEAR, END_YEAR)

        if not df.empty:
            all_industry_data[industry] = df
            industry_list.append(industry)
            print(f"成功! {len(df)} 条记录")
        else:
            failed_industries.append(industry)
            print("失败!")

    print(f"\n成功获取 {len(all_industry_data)} 个行业的数据")

    if failed_industries:
        print(f"失败行业: {', '.join(failed_industries)}")

    if all_industry_data:
        print("\n正在保存行业指数数据...")

        returns_dict = {}
        for industry, df in all_industry_data.items():
            if 'close' in df.columns:
                returns_dict[industry] = df['close'].pct_change().dropna()

        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            returns_df.to_csv(os.path.join(DATA_DIR, 'industry_returns.csv'))
            print(f"  - 行业收益率数据: industry_returns.csv ({returns_df.shape})")

        for industry, df in all_industry_data.items():
            df.to_csv(os.path.join(DATA_DIR, f'industry_{industry}.csv'))
        print(f"  - 各行业指数数据已保存")

        print("\n正在生成行业信息表...")
        industry_info = []
        for industry, df in all_industry_data.items():
            industry_info.append({
                'industry': industry,
                'industry_code': SW_INDUSTRY_CODES.get(industry, ''),
                'n_records': len(df),
                'start_date': df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else '',
                'end_date': df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else ''
            })

        industry_info_df = pd.DataFrame(industry_info)
        industry_info_df.to_csv(os.path.join(DATA_DIR, 'industry_info.csv'), index=False)
        print(f"行业信息表已保存: industry_info.csv")

        print("\n正在保存配置文件...")
        config_data = []
        for industry, code in SW_INDUSTRY_CODES.items():
            config_data.append({
                'industry': industry,
                'industry_code': code
            })
        config_df = pd.DataFrame(config_data)
        config_df.to_csv(os.path.join(DATA_DIR, 'sw_industry_codes.csv'), index=False)
        print(f"配置文件已保存: sw_industry_codes.csv")

        print("\n正在获取行业列表信息...")
        try:
            sw_info = ak.sw_index_first_info()
            sw_info.to_csv(os.path.join(DATA_DIR, 'sw_first_info.csv'), index=False)
            print(f"申万一级行业信息已保存: sw_first_info.csv")
        except Exception as e:
            print(f"获取申万行业信息失败: {e}")

    print("\n" + "=" * 70)
    print("数据获取完成!")
    print("=" * 70)
    print(f"\n数据保存位置: {DATA_DIR}")
    print("\n生成的文件列表:")

    for f in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, f)
        fsize = os.path.getsize(fpath)
        print(f"  - {f} ({fsize:,} bytes)")

    return all_industry_data

if __name__ == "__main__":
    main()

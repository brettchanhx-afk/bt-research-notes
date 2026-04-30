# -*- coding: utf-8 -*-
"""
数据加载模块：ETF数据获取
支持多数据源：efinance > tushare > akshare > baostock
"""

import warnings
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==================== 数据源初始化 ====================
def get_tushare_api():
    """获取Tushare API"""
    try:
        import tushare as ts
        from config import TUSHARE_CONFIG
        pro = ts.pro_api(TUSHARE_CONFIG['token'])
        pro._DataApi__http_url = TUSHARE_CONFIG['api_url']
        return pro
    except Exception as e:
        print(f"Tushare初始化失败: {e}")
        return None

def get_akshare_data():
    """获取akshare数据访问接口"""
    try:
        import akshare as ak
        return ak
    except Exception as e:
        print(f"Akshare导入失败: {e}")
        return None

def get_baostock_connection():
    """获取baostock连接"""
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == '0':
            return bs
        return None
    except Exception as e:
        print(f"Baostock连接失败: {e}")
        return None

# ==================== ETF基础信息获取 ====================
def get_etf_basic_info():
    """获取ETF基本信息"""
    etf_list = []
    
    # 方法1: 使用tushare获取ETF基础信息
    try:
        pro = get_tushare_api()
        if pro:
            print("尝试使用tushare获取ETF列表...")
            df = pro.fund_basic(market='E')
            if df is not None and len(df) > 0:
                # 筛选有benchmark的（这些是真正的ETF）
                etfs = df[df['benchmark'].notna()].copy()
                # 【关键】筛选产品名包含ETF的（真正的ETF产品）
                stock_etfs = etfs[etfs['name'].str.contains('ETF', na=False) & etfs['fund_type'].str.contains('股票', na=False)]
                print(f"成功获取Tushare股票ETF: {len(stock_etfs)}条")
                
                # 标准化列名
                stock_etfs = stock_etfs.rename(columns={
                    'ts_code': 'fund_code',
                    'name': 'fund_name',
                    'benchmark': 'benchmark'
                })
                
                # 提取index_code（从benchmark字符串中解析）
                def extract_index_code(benchmark):
                    if pd.isna(benchmark):
                        return None
                    # 解析benchmark字符串，如 "沪深300指数*80%+中证全债指数*20%" -> "000300.SH"
                    bm = str(benchmark)
                    # 常见指数映射
                    index_map = {
                        '沪深300': '000300.SH',
                        '上证50': '000016.SH',
                        '中证500': '000905.SH',
                        '中证100': '000903.SH',
                        '中证1000': '000852.SH',
                        '中证A500': '000510.SH',
                        '中证800': '000906.SH',
                        '中证1000': '000852.SH',
                        '创业板': '399006.SZ',
                        '科创板': '000688.SH',
                        '上证180': '000010.SH',
                        '深证100': '399004.SZ',
                    }
                    for name, code in index_map.items():
                        if name in bm:
                            return code
                    return None
                
                stock_etfs['index_code'] = stock_etfs['benchmark'].apply(extract_index_code)
                stock_etfs['etf_type'] = stock_etfs['fund_type'].apply(lambda x: '股票ETF' if pd.notna(x) else '其他')
                
                etf_list.append(stock_etfs)
    except Exception as e:
        print(f"tushare获取失败: {e}")
    
    # 如果所有数据源都失败，使用模拟数据
    if len(etf_list) == 0:
        print("使用模拟ETF数据进行演示...")
        return generate_mock_etf_data()
    
    # 合并所有ETF数据
    result = pd.concat(etf_list, ignore_index=True) if etf_list else pd.DataFrame()
    return result

def generate_mock_etf_data():
    """生成模拟ETF数据进行演示"""
    np.random.seed(42)
    
    # 宽基ETF
    wide_base_etfs = [
        ('510050.SH', '华夏上证50ETF', '000016.SH', '规模风格', 1486.35),
        ('510310.SH', '易方达沪深300ETF', '000300.SH', '规模风格', 2490.6),
        ('510330.SH', '华夏沪深300ETF', '000300.SH', '规模风格', 1681.0),
        ('512050.SH', '华夏中证A500ETF', '000510.SH', '规模风格', 210.19),
        ('515800.SH', '汇添富中证800ETF', '000906.SH', '规模风格', 58.06),
        ('588080.SH', '易方达上证科创板50ETF', '000688.SH', '规模风格', 608.60),
        ('510180.SH', '华安上证180ETF', '000010.SH', '规模风格', 212.66),
    ]
    
    # 行业ETF
    industry_etfs = [
        ('512660.SH', '国泰中证军工ETF', '399967.SZ', '行业', 106.91),
        ('512800.SH', '华宝中证银行ETF', '399986.SZ', '行业', 80.18),
        ('515790.SH', '华泰柏瑞中证光伏产业ETF', '931151.CSI', '行业', 95.86),
        ('512010.SH', '易方达沪深300医药卫生ETF', '000913.SH', '行业', 232.19),
        ('512980.SH', '广发中证传媒ETF', '399971.SZ', '行业', 28.14),
        ('159819.SZ', '易方达中证人工智能ETF', '930713.CSI', '行业', 135.07),
        ('516310.SH', '易方达中证银行ETF', '399986.SZ', '行业', 11.55),
        ('512390.SH', '平安MSCI中国A股低波动ETF', '707918.MI', '行业', 1.80),
    ]
    
    # 债券ETF
    bond_etfs = [
        ('511360.SH', '海富通中证短融ETF', 'h11014.SHI', '债券', 263.59),
        ('511220.SH', '海富通上证城投债ETF', 'h11098.SHI', '债券', 146.67),
        ('511260.SH', '国泰上证10年期国债ETF', 'h11077.SHI', '债券', 28.04),
    ]
    
    # 商品ETF
    commodity_etfs = [
        ('159980.SZ', '大成有色金属期货ETF', 'IMCI.SHF', '商品', 9.33),
        ('159981.SZ', '华夏饲料豆粕期货ETF', 'M99.SHF', '商品', 5.12),
    ]
    
    # 跨境ETF
    cross_border_etfs = [
        ('513050.SH', '易方达中证海外互联ETF', 'h30533.CSI', '跨境', 388.06),
        ('159605.SZ', '广发中证海外中国互联网30ETF', '930604.CSI', '跨境', 62.41),
        ('513360.SH', '博时中证全球中国教育ETF', '931456.CSI', '跨境', 5.74),
    ]
    
    all_etfs = wide_base_etfs + industry_etfs + bond_etfs + commodity_etfs + cross_border_etfs
    
    etf_data = []
    for code, name, index_code, etf_type, scale in all_etfs:
        # 管理费率
        if etf_type == '债券':
            mgmt_fee = 0.003
            custody_fee = 0.001
        elif etf_type == '商品':
            mgmt_fee = 0.005
            custody_fee = 0.002
        else:
            mgmt_fee = 0.005
            custody_fee = 0.001
        
        etf_data.append({
            'fund_code': code,
            'fund_name': name,
            'index_code': index_code,
            'etf_type': etf_type,
            'scale': scale,
            'mgmt_fee': mgmt_fee,
            'custody_fee': custody_fee,
            'listing_date': '2015-01-01',
        })
    
    return pd.DataFrame(etf_data)

# ==================== ETF历史数据获取 ====================
def get_etf_historical_data(etf_code, start_date=None, end_date=None):
    """获取ETF历史净值数据"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    df = None
    
    # 方法1: 使用efinance
    try:
        import efinance as ef
        print(f"尝试使用efinance获取{etf_code}数据...")
        df = ef.fund.get_quote_history(etf_code, beg=start_date, end=end_date)
        if df is not None and len(df) > 0:
            print(f"  efinance成功获取: {len(df)}条")
            return df
    except Exception as e:
        print(f"  efinance失败: {e}")
    
    # 方法2: 使用akshare
    try:
        import akshare as ak
        print(f"尝试使用akshare获取{etf_code}数据...")
        # 去除后缀获取纯代码
        code = etf_code.replace('.SH', '').replace('.SZ', '')
        df = ak.fund_etf_hist_em(symbol=code)
        if df is not None and len(df) > 0:
            print(f"  akshare成功获取: {len(df)}条")
            return df
    except Exception as e:
        print(f"  akshare失败: {e}")
    
    # 方法3: 使用tushare
    try:
        pro = get_tushare_api()
        if pro:
            print(f"尝试使用tushare获取{etf_code}数据...")
            df = pro.fund_nav(ts_code=etf_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                print(f"  tushare成功获取: {len(df)}条")
                return df
    except Exception as e:
        print(f"  tushare失败: {e}")
    
    return None

# ==================== 指数成分股获取 ====================
def get_index_constituents(index_code):
    """获取指数成分股"""
    
    # 方法1: 使用tushare
    try:
        pro = get_tushare_api()
        if pro:
            print(f"尝试使用tushare获取{index_code}成分股...")
            df = pro.index_weight(index_code=index_code)
            if df is not None and len(df) > 0:
                print(f"  tushare成功获取: {len(df)}条")
                return df
    except Exception as e:
        print(f"  tushare失败: {e}")
    
    # 方法2: 使用akshare
    try:
        import akshare as ak
        print(f"尝试使用akshare获取{index_code}成分股...")
        # 指数成分股权重
        df = ak.index_weight_cons(symbol=index_code)
        if df is not None and len(df) > 0:
            print(f"  akshare成功获取: {len(df)}条")
            return df
    except Exception as e:
        print(f"  akshare失败: {e}")
    
    # 返回模拟数据
    return generate_mock_constituents(index_code)

def generate_mock_constituents(index_code):
    """生成模拟成分股数据"""
    np.random.seed(hash(index_code) % 2**32)
    
    # 模拟成分股列表
    num_stocks = np.random.randint(20, 50)
    stocks = []
    
    stock_pool = [
        ('600519.SH', '贵州茅台', 0.08),
        ('000858.SZ', '五粮液', 0.05),
        ('600036.SH', '招商银行', 0.05),
        ('601318.SH', '中国平安', 0.04),
        ('000333.SZ', '美的集团', 0.03),
        ('300750.SZ', '宁德时代', 0.04),
        ('600900.SH', '长江电力', 0.03),
        ('601166.SH', '兴业银行', 0.02),
        ('002594.SZ', '比亚迪', 0.03),
        ('600887.SH', '伊利股份', 0.02),
        ('000001.SZ', '平安银行', 0.02),
        ('601398.SH', '工商银行', 0.03),
        ('600030.SH', '中信证券', 0.02),
        ('601888.SH', '中国中免', 0.02),
        ('002415.SZ', '海康威视', 0.02),
    ]
    
    # 随机选择成分股
    selected = np.random.choice(len(stock_pool), min(num_stocks, len(stock_pool)), replace=False)
    total_weight = 0
    
    for i in selected:
        code, name, base_weight = stock_pool[i]
        # 添加随机波动
        weight = base_weight * np.random.uniform(0.8, 1.2)
        total_weight += weight
        stocks.append({
            'index_code': index_code,
            'con_code': code,
            'con_name': name,
            'weight': weight,
        })
    
    # 归一化权重
    if stocks:
        total = sum(s['weight'] for s in stocks)
        for s in stocks:
            s['weight'] = s['weight'] / total
    
    return pd.DataFrame(stocks)

# ==================== 基准指数数据获取 ====================
def get_benchmark_data(benchmark_code='000300.SH', start_date=None, end_date=None):
    """获取基准指数数据（沪深300）"""
    
    if start_date is None:
        start_date = '20140101'
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    df = None
    
    # 方法1: 使用akshare
    try:
        import akshare as ak
        print(f"尝试使用akshare获取{benchmark_code}数据...")
        df = ak.stock_zh_index_daily(symbol=benchmark_code.replace('.SH', '').replace('.SZ', ''))
        if df is not None and len(df) > 0:
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date[:4]+'-'+start_date[4:6]+'-'+start_date[6:]) & 
                     (df['date'] <= end_date[:4]+'-'+end_date[4:6]+'-'+end_date[6:])]
            print(f"  akshare成功获取: {len(df)}条")
            return df
    except Exception as e:
        print(f"  akshare失败: {e}")
    
    # 方法2: 使用efinance
    try:
        import efinance as ef
        print(f"尝试使用efinance获取{benchmark_code}数据...")
        df = ef.stock.get_quote_history(benchmark_code, beg=start_date, end=end_date)
        if df is not None and len(df) > 0:
            print(f"  efinance成功获取: {len(df)}条")
            return df
    except Exception as e:
        print(f"  efinance失败: {e}")
    
    return None

# ==================== 股票财务数据获取 ====================
def get_stock_financial_data(stock_code, start_date=None, end_date=None):
    """获取股票财务数据（ROE、营收等）"""
    
    # 方法1: 使用tushare
    try:
        pro = get_tushare_api()
        if pro:
            print(f"尝试使用tushare获取{stock_code}财务数据...")
            df = pro.fina_indicator(ts_code=stock_code, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                print(f"  tushare成功获取: {len(df)}条")
                return df
    except Exception as e:
        print(f"  tushare失败: {e}")
    
    # 返回模拟数据
    return generate_mock_financial_data(stock_code)

def generate_mock_financial_data(stock_code):
    """生成模拟财务数据"""
    np.random.seed(hash(stock_code) % 2**32)
    
    # 模拟季度财务数据
    dates = pd.date_range(end=datetime.now(), periods=8, freq='Q')
    data = []
    
    for date in dates:
        data.append({
            'ts_code': stock_code,
            'ann_date': date.strftime('%Y%m%d'),
            'roe_ttm': np.random.uniform(5, 25),  # ROE_TTM
            'revenue_yoy': np.random.uniform(-10, 50),  # 营收同比
            'net_profit_yoy': np.random.uniform(-20, 60),  # 净利润同比
        })
    
    return pd.DataFrame(data)

# ==================== 数据保存和加载 ====================
def save_data(df, filename, data_dir=None):
    """保存数据到CSV"""
    if data_dir is None:
        from config import DATA_DIR
        data_dir = DATA_DIR
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据已保存: {filepath}")
    return filepath

def load_data(filename, data_dir=None):
    """从CSV加载数据"""
    if data_dir is None:
        from config import DATA_DIR
        data_dir = DATA_DIR
    
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, encoding='utf-8-sig')
    return None

# ==================== 主数据获取函数 ====================
def load_all_etf_data():
    """加载所有ETF数据"""
    from config import DATA_DIR
    
    print("=" * 60)
    print("开始加载ETF数据...")
    print("=" * 60)
    
    # 尝试加载缓存数据
    cache_file = os.path.join(DATA_DIR, 'etf_basic_info.csv')
    if os.path.exists(cache_file):
        print("发现缓存数据，加载中...")
        etf_df = pd.read_csv(cache_file, encoding='utf-8-sig')
        print(f"已加载 {len(etf_df)} 只ETF信息")
        return etf_df
    
    # 获取ETF基础信息
    etf_df = get_etf_basic_info()
    
    if etf_df is not None and len(etf_df) > 0:
        save_data(etf_df, 'etf_basic_info.csv', DATA_DIR)
        print(f"成功获取 {len(etf_df)} 只ETF信息")
    else:
        print("无法获取ETF数据，使用模拟数据")
        etf_df = generate_mock_etf_data()
        save_data(etf_df, 'etf_basic_info.csv', DATA_DIR)
    
    return etf_df

# ==================== 测试函数 ====================
if __name__ == '__main__':
    print("测试数据加载模块...")
    
    # 测试ETF基础信息获取
    etf_df = load_all_etf_data()
    print(f"\nETF数据概览:")
    print(etf_df.head())
    print(f"\nETF类型分布:")
    print(etf_df['etf_type'].value_counts())
    
    # 测试单只ETF历史数据
    test_etf = '510310.SH'
    hist_df = get_etf_historical_data(test_etf)
    if hist_df is not None:
        print(f"\n{test_etf}历史数据: {len(hist_df)}条")
    
    # 测试指数成分股
    test_index = '000300.SH'
    const_df = get_index_constituents(test_index)
    if const_df is not None:
        print(f"\n{test_index}成分股: {len(const_df)}只")

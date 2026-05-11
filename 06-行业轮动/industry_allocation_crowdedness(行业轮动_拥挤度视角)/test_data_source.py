import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, BASE_DIR)

print("测试tushare连接...")

try:
    import tushare as ts
    token = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
    pro = ts.pro_api(token)
    pro._DataApi__token = token
    pro._DataApi__http_url = "http://jiaoch.site"

    print("尝试获取交易日历...")
    cal = pro.trade_cal(start_date='20200101', end_date='20200110', api=pro)
    print(f"交易日历获取结果: {cal}")

    print("\n尝试获取指数列表...")
    try:
        index_info = pro.index_basic(market='SW', api=pro)
        print(f"申万指数列表: {index_info.head() if index_info is not None else 'None'}")
    except Exception as e:
        print(f"获取指数列表失败: {e}")

    print("\n尝试获取上证指数日线数据...")
    try:
        sh_df = pro.index_daily(ts_code='000001.SH', start_date='20200101', end_date='20200110', api=pro)
        print(f"上证指数数据: {sh_df if sh_df is not None else 'None'}")
    except Exception as e:
        print(f"获取上证指数失败: {e}")

except Exception as e:
    print(f"连接失败: {e}")
    import traceback
    traceback.print_exc()

print("\n尝试使用efinance...")
try:
    import efinance as ef
    print("efinance模块可用")
    codes = ['801010', '801020', '801030']
    for code in codes:
        try:
            df = ef.stock.get_quote_history(code)
            if df is not None and len(df) > 0:
                print(f"{code}: 获取成功 {len(df)} 条")
            else:
                print(f"{code}: 无数据")
        except Exception as e2:
            print(f"{code}: 获取失败 {e2}")
except ImportError:
    print("efinance未安装")

print("\n尝试使用baostock...")
try:
    import baostock as bs
    bs.login()
    print("baostock登录成功")
    rs = bs.query_history_k_data_plus("sh.000001",
        "date,code,open,high,low,close,volume",
        start_date='2020-01-01', end_date='2020-01-10',
        frequency="d", adjustflag="3")
    data = rs.get_data()
    print(f"baostock数据: {data.head()}")
    bs.logout()
except ImportError:
    print("baostock未安装")
except Exception as e:
    print(f"baostock失败: {e}")
import sys
import os
sys.path.insert(0, 'd:/Documents/trae_projects/momentum_enhanced_risk_parity')

import tushare as ts
from source.config import TUSHARE_TOKEN, TUSHARE_API_URL, ASSETS, DATA_DIR, BACKTEST_PARAMS

print("=" * 60)
print("初始化 tushare API...")
print("=" * 60)

pro = ts.pro_api(TUSHARE_TOKEN)
pro._DataApi__token = TUSHARE_TOKEN
pro._DataApi__http_url = TUSHARE_API_URL

print(f"API URL: {TUSHARE_API_URL}")
print(f"Token: {TUSHARE_TOKEN[:20]}...")

print("\n" + "=" * 60)
print("测试各类型数据接口...")
print("=" * 60)

test_results = {}

# 1. 测试沪深300指数
print("\n[1] 测试沪深300指数 (000300.SH)...")
try:
    df = pro.index_daily(ts_code='000300.SH', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        print(f"  最近日期: {df['trade_date'].max()}")
        print(f"  收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
        test_results['CSI300'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['CSI300'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['CSI300'] = False

# 2. 测试恒生指数
print("\n[2] 测试恒生指数 (HSI.HK)...")
try:
    df = pro.index_daily(ts_code='HSI.HK', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['HSI'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['HSI'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['HSI'] = False

# 3. 测试日经225
print("\n[3] 测试日经225 (N225.JP)...")
try:
    df = pro.index_daily(ts_code='N225.JP', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['Nikkei225'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['Nikkei225'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['Nikkei225'] = False

# 4. 测试标普500
print("\n[4] 测试标普500 (SPX.GI)...")
try:
    df = pro.index_daily(ts_code='SPX.GI', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['SP500'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['SP500'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['SP500'] = False

# 5. 测试黄金期货
print("\n[5] 测试COMEX黄金 (GC00Y.NYM)...")
try:
    df = ts.pro_bar(ts_code='GC00Y.NYM', api=pro, start_date='20240101', end_date='20240131', asset='FT')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['Gold'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['Gold'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['Gold'] = False

# 6. 测试布油期货
print("\n[6] 测试ICE布油 (BZ00Y.NYM)...")
try:
    df = ts.pro_bar(ts_code='BZ00Y.NYM', api=pro, start_date='20240101', end_date='20240131', asset='FT')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['BrentOil'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['BrentOil'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['BrentOil'] = False

# 7. 测试铜期货
print("\n[7] 测试SHFE铜 (CU00Y.SHF)...")
try:
    df = ts.pro_bar(ts_code='CU00Y.SHF', api=pro, start_date='20240101', end_date='20240131', asset='FT')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['Copper'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['Copper'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['Copper'] = False

# 8. 测试美国国债ETF
print("\n[8] 测试美国国债7-10年ETF (IEF.US)...")
try:
    df = ts.pro_bar(ts_code='IEF.US', api=pro, start_date='20240101', end_date='20240131', asset='O')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['USTBond'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['USTBond'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['USTBond'] = False

# 9. 测试中债国债指数
print("\n[9] 测试中债-国债总财富(5-7年)指数 (CBA00603.CI)...")
try:
    df = pro.bond_指数(index_code='CBA00603.CI', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['CNTBond'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['CNTBond'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['CNTBond'] = False

# 10. 测试中债企业债AAA指数
print("\n[10] 测试中债-企业债AAA财富指数 (CBA00701.CI)...")
try:
    df = pro.bond_指数(index_code='CBA00701.CI', start_date='20240101', end_date='20240131')
    if df is not None and len(df) > 0:
        print(f"  ✓ 成功! 获取 {len(df)} 条记录")
        test_results['CNCorpBond'] = True
    else:
        print("  ✗ 返回空数据")
        test_results['CNCorpBond'] = False
except Exception as e:
    print(f"  ✗ 失败: {e}")
    test_results['CNCorpBond'] = False

print("\n" + "=" * 60)
print("数据获取测试结果汇总")
print("=" * 60)
available = [k for k, v in test_results.items() if v]
unavailable = [k for k, v in test_results.items() if not v]

print(f"\n✓ 可获取数据的资产 ({len(available)}): {available}")
print(f"✗ 无法获取数据的资产 ({len(unavailable)}): {unavailable}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
if len(unavailable) > 0:
    print(f"\n以下 {len(unavailable)} 种资产数据无法通过 tushare 获取:")
    for asset in unavailable:
        asset_info = ASSETS.get(asset, {})
        print(f"  - {asset_info.get('name', asset)} ({asset_info.get('ts_code', 'N/A')})")
    print("\n请提供以下任一解决方案:")
    print("  1. 提供这些资产的本地数据文件 (CSV/Excel格式)")
    print("  2. 提供其他可用的数据源信息")
    print("  3. 确认这些资产是否在tushare上有其他合约代码")
else:
    print("\n所有资产数据均可获取!")

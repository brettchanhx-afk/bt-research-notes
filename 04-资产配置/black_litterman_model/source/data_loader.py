"""
data_loader.py - 大类资产日频数据获取 (tushare + akshare + EastMoney)

数据源:
  CSI300  → tushare  index_daily(000300.SH)          ✅ 可靠
  SP500   → akshare  index_global_hist_em(标普500)      ✅ 修正列名: '收盘'
  HSI     → akshare  index_global_hist_em(恒生指数)    ✅ 修正列名: '收盘'
  CR_GOV  → EastMoney push2his API (1.000012)         ✅ 可靠
  CR_CORP → EastMoney push2his API (1.000013)         ✅ 可靠
  NHCI    → tushare  fund_daily(159934.SZ)            ✅ 黄金ETF兜底
"""

import os
import time
import json
import re
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

# ========== Tushare 初始化 (严格按用户规则) ==========
import tushare as ts
TUSHARE_TOKEN = "1015d5b62774ab54f44bc6ef3ed95b02e2d8fb9a68dac06c0e344a3f3ceb"
pro = ts.pro_api(TUSHARE_TOKEN)
pro._DataApi__token = TUSHARE_TOKEN
pro._DataApi__http_url = "http://jiaoch.site"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


# ============================================================================
# 工具
# ============================================================================

def _sleep(s: float = 0.5):
    time.sleep(s)


def _to_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """将日期列设为索引并排序"""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df.set_index(col).sort_index()


def _filter_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按日期区间筛选"""
    sd = pd.to_datetime(f"{start[:4]}-{start[4:6]}-{start[6:]}")
    ed = pd.to_datetime(f"{end[:4]}-{end[4:6]}-{end[6:]}")
    return df[(df.index >= sd) & (df.index <= ed)]


# ============================================================================
# DataLoader
# ============================================================================

class DataLoader:
    """
    大类资产日频收益率 (%)
    资产: CSI300, SP500, HSI, CR_GOV, CR_CORP, NHCI
    """

    def fetch_all_assets(
        self,
        start: str = '20061101',
        end: str = '20230131',
    ) -> pd.DataFrame:
        print("[DataLoader] 获取6大类资产 ...")
        print(f"  tushare: {pro._DataApi__http_url}")

        results: dict[str, pd.DataFrame] = {}

        # 1. CSI300
        df = self._csi300(start, end)
        if df is not None:
            results['CSI300'] = df
            print(f"  [OK] 沪深300: {len(df)} 条")
        else:
            print(f"  [FAIL] 沪深300")

        # 2. SP500
        df = self._sp500(start, end)
        if df is not None:
            results['SP500'] = df
            print(f"  [OK] 标普500: {len(df)} 条")
        else:
            print(f"  [FAIL] 标普500")

        # 3. HSI
        df = self._hsi(start, end)
        if df is not None:
            results['HSI'] = df
            print(f"  [OK] 恒生指数: {len(df)} 条")
        else:
            print(f"  [FAIL] 恒生指数")

        # 4. CR_GOV - EastMoney 上证国债指数
        df = self._eastmoney_bond('1.000012', 'CR_GOV', start, end)
        if df is not None:
            results['CR_GOV'] = df
            print(f"  [OK] 中债国债总财富(上证国债): {len(df)} 条")
        else:
            print(f"  [WARN] CR_GOV 失败，改用国债ETF(511010)")
            df = self._fund_etf('511010.SH', start, end)
            if df is not None:
                results['CR_GOV'] = df

        # 5. CR_CORP - 中证企业债
        df = self._eastmoney_bond('1.000013', 'CR_CORP', start, end)
        if df is not None:
            results['CR_CORP'] = df
            print(f"  [OK] 中债企业债总财富: {len(df)} 条")
        else:
            print(f"  [WARN] CR_CORP 失败，改用城投债ETF(511270)")
            df = self._fund_etf('511270.SH', start, end)
            if df is not None:
                results['CR_CORP'] = df

        # 6. NHCI = 黄金ETF
        df = self._fund_etf('159934.SZ', start, end)
        if df is not None:
            results['NHCI'] = df
            print(f"  [OK] 南华商品(黄金ETF代理): {len(df)} 条")
        else:
            df = self._fund_etf('518880.SH', start, end)
            if df is not None:
                results['NHCI'] = df
                print(f"  [OK] 南华商品(黄金ETF-518880): {len(df)} 条")

        if not results:
            raise RuntimeError("全部数据获取失败，请检查网络")

        # 合并对齐
        prices = results[list(results.keys())[0]].copy()
        for name, df in list(results.items())[1:]:
            prices = prices.join(df, how='outer')
        prices = prices.sort_index().ffill().dropna(how='all')

        # 对数收益率 %
        returns = np.log(prices / prices.shift(1)) * 100
        returns = returns.dropna(how='all')

        valid = [c for c in returns.columns if not returns[c].isna().all()]
        print(f"\n[DataLoader] 完成: shape={returns.shape}")
        print(f"  日期: {returns.index[0].date()} ~ {returns.index[-1].date()}")
        print(f"  有效资产: {valid}")
        return returns

    # ── CSI300 (tushare) ─────────────────────────────────────────────
    def _csi300(self, start: str, end: str) -> pd.DataFrame | None:
        for _ in range(3):
            try:
                _sleep(0.4)
                df = pro.index_daily(
                    ts_code='000300.SH',
                    start_date=start,
                    end_date=end
                )
                if df is not None and len(df) > 0:
                    df = _to_dt(df, 'trade_date')
                    return df[['close']].rename(columns={'close': 'CSI300'})
            except Exception as e:
                _sleep(1)
        return None

    # ── SP500 (akshare) ────────────────────────────────────────────────
    def _sp500(self, start: str, end: str) -> pd.DataFrame | None:
        try:
            import akshare as ak
            df = ak.index_global_hist_em(symbol='标普500')
            if df is None or df.empty:
                return None
            # 列布局(iloc): 0=日期, 1=代码, 2=名称, 3=开盘, 4=收盘, 5=最高, 6=最低, 7=涨跌幅
            result = pd.DataFrame({
                'date':  pd.to_datetime(df.iloc[:, 0]),
                'SP500': df.iloc[:, 4].astype(float),
            }).set_index('date').sort_index()
            result = _filter_date(result, start, end)
            return result if not result.empty else None
        except Exception as e:
            print(f"    SP500 error: {e}")
        return None

    # ── HSI (akshare) ─────────────────────────────────────────────────
    def _hsi(self, start: str, end: str) -> pd.DataFrame | None:
        try:
            import akshare as ak
            df = ak.index_global_hist_em(symbol='恒生指数')
            if df is None or df.empty:
                return None
            result = pd.DataFrame({
                'date':  pd.to_datetime(df.iloc[:, 0]),
                'HSI':   df.iloc[:, 4].astype(float),
            }).set_index('date').sort_index()
            result = _filter_date(result, start, end)
            return result if not result.empty else None
        except Exception as e:
            print(f"    HSI error: {e}")
        return None

    # ── EastMoney 债券指数 ─────────────────────────────────────────────
    def _eastmoney_bond(
        self, secid: str, col_name: str, start: str, end: str
    ) -> pd.DataFrame | None:
        """
        EastMoney K线 API 获取债券指数历史
        secid:
          1.000012 = 上证国债指数 (代理中债国债总财富)
          1.000013 = 上证企业债指数 (代理中债企业债总财富)
          1.000016 = 中债国债总财富
        """
        try:
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "secid": secid,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",   # 日K
                "fqt": "1",     # 前复权
                "beg": start,
                "end": end,
            }
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if not (data.get('data') and data['data'].get('klines')):
                return None

            lines = data['data']['klines']
            records = []
            for line in lines:
                parts = line.split(',')
                # 格式: 日期,开盘,收盘,最高,最低,成交量,成交额,涨跌幅,...
                records.append({
                    'date':   parts[0],
                    'open':   float(parts[1]),
                    'close':  float(parts[2]),
                    'high':   float(parts[3]),
                    'low':    float(parts[4]),
                    'vol':    float(parts[5]),
                })

            df = pd.DataFrame(records)
            df = _to_dt(df, 'date')
            df = _filter_date(df, start, end)
            if df.empty:
                return None
            return df[['close']].rename(columns={'close': col_name})
        except Exception as e:
            print(f"    EastMoney({secid}) error: {e}")
        return None

    # ── 基金ETF (tushare, 兜底) ──────────────────────────────────────
    def _fund_etf(self, ts_code: str, start: str, end: str) -> pd.DataFrame | None:
        for _ in range(3):
            try:
                _sleep(0.4)
                df = pro.fund_daily(ts_code=ts_code, start_date=start, end_date=end)
                if df is not None and len(df) > 0:
                    df = _to_dt(df, 'trade_date')
                    return df[['close']].rename(columns={'close': ts_code})
            except Exception:
                _sleep(1)
        return None

    # ── 工具 ────────────────────────────────────────────────────────
    def to_monthly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """日频 → 月频复利累计 (%)"""
        m = (1 + daily / 100).resample('M').prod() - 1
        m = m * 100
        m.index = m.index.to_period('M').to_timestamp()
        return m

    def save(self, df: pd.DataFrame, fname='asset_returns.csv'):
        p = os.path.join(DATA_DIR, fname)
        df.to_csv(p)
        print(f"[DataLoader] 保存: {p}  ({os.path.getsize(p)/1024:.0f} KB)")

    def load(self, fname='asset_returns.csv') -> pd.DataFrame | None:
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            print(f"[DataLoader] 加载缓存: {p}  shape={df.shape}")
            return df
        return None


# ============================================================================
# 便捷入口
# ============================================================================

def load_all_returns(
    start: str = '20061101',
    end: str = '20230131',
    force: bool = False,
) -> pd.DataFrame:
    loader = DataLoader()
    if not force:
        cached = loader.load()
        if cached is not None:
            return cached
    df = loader.fetch_all_assets(start, end)
    loader.save(df)
    return df


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("DataLoader 自测 (2021-01 ~ 2023-01)")
    print("=" * 55)
    try:
        df = load_all_returns('20210101', '20230131', force=True)
        print(df.describe().round(4).to_string())
    except Exception as e:
        print(f"FAIL: {e}")

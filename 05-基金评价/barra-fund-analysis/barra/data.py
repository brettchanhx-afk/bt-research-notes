# -*- coding: utf-8 -*-
"""
data.py — 数据获取模块
  - efinance  : 基金历史净值
  - baostock  : 指数历史行情
  - 本地缓存  : 避免重复请求
"""

import os
import logging
import pandas as pd
import numpy as np
import baostock as bs
import efinance as ef

logger = logging.getLogger(__name__)


class DataLoader:
    """统一数据加载器，支持本地 CSV 缓存。"""

    def __init__(self, data_dir: str, start_date: str, end_date: str):
        self.data_dir   = data_dir
        self.start_date = start_date
        self.end_date   = end_date
        os.makedirs(data_dir, exist_ok=True)

    # ── 基金净值 ──────────────────────────────────────────
    def load_fund_nav(self, fund_code: str, force_reload: bool = False) -> pd.DataFrame:
        """
        获取基金累计净值，返回 DataFrame(index=date, columns=['nav','return'])
        优先读取本地缓存，force_reload=True 时强制重新下载。
        """
        cache_path = os.path.join(self.data_dir, f"fund_{fund_code}.csv")

        if os.path.exists(cache_path) and not force_reload:
            logger.info(f"[cache] 读取基金 {fund_code} 缓存: {cache_path}")
            df = pd.read_csv(cache_path, index_col="date", parse_dates=True)
            return df

        logger.info(f"[efinance] 下载基金 {fund_code} 净值...")
        raw = ef.fund.get_quote_history(fund_code, pz=10000)

        if raw is None or len(raw) == 0:
            raise ValueError(f"efinance 未返回基金 {fund_code} 的数据")

        # 自动识别列名（中英文兼容）
        col_map = {}
        for col in raw.columns:
            # 尝试解码中文
            try:
                col_decoded = col.encode('latin1').decode('gbk')
            except:
                col_decoded = col
            
            lc = col_decoded.lower()
            if "日期" in col_decoded or "date" in lc:
                col_map[col] = "date"
            elif "累计净值" in col_decoded:
                col_map[col] = "acc_nav"
            elif "单位净值" in col_decoded:
                col_map[col] = "nav"

        raw = raw.rename(columns=col_map)
        
        # 优先使用累计净值，否则用单位净值
        if "acc_nav" in raw.columns:
            raw["nav"] = raw["acc_nav"]
        elif "nav" not in raw.columns:
            # 如果都没找到，用第一列数值列
            numeric_cols = raw.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                raw["nav"] = raw[numeric_cols[0]]
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()

        # 过滤日期范围
        raw = raw.loc[self.start_date : self.end_date]
        
        # 确保nav列存在
        if "nav" not in raw.columns:
            raise ValueError(f"无法找到净值列，可用列: {list(raw.columns)}")
        
        raw["nav"]    = pd.to_numeric(raw["nav"], errors="coerce")
        raw["return"] = raw["nav"].pct_change()
        df = raw[["nav", "return"]].dropna()

        df.to_csv(cache_path)
        logger.info(f"[efinance] 基金 {fund_code}: {len(df)} 条，已缓存")
        return df

    # ── 指数行情 ──────────────────────────────────────────
    def load_index(self, code: str, name: str, force_reload: bool = False) -> pd.Series:
        """
        获取单个指数日收益率，返回 Series(index=date, name=name)
        """
        cache_path = os.path.join(self.data_dir, f"index_{name}.csv")

        if os.path.exists(cache_path) and not force_reload:
            logger.info(f"[cache] 读取指数 {name} 缓存")
            s = pd.read_csv(cache_path, index_col="date", parse_dates=True).squeeze()
            s.name = f"{name}_return"
            return s

        logger.info(f"[baostock] 下载指数 {name} ({code})...")
        lg = bs.login()
        if lg.error_code != "0":
            raise ConnectionError(f"baostock 登录失败: {lg.error_msg}")

        rs = bs.query_history_k_data_plus(
            code, "date,close",
            start_date=self.start_date,
            end_date=self.end_date,
            frequency="d",
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if not rows:
            logger.warning(f"[baostock] 指数 {name} 无数据，返回空 Series")
            return pd.Series(dtype=float, name=name)

        df = pd.DataFrame(rows, columns=["date", "close"])
        df["date"]  = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.set_index("date").sort_index()
        df["return"] = df["close"].pct_change()
        df = df.dropna()
        
        s = df["return"].copy()
        s.name = f"{name}_return"

        s.to_frame().to_csv(cache_path)
        logger.info(f"[baostock] 指数 {name}: {len(s)} 条，已缓存")
        return s

    def load_all_indexes(self, factor_indexes: dict, force_reload: bool = False) -> pd.DataFrame:
        """批量下载所有因子指数，返回宽表 DataFrame。"""
        series_list = []
        for name, code in factor_indexes.items():
            s = self.load_index(code, name, force_reload=force_reload)
            if len(s) > 0:  # 只添加非空序列
                series_list.append(s)
        
        if not series_list:
            logger.error("所有指数数据都为空")
            return pd.DataFrame()
        
        # 使用join而不是concat，可以处理部分缺失
        result = series_list[0].to_frame()
        for s in series_list[1:]:
            result = result.join(s, how='outer')
        
        # 对于缺失的列，用0填充
        result = result.fillna(0)
        return result

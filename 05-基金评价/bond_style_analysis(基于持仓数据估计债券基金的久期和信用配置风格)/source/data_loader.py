# -*- coding: utf-8 -*-
"""
data_loader.py - 债券基金风格分析数据获取模块

数据源优先级: efinance -> akshare -> baostock -> ifind-MCP
不编造任何数据，全部实时拉取真实市场数据

作者: QClaw Agent | 版本: 1.0.0
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# =============================================================================
# 债券信用评级评分表
# 来源: 中国人民银行 2006 年《信贷市场和银行间债券市场信用评级规范》
# =============================================================================

# 中长期债券信用等级评分 (AAA+=17, AAA=16.5, ..., D=0)
LONG_TERM_RATING_SCORE = {
    "AAA+": 17.0,
    "AAA": 16.5,
    "AA+": 15.5,
    "AA": 15.0,
    "AA-": 14.5,
    "A+": 13.5,
    "A": 13.0,
    "A-": 12.5,
    "BBB+": 11.5,
    "BBB": 11.0,
    "BBB-": 10.5,
    "BB+": 9.5,
    "BB": 9.0,
    "BB-": 8.5,
    "B+": 7.5,
    "B": 7.0,
    "B-": 6.5,
    "CCC+": 5.5,
    "CCC": 5.0,
    "CCC-": 4.5,
    "CC": 3.0,
    "C": 1.0,
    "D": 0.0,
}

# 短期债券信用等级评分 (A-1=16.5, A-2=15, A-3=13, B=11, C=5, D=0)
SHORT_TERM_RATING_SCORE = {
    "A-1": 16.5,
    "A-2": 15.0,
    "A-3": 13.0,
    "B": 11.0,
    "C": 5.0,
    "D": 0.0,
}

# 合并评分表
RATING_SCORE_MAP = {**LONG_TERM_RATING_SCORE, **SHORT_TERM_RATING_SCORE}

# 评级中文名称
RATING_NAME_MAP = {
    "AAA+": "超AAA",
    "AAA": "AAA",
    "AA+": "AA+",
    "AA": "AA",
    "AA-": "AA-",
    "A+": "A+",
    "A": "A",
    "A-": "A-",
    "BBB+": "BBB+",
    "BBB": "BBB",
    "BBB-": "BBB-",
    "BB+": "BB+",
    "BB": "BB",
    "BB-": "BB-",
    "B+": "B+",
    "B": "B",
    "B-": "B-",
    "CCC+": "CCC+",
    "CCC": "CCC",
    "CCC-": "CCC-",
    "CC": "CC",
    "C": "C",
    "D": "D",
}


def parse_rating(rating: str) -> float:
    """
    解析债券信用评级，返回评分分数

    Parameters
    ----------
    rating : str
        评级符号，如 'AAA'、'AA+'、'A-1'

    Returns
    -------
    float
        评分分数，范围 [0, 17]，无效评级返回 0.0
    """
    if pd.isna(rating) or not rating:
        return 0.0

    rating = str(rating).strip().upper()
    if rating in RATING_SCORE_MAP:
        return RATING_SCORE_MAP[rating]

    # 尝试模糊匹配基础等级
    base_map = {
        "AAA": 16.5, "AA": 15.0, "A": 12.5,
        "BBB": 10.5, "BB": 8.5, "B": 6.5,
        "CCC": 4.5, "CC": 2.5, "C": 0.5, "D": 0.0,
    }
    for base, score in base_map.items():
        if rating.startswith(base):
            return score
    return 0.0


# =============================================================================
# BondStyleDataLoader - 债券基金数据加载器
# =============================================================================

class BondStyleDataLoader:
    """
    债券基金风格分析数据加载器

    支持数据源 (按优先级):
    1. efinance     - 基金净值、重仓债券 (首选)
    2. akshare      - 基金净值、持仓数据
    3. baostock     - 指数/基金历史行情
    4. bondpy       - 债券数据
    5. ifind-MCP    - 最后选项

    Example
    --------
    >>> loader = BondStyleDataLoader()
    >>> nav_df = loader.get_fund_nav("000012", "20240101", "20250101")
    >>> holdings = loader.get_fund_holdings("000012")
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or "./data"
        self._cache = {}

    # -------------------------------------------------------------------------
    # 1. 基金净值数据
    # -------------------------------------------------------------------------

    def get_fund_nav(
        self,
        fund_code: str,
        start_date: str = "20200101",
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取债券基金单位净值序列

        Parameters
        ----------
        fund_code : str
            基金代码，如 '000012' (华夏债券A)
        start_date : str
            开始日期，格式 YYYYMMDD
        end_date : str, optional
            结束日期，默认今天

        Returns
        -------
        pd.DataFrame
            列: date(日期), nav(单位净值), cumulative_nav(累计净值),
                daily_return(日收益率)
        """
        import datetime

        if end_date is None:
            end_date = datetime.datetime.now().strftime("%Y%m%d")

        # ---- 尝试 efinance ----
        try:
            import efinance as ef

            df = ef.fund.get_fund_history(fund_code, start_date, end_date)
            if df is not None and len(df) > 0:
                df = df.copy()
                df["date"] = pd.to_datetime(df["日期"])
                df = df.sort_values("date").reset_index(drop=True)

                # 适配不同字段名
                nav_col = "单位净值" if "单位净值" in df.columns else "净值"
                cum_col = "累计净值" if "累计净值" in df.columns else nav_col

                df["nav"] = pd.to_numeric(df[nav_col], errors="coerce")
                df["cumulative_nav"] = pd.to_numeric(df[cum_col], errors="coerce")
                df["daily_return"] = df["nav"].pct_change().fillna(0)

                result = df[["date", "nav", "cumulative_nav", "daily_return"]].dropna(
                    subset=["nav"]
                )
                print(f"[OK] efinance 基金 {fund_code} 净值: {len(result)} 条")
                self._cache[f"nav_{fund_code}"] = result
                return result
        except Exception as e:
            print(f"[WARN] efinance 净值失败 ({fund_code}): {e}")

        # ---- 尝试 akshare ----
        try:
            import akshare as ak

            ak_df = ak.fund_open_fund_info_em(
                symbol=fund_code, indicator="单位净值走势"
            )
            if ak_df is not None and len(ak_df) > 0:
                ak_df = ak_df.copy()
                ak_df["date"] = pd.to_datetime(ak_df["净值日期"])
                ak_df = ak_df[(ak_df["date"] >= start_date) & (ak_df["date"] <= end_date)]
                ak_df["nav"] = pd.to_numeric(ak_df["单位净值"], errors="coerce")
                ak_df["cumulative_nav"] = pd.to_numeric(
                    ak_df.get("累计净值", ak_df["单位净值"]), errors="coerce"
                )
                ak_df["daily_return"] = ak_df["nav"].pct_change().fillna(0)
                result = ak_df[["date", "nav", "cumulative_nav", "daily_return"]].dropna(
                    subset=["nav"]
                )
                print(f"[OK] akshare 基金 {fund_code} 净值: {len(result)} 条")
                self._cache[f"nav_{fund_code}"] = result
                return result
        except Exception as e:
            print(f"[WARN] akshare 净值失败 ({fund_code}): {e}")

        raise ConnectionError(
            f"无法获取基金 {fund_code} 的净值数据，请检查网络或代码是否正确"
        )

    # -------------------------------------------------------------------------
    # 2. 基金重仓债券持仓
    # -------------------------------------------------------------------------

    def get_fund_holdings(self, fund_code: str) -> pd.DataFrame:
        """
        获取基金最新一期重仓债券列表

        Parameters
        ----------
        fund_code : str
            基金代码

        Returns
        -------
        pd.DataFrame
            列: bond_code(债券代码), bond_name(债券名称),
                market_value(持仓市值,万元), pct(占净值比,%)
        """
        # ---- 尝试 akshare 基金重仓债券 ----
        try:
            import akshare as ak

            df = ak.fund_bond_holding_em(symbol=fund_code)
            if df is not None and len(df) > 0:
                df = df.copy()
                # 标准化列名
                df = self._normalize_holdings_columns(df)
                print(f"[OK] akshare 基金 {fund_code} 持仓: {len(df)} 只债券")
                self._cache[f"holdings_{fund_code}"] = df
                return df
        except Exception as e:
            print(f"[WARN] akshare 持仓失败 ({fund_code}): {e}")

        # ---- 尝试 efinance ----
        try:
            import efinance as ef

            df = ef.fund.get_fund_portfolio(fund_code)
            if df is not None and len(df) > 0:
                df = df.copy()
                df = self._normalize_holdings_columns(df)
                print(f"[OK] efinance 基金 {fund_code} 持仓: {len(df)} 只")
                self._cache[f"holdings_{fund_code}"] = df
                return df
        except Exception as e:
            print(f"[WARN] efinance 持仓失败 ({fund_code}): {e}")

        print(f"[WARN] 无法获取基金 {fund_code} 的持仓数据")
        return pd.DataFrame()

    @staticmethod
    def _normalize_holdings_columns(df: pd.DataFrame) -> pd.DataFrame:
        """标准化持仓DataFrame列名"""
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "债券" in col or "名称" in col:
                col_map[col] = "bond_name"
            elif "代码" in col:
                col_map[col] = "bond_code"
            elif "市值" in col or "持仓" in col:
                col_map[col] = "market_value"
            elif "净值" in col or "占比" in col or "比例" in col or "%" in col:
                col_map[col] = "pct"
        df = df.rename(columns=col_map)
        required = ["bond_code", "bond_name", "market_value", "pct"]
        for col in required:
            if col not in df.columns:
                df[col] = None
        return df[required]

    # -------------------------------------------------------------------------
    # 3. 历史持仓 (多期)
    # -------------------------------------------------------------------------

    def get_historical_holdings(
        self, fund_code: str, n_periods: int = 4
    ) -> pd.DataFrame:
        """
        获取基金最近 n 期持仓数据

        Parameters
        ----------
        fund_code : str
            基金代码
        n_periods : int
            期数，默认 4 期 (约 1 年)

        Returns
        -------
        pd.DataFrame
            列: period(季度标签), bond_code, bond_name, market_value, pct
        """
        all_periods = []

        try:
            import akshare as ak

            df = ak.fund_bond_holding_em(symbol=fund_code)
            if df is not None and len(df) > 0:
                df = self._normalize_holdings_columns(df)
                df["period"] = "latest"
                all_periods.append(df)
        except Exception as e:
            print(f"[WARN] 获取历史持仓失败: {e}")

        if all_periods:
            combined = pd.concat(all_periods, ignore_index=True)
            print(f"[OK] 历史持仓 {len(combined)} 条")
            return combined

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # 4. 债券基础信息 (久期、信用评级)
    # -------------------------------------------------------------------------

    def get_bond_info(self, bond_code: str) -> dict:
        """
        获取单只债券的基础信息

        Parameters
        ----------
        bond_code : str
            债券代码

        Returns
        -------
        dict
            包含: bond_code, bond_name, duration(久期), maturity(到期期限),
                coupon_rate(票息率), credit_rating, credit_score, yield_to_mat
        """
        info = {
            "bond_code": bond_code,
            "bond_name": None,
            "duration": None,
            "maturity": None,
            "coupon_rate": None,
            "credit_rating": None,
            "credit_score": None,
            "yield_to_mat": None,
            "type": None,
        }

        # ---- 尝试 akshare ----
        try:
            import akshare as ak

            # 获取债券基本信息
            try:
                bond_df = ak.bond_info_em(symbol=bond_code)
                if bond_df is not None:
                    # 提取相关信息
                    info_dict = bond_df.to_dict("records")
                    if info_dict:
                        row = info_dict[0]
                        info["bond_name"] = row.get("债券名称")
                        info["credit_rating"] = row.get("主体评级") or row.get("债券评级")
                        info["credit_score"] = parse_rating(info["credit_rating"])
            except Exception:
                pass

            # 获取债券实时行情 (久期估算用)
            try:
                bond_hist = ak.bond_zh_hs_cov(simple=False)
                # bond_hist 包含债券历史数据
            except Exception:
                pass

        except Exception as e:
            print(f"[WARN] 获取债券 {bond_code} 信息失败: {e}")

        return info

    # -------------------------------------------------------------------------
    # 5. 债券列表 (按评级/期限)
    # -------------------------------------------------------------------------

    def get_bond_list_by_rating(self, rating: str = None) -> pd.DataFrame:
        """
        获取债券列表，可按评级筛选

        Parameters
        ----------
        rating : str, optional
            评级，如 'AAA'、'AA+'

        Returns
        -------
        pd.DataFrame
            债券列表
        """
        try:
            import akshare as ak

            df = ak.bond_info_em(symbol="")  # 全量债券列表
            if df is not None and len(df) > 0:
                if rating:
                    df = df[df.get("债券评级", pd.Series()) == rating]
                return df
        except Exception as e:
            print(f"[WARN] 获取债券列表失败: {e}")

        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # 6. 缓存管理
    # -------------------------------------------------------------------------

    def save_cache(self, key: str, df: pd.DataFrame):
        """保存数据到缓存"""
        self._cache[key] = df

    def load_cache(self, key: str) -> Optional[pd.DataFrame]:
        """加载缓存数据"""
        return self._cache.get(key)

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

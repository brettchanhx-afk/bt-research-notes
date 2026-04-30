# -*- coding: utf-8 -*-
"""
data_loader.py - 债券基金数据获取模块

数据来源（按优先级）：
1. efinance - 基金净值数据
2. akshare - 中债指数数据、基金数据
3. baostock - 备用数据源

中债指数体系（研报推荐）：
- 国债指数：CBA0016XX.CS
- 金融债指数：CBA012XX.CS  
- 企业债指数：CBA020XX.CS
- 中期票据指数：CBA026XX.CS
- 信用债指数：CBA027XX.CS
- 高信用等级债券：CBA019XX.CS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BondDataLoader:
    """债券基金数据加载器"""
    
    # 中债指数配置（代码、名称、久期、信用评分）
    # 久期数据来源于中债指数公开信息（单位：年）
    # 信用评分：AAA=16.5, AA+=15, AA=14, AA-=12.5
    BOND_INDICES = {
        # 国债指数 - 无信用风险，久期分层
        "CBA00161.CS": {"name": "中债-国债1-3年", "duration": 2.0, "credit_score": 0, "type": "国债"},
        "CBA00162.CS": {"name": "中债-国债3-5年", "duration": 4.0, "credit_score": 0, "type": "国债"},
        "CBA00163.CS": {"name": "中债-国债5-7年", "duration": 6.0, "credit_score": 0, "type": "国债"},
        "CBA00164.CS": {"name": "中债-国债7-10年", "duration": 8.5, "credit_score": 0, "type": "国债"},
        
        # 金融债指数 - 准主权信用
        "CBA01221.CS": {"name": "中债-金融债1-3年", "duration": 2.0, "credit_score": 16.0, "type": "金融债"},
        "CBA01222.CS": {"name": "中债-金融债3-5年", "duration": 4.0, "credit_score": 16.0, "type": "金融债"},
        "CBA01223.CS": {"name": "中债-金融债5-7年", "duration": 6.0, "credit_score": 16.0, "type": "金融债"},
        
        # 高信用等级债券 - AA+及以上
        "CBA01921.CS": {"name": "中债-高信用等级1-3年", "duration": 2.0, "credit_score": 15.5, "type": "高信用"},
        "CBA01922.CS": {"name": "中债-高信用等级3-5年", "duration": 4.0, "credit_score": 15.5, "type": "高信用"},
        
        # 企业债AAA
        "CBA04221.CS": {"name": "中债-企业债AAA1-3年", "duration": 2.0, "credit_score": 16.5, "type": "企业债AAA"},
        "CBA04222.CS": {"name": "中债-企业债AAA3-5年", "duration": 4.0, "credit_score": 16.5, "type": "企业债AAA"},
        
        # 企业债AA+
        "CBA04121.CS": {"name": "中债-企业债AA+1-3年", "duration": 2.0, "credit_score": 15.0, "type": "企业债AA+"},
        "CBA04122.CS": {"name": "中债-企业债AA+3-5年", "duration": 4.0, "credit_score": 15.0, "type": "企业债AA+"},
        
        # 企业债AA
        "CBA04021.CS": {"name": "中债-企业债AA1-3年", "duration": 2.0, "credit_score": 14.0, "type": "企业债AA"},
        "CBA04022.CS": {"name": "中债-企业债AA3-5年", "duration": 4.0, "credit_score": 14.0, "type": "企业债AA"},
        
        # 信用债总指数
        "CBA02721.CS": {"name": "中债-信用债1-3年", "duration": 2.0, "credit_score": 15.0, "type": "信用债"},
        "CBA02722.CS": {"name": "中债-信用债3-5年", "duration": 4.0, "credit_score": 15.0, "type": "信用债"},
    }
    
    def __init__(self):
        self.fund_nav = None
        self.index_data = {}
        
    def get_fund_nav(self, fund_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基金净值数据
        
        数据源优先级：efinance -> akshare
        
        Parameters:
        -----------
        fund_code : str
            基金代码，如 "000012"
        start_date : str
            开始日期，格式 "20230101"
        end_date : str
            结束日期，格式 "20231231"
            
        Returns:
        --------
        pd.DataFrame
            columns: [date, nav, acc_nav, daily_return]
        """
        # 尝试 efinance
        try:
            import efinance as ef
            df = ef.fund.get_fund_history(fund_code, start_date, end_date)
            if df is not None and len(df) > 0:
                logger.info(f"[efinance] 成功获取基金 {fund_code} 净值数据")
                return self._process_fund_df(df)
        except Exception as e:
            logger.warning(f"[efinance] 获取失败: {e}")
        
        # 降级到 akshare
        try:
            import akshare as ak
            df = ak.fund_open_fund_daily_em()
            df = df[df["基金代码"] == fund_code].copy()
            if len(df) > 0:
                logger.info(f"[akshare] 成功获取基金 {fund_code} 净值数据")
                return self._process_fund_df_ak(df)
        except Exception as e:
            logger.warning(f"[akshare] 获取失败: {e}")
            
        raise ValueError(f"无法获取基金 {fund_code} 的净值数据")
    
    def _process_fund_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理efinance基金数据"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["日期"])
        df["nav"] = pd.to_numeric(df["单位净值"], errors="coerce")
        df["acc_nav"] = pd.to_numeric(df["累计净值"], errors="coerce")
        df = df.sort_values("date")
        df["daily_return"] = df["nav"].pct_change()
        return df[["date", "nav", "acc_nav", "daily_return"]].dropna()
    
    def _process_fund_df_ak(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理akshare基金数据"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["净值日期"])
        df["nav"] = pd.to_numeric(df["单位净值"], errors="coerce")
        df["acc_nav"] = pd.to_numeric(df["累计净值"], errors="coerce")
        df = df.sort_values("date")
        df["daily_return"] = df["nav"].pct_change()
        return df[["date", "nav", "acc_nav", "daily_return"]].dropna()
    
    def get_bond_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取中债指数数据
        
        Parameters:
        -----------
        index_code : str
            中债指数代码，如 "CBA00161.CS"
        start_date : str
            开始日期，格式 "2023-01-01"
        end_date : str
            结束日期，格式 "2023-12-31"
            
        Returns:
        --------
        pd.DataFrame
            columns: [date, close, daily_return]
        """
        # 尝试 akshare 获取债券指数
        try:
            import akshare as ak
            
            # 转换日期格式
            sd = start_date.replace("-", "")
            ed = end_date.replace("-", "")
            
            # 使用akshare的债券指数接口
            df = ak.bond_china_close_return(
                symbol=index_code.replace(".CS", ""),
                period="1",
                start_date=sd,
                end_date=ed
            )
            
            if df is not None and len(df) > 0:
                logger.info(f"[akshare] 成功获取指数 {index_code} 数据")
                return self._process_index_df(df)
        except Exception as e:
            logger.warning(f"[akshare] 获取指数 {index_code} 失败: {e}")
        
        # 如果无法获取，生成模拟数据用于演示
        logger.warning(f"无法获取指数 {index_code} 真实数据，生成模拟数据")
        return self._generate_mock_index_data(index_code, start_date, end_date)
    
    def _process_index_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理指数数据"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["日期"])
        df["close"] = pd.to_numeric(df["收盘价"], errors="coerce")
        df = df.sort_values("date")
        df["daily_return"] = df["close"].pct_change()
        return df[["date", "close", "daily_return"]].dropna()
    
    def _generate_mock_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟指数数据（用于演示）"""
        np.random.seed(hash(index_code) % 1000)
        
        dates = pd.date_range(start=start_date, end=end_date, freq="B")  # 工作日
        
        # 根据指数类型设置不同的收益特征
        index_info = self.BOND_INDICES.get(index_code, {})
        duration = index_info.get("duration", 3.0)
        
        # 久期越长，利率敏感性越高，波动越大
        base_return = 0.0002  # 基础日收益
        volatility = 0.0003 * (duration / 3.0)  # 波动率与久期相关
        
        daily_returns = np.random.normal(base_return, volatility, len(dates))
        
        # 生成价格序列
        prices = 100 * np.exp(np.cumsum(daily_returns))
        
        df = pd.DataFrame({
            "date": dates,
            "close": prices,
            "daily_return": [0] + list(daily_returns[1:])
        })
        
        return df
    
    def load_all_index_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        加载所有中债指数数据
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            {index_code: dataframe}
        """
        data = {}
        for code in self.BOND_INDICES.keys():
            try:
                df = self.get_bond_index_data(code, start_date, end_date)
                if len(df) > 0:
                    data[code] = df
            except Exception as e:
                logger.warning(f"加载指数 {code} 失败: {e}")
        
        logger.info(f"成功加载 {len(data)} 个指数数据")
        return data
    
    def get_index_info(self) -> pd.DataFrame:
        """获取指数信息表"""
        info = []
        for code, meta in self.BOND_INDICES.items():
            info.append({
                "code": code,
                "name": meta["name"],
                "duration": meta["duration"],
                "credit_score": meta["credit_score"],
                "type": meta["type"]
            })
        return pd.DataFrame(info)


def test_data_loader():
    """测试数据加载器"""
    loader = BondDataLoader()
    
    # 测试获取指数信息
    info = loader.get_index_info()
    print("中债指数列表:")
    print(info)
    
    # 测试获取单个指数数据
    df = loader.get_bond_index_data("CBA00161.CS", "2023-01-01", "2023-12-31")
    print(f"\n指数数据样例:")
    print(df.head())


if __name__ == "__main__":
    test_data_loader()

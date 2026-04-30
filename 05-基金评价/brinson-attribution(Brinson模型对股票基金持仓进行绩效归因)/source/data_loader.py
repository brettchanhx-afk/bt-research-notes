"""
data_loader.py - 数据获取与加载模块
使用efinance、akshare等开源库获取基金持仓、行业分类、基准指数数据
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据获取库
try:
    import efinance as ef
    EF_AVAILABLE = True
except ImportError:
    EF_AVAILABLE = False
    print("警告: efinance库未安装，部分功能将使用akshare替代")

try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False
    print("警告: akshare库未安装")


class FundDataLoader:
    """基金数据加载器"""
    
    # 申万一级行业分类（31个行业）
    SW_SECTORS = [
        '农林牧渔', '基础化工', '钢铁', '有色金属', '电子', '家用电器', '食品饮料',
        '纺织服饰', '轻工制造', '医药生物', '公用事业', '交通运输', '房地产', '商贸零售',
        '社会服务', '银行', '非银金融', '综合', '建筑材料', '建筑装饰', '电力设备',
        '机械设备', '国防军工', '计算机', '传媒', '通信', '煤炭', '石油石化', '环保',
        '美容护理', '汽车'
    ]
    
    def __init__(self):
        """初始化数据加载器"""
        self.cache = {}
    
    def get_fund_holdings(
        self,
        fund_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取基金持仓数据
        
        Parameters:
            fund_code: 基金代码（如'000001'）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        
        Returns:
            pd.DataFrame: 持仓数据，包含date, stock_code, stock_name, sector, weight列
        """
        if EF_AVAILABLE:
            try:
                # 使用efinance获取基金持仓
                holdings = ef.fund.get_fund_holdings(fund_code)
                if holdings is not None and not holdings.empty:
                    # 处理数据格式
                    holdings = self._process_ef_holdings(holdings, fund_code)
                    # 过滤日期
                    holdings = holdings[
                        (holdings['date'] >= start_date) & 
                        (holdings['date'] <= end_date)
                    ]
                    return holdings
            except Exception as e:
                print(f"efinance获取持仓失败: {e}")
        
        if AK_AVAILABLE:
            try:
                # 使用akshare获取基金持仓
                holdings = ak.fund_portfolio_hold_em(symbol=fund_code, date=end_date[:4])
                if holdings is not None and not holdings.empty:
                    holdings = self._process_ak_holdings(holdings, fund_code)
                    return holdings
            except Exception as e:
                print(f"akshare获取持仓失败: {e}")
        
        # 如果都失败，返回模拟数据
        print("使用模拟持仓数据")
        return self._generate_mock_holdings(fund_code, start_date, end_date)
    
    def _process_ef_holdings(self, holdings: pd.DataFrame, fund_code: str) -> pd.DataFrame:
        """处理efinance持仓数据"""
        # 根据实际efinance返回格式调整
        df = holdings.copy()
        df['fund_code'] = fund_code
        # 重命名列以统一格式
        column_mapping = {
            '股票代码': 'stock_code',
            '股票名称': 'stock_name',
            '占净值比例': 'weight',
            '行业': 'sector',
            '报告期': 'date'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df
    
    def _process_ak_holdings(self, holdings: pd.DataFrame, fund_code: str) -> pd.DataFrame:
        """处理akshare持仓数据"""
        df = holdings.copy()
        df['fund_code'] = fund_code
        column_mapping = {
            '股票代码': 'stock_code',
            '股票名称': 'stock_name',
            '占净值比例': 'weight',
            '行业': 'sector',
            '季度': 'date'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df
    
    def _generate_mock_holdings(
        self,
        fund_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """生成模拟持仓数据用于测试"""
        np.random.seed(42)
        
        # 生成季度日期
        dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        holdings_list = []
        for date in dates:
            # 随机选择20-30只股票
            n_stocks = np.random.randint(20, 31)
            
            # 随机选择行业
            sectors = np.random.choice(self.SW_SECTORS, n_stocks)
            
            # 生成股票代码和名称
            stock_codes = [f"{np.random.randint(600000, 699999):06d}" for _ in range(n_stocks)]
            stock_names = [f"股票{i+1}" for i in range(n_stocks)]
            
            # 生成权重（归一化到100%）
            weights = np.random.random(n_stocks)
            weights = weights / weights.sum()
            
            for i in range(n_stocks):
                holdings_list.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'fund_code': fund_code,
                    'stock_code': stock_codes[i],
                    'stock_name': stock_names[i],
                    'sector': sectors[i],
                    'weight': weights[i]
                })
        
        return pd.DataFrame(holdings_list)
    
    def get_sector_index_returns(
        self,
        start_date: str,
        end_date: str,
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        获取申万行业指数收益率
        
        Parameters:
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率（'D'日, 'W'周, 'M'月）
        
        Returns:
            pd.DataFrame: 各行业指数收益率，列为行业名，行为日期
        """
        if AK_AVAILABLE:
            try:
                # 使用akshare获取申万行业指数
                sector_returns = []
                
                for sector in self.SW_SECTORS[:5]:  # 先获取前5个行业作为示例
                    try:
                        # 申万行业指数代码映射（简化处理）
                        index_code = self._get_sw_index_code(sector)
                        if index_code:
                            df = ak.index_zh_a_hist(symbol=index_code, period="daily",
                                                    start_date=start_date.replace('-', ''),
                                                    end_date=end_date.replace('-', ''))
                            if df is not None and not df.empty:
                                df['sector'] = sector
                                sector_returns.append(df)
                    except Exception as e:
                        print(f"获取{sector}行业数据失败: {e}")
                
                if sector_returns:
                    combined = pd.concat(sector_returns, ignore_index=True)
                    return self._process_sector_returns(combined)
                    
            except Exception as e:
                print(f"akshare获取行业收益失败: {e}")
        
        # 返回模拟数据
        print("使用模拟行业收益数据")
        return self._generate_mock_sector_returns(start_date, end_date, freq)
    
    def _get_sw_index_code(self, sector_name: str) -> Optional[str]:
        """获取申万行业指数代码"""
        # 申万行业指数代码映射（部分示例）
        code_mapping = {
            '农林牧渔': '801010',
            '基础化工': '801030',
            '钢铁': '801040',
            '有色金属': '801050',
            '电子': '801080',
            '家用电器': '801110',
            '食品饮料': '801120',
            '纺织服饰': '801130',
            '医药生物': '801150',
            '银行': '801780',
            '非银金融': '801790',
            '房地产': '801180',
            '汽车': '801880',
            '计算机': '801750',
            '传媒': '801760',
            '通信': '801770',
        }
        return code_mapping.get(sector_name)
    
    def _process_sector_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理行业收益数据"""
        # 计算收益率
        df['date'] = pd.to_datetime(df['日期'])
        df['close'] = df['收盘'].astype(float)
        df = df.sort_values(['sector', 'date'])
        df['return'] = df.groupby('sector')['close'].pct_change()
        
        # 透视表
        pivot_df = df.pivot(index='date', columns='sector', values='return')
        return pivot_df
    
    def _generate_mock_sector_returns(
        self,
        start_date: str,
        end_date: str,
        freq: str = 'M'
    ) -> pd.DataFrame:
        """生成模拟行业收益数据"""
        np.random.seed(42)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        data = {}
        for sector in self.SW_SECTORS:
            # 生成正态分布的收益率
            returns = np.random.normal(0.005, 0.08, len(dates))
            data[sector] = returns
        
        return pd.DataFrame(data, index=dates)
    
    def get_benchmark_returns(
        self,
        benchmark_code: str,
        start_date: str,
        end_date: str,
        freq: str = 'M'
    ) -> pd.Series:
        """
        获取基准指数收益率
        
        Parameters:
            benchmark_code: 基准指数代码（如'000300'为沪深300）
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率
        
        Returns:
            pd.Series: 基准指数收益率
        """
        if AK_AVAILABLE:
            try:
                df = ak.index_zh_a_hist(symbol=benchmark_code, period="daily",
                                        start_date=start_date.replace('-', ''),
                                        end_date=end_date.replace('-', ''))
                if df is not None and not df.empty:
                    df['date'] = pd.to_datetime(df['日期'])
                    df['close'] = df['收盘'].astype(float)
                    df = df.set_index('date').sort_index()
                    
                    # 重采样到指定频率
                    if freq == 'M':
                        df_resampled = df['close'].resample('M').last()
                    elif freq == 'W':
                        df_resampled = df['close'].resample('W').last()
                    else:
                        df_resampled = df['close']
                    
                    returns = df_resampled.pct_change().dropna()
                    return returns
                    
            except Exception as e:
                print(f"获取基准指数失败: {e}")
        
        # 返回模拟数据
        print("使用模拟基准收益数据")
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.004, 0.06, len(dates)), index=dates)
        return returns
    
    def get_stock_sector_mapping(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        获取股票所属行业映射
        
        Parameters:
            stock_codes: 股票代码列表
        
        Returns:
            Dict[str, str]: 股票代码到行业的映射
        """
        if AK_AVAILABLE:
            try:
                # 使用akshare获取股票行业信息
                stock_info = ak.stock_individual_info_em()
                mapping = dict(zip(stock_info['股票代码'], stock_info['行业']))
                return {code: mapping.get(code, '未知') for code in stock_codes}
            except Exception as e:
                print(f"获取股票行业信息失败: {e}")
        
        # 随机分配行业
        np.random.seed(42)
        return {code: np.random.choice(self.SW_SECTORS) for code in stock_codes}
    
    def aggregate_holdings_by_sector(
        self,
        holdings_df: pd.DataFrame,
        date_col: str = 'date',
        sector_col: str = 'sector',
        weight_col: str = 'weight'
    ) -> pd.DataFrame:
        """
        将持仓数据按行业聚合
        
        Parameters:
            holdings_df: 持仓数据
            date_col: 日期列名
            sector_col: 行业列名
            weight_col: 权重列名
        
        Returns:
            pd.DataFrame: 聚合后的行业权重
        """
        return holdings_df.groupby([date_col, sector_col])[weight_col].sum().reset_index()


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def align_data(
        portfolio_weights: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        date_col: str = 'date'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        对齐四个数据集的时间和行业维度
        
        Parameters:
            portfolio_weights: 组合权重
            portfolio_returns: 组合收益
            benchmark_weights: 基准权重
            benchmark_returns: 基准收益
            date_col: 日期列名
        
        Returns:
            对齐后的四个DataFrame
        """
        # 获取共同日期
        dates = set(portfolio_weights[date_col])
        dates = dates.intersection(set(portfolio_returns[date_col]))
        dates = dates.intersection(set(benchmark_weights[date_col]))
        dates = dates.intersection(set(benchmark_returns[date_col]))
        common_dates = sorted(list(dates))
        
        # 过滤日期
        pw = portfolio_weights[portfolio_weights[date_col].isin(common_dates)]
        pr = portfolio_returns[portfolio_returns[date_col].isin(common_dates)]
        bw = benchmark_weights[benchmark_weights[date_col].isin(common_dates)]
        br = benchmark_returns[benchmark_returns[date_col].isin(common_dates)]
        
        return pw, pr, bw, br
    
    @staticmethod
    def normalize_weights(weights_df: pd.DataFrame, 
                          date_col: str = 'date',
                          weight_col: str = 'weight') -> pd.DataFrame:
        """
        归一化权重使其每期总和为1
        
        Parameters:
            weights_df: 权重数据
            date_col: 日期列名
            weight_col: 权重列名
        
        Returns:
            pd.DataFrame: 归一化后的权重
        """
        df = weights_df.copy()
        df[weight_col] = df.groupby(date_col)[weight_col].transform(
            lambda x: x / x.sum()
        )
        return df


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载模块...")
    
    loader = FundDataLoader()
    
    # 测试获取基金持仓
    print("\n1. 测试获取基金持仓:")
    holdings = loader.get_fund_holdings('000001', '2023-01-01', '2023-12-31')
    print(holdings.head())
    
    # 测试获取行业收益
    print("\n2. 测试获取行业收益:")
    sector_returns = loader.get_sector_index_returns('2023-01-01', '2023-12-31', 'M')
    print(sector_returns.head())
    
    # 测试获取基准收益
    print("\n3. 测试获取基准收益:")
    benchmark_returns = loader.get_benchmark_returns('000300', '2023-01-01', '2023-12-31', 'M')
    print(benchmark_returns.head())
    
    print("\n数据加载模块测试完成!")

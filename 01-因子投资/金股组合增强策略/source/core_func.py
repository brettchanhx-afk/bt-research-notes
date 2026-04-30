from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from jqdata import *
from sqlalchemy.sql import func

from .toolkit import TradeDays
def get_stock_industry_name(codes: Union[str, List], date: str, level: str = 'sw_l1') -> Dict:
    """获取股票申万行业级别名称

    Parameters
    ----------
    codes : Union[str,List]
        标的
    date : str
        日期
    level : str, optional
        行业级别同聚宽, by default 'sw_l1'

    Returns
    -------
    Dict
        k-code,v-行业名称
    """
    def _get_dict_values(k: str, dic: Dict) -> str:

        try:
            industry_dic = dic[level]
        except KeyError as e:

            print(f'证券代码:{k},{date}未查询到{level}行业名称')
            return np.nan
        return industry_dic['industry_name']

    dic: Dict = get_industry(codes, date=date)

    return {k: _get_dict_values(k, v) for k, v in dic.items()}


def offset_limit_func(model, fields: Union[List, Tuple], limit: int,
                      *args) -> pd.DataFrame:
    """利用offset多次查询以跳过限制

    Args:
        model (_type_): model
        fields (Union[List, Tuple]): 查询字段
        limit (int): 限制
        args: 用于查询的条件
    Returns:
        pd.DataFrame

    """
    total_size: int = model.run_query(query(
        func.count('*')).filter(*args)).iloc[0, 0]
    # print('总数%s' % total_size)
    dfs: List = []
    # 以limit为步长循环offset的参数
    for i in range(0, total_size, limit):

        q = query(*fields).filter(*args).offset(i).limit(limit)  # 自第i条数据之后进行获取
        df: pd.DataFrame = model.run_query(q)
        # print(i, len(df))
        dfs.append(df)

    df: pd.DataFrame = pd.concat(dfs)

    return df


def get_sw1_price(code: Union[str, List], start_date: str, end_date: str, fields: Union[str, List]) -> pd.DataFrame:
    """获取申万行业日线数据

    Parameters
    ----------
    code : Union[str, List]
        标的代码
    start_date : str
        起始日
    end_date : str
        结束日
    fields : Union[str, List]
        查询字段

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(code, str):
        code = [code]
    if isinstance(fields, str):
        fields = [fields]

    fields = list(set(fields + ['date', 'code']))

    fields: Tuple = tuple(
        getattr(finance.SW1_DAILY_PRICE, field)
        for field in fields)

    df = offset_limit_func(finance, fields, 4000, finance.SW1_DAILY_PRICE.code.in_(code),
                           finance.SW1_DAILY_PRICE.date >= start_date,
                           finance.SW1_DAILY_PRICE.date <= end_date)

    # df['code'] = df['code'].apply(lambda x:x+'.SI')
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'date': 'trade_date'}, inplace=True)

    return df

class PrepareData():

    def __init__(self, gold_stock_frame: pd.DataFrame, start_dt: str, end_dt: str) -> None:

        self.gold_stock_frame = gold_stock_frame
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.td = td = TradeDays()
        self._add_tradeday_monthend()

    def init_data(self) -> pd.DataFrame:

        codes: List = self.gold_stock_frame['code'].unique().tolist()

        # 获取金股申万行业
        Stock2IndustryName: Dict = get_stock_industry_name(codes, self.end_dt)

        # 旧code无法识别行业 实际为房地产
        Stock2IndustryName['000043.XSHE'] = '房地产I'

        # 获取行业列表
        classify: pd.DataFrame = get_industries('sw_l1', date=self.end_dt)
        IndustryCode2SecName: Dict = classify['name'].to_dict()
        industry_code: List = list(IndustryCode2SecName.keys())

        # 获取区间月度序列
        periods: pd.Series = self.td.get_tradedays_month_end()['MonthEnd']
        periods: List = periods.loc[self.start_dt:self.end_dt].dt.strftime(
            '%Y-%m-%d').tolist()

        # 获取个股月度数据
        price = pd.concat(
            (get_price(codes, i, i, fields='close', panel=False) for i in periods))
        # 添加行业名称
        price['sw_l1'] = price['code'].map(Stock2IndustryName)

        # 获取行业月度收盘价数据
        industry_price = pd.concat(
            (get_sw1_price(industry_code, i, i, fields='close') for i in periods))
        # 添加行业名称
        industry_price['sw_l1'] = industry_price['code'].map(
            IndustryCode2SecName)

        self.stocK_price = price
        self.industry_price = industry_price

    def get_forward_returns(self) -> None:

        # 获取收益率
        month_stock_pct: pd.DataFrame = (self.stocK_price.pipe(pd.DataFrame.pivot_table,
                                                               index='time', columns=['sw_l1', 'code'], values='close')
                                         .pipe(pd.DataFrame.pct_change))

        month_industry_pct: pd.DataFrame = (self.stocK_price.pipe(pd.DataFrame.pivot_table,
                                                                  index='time', columns='sw_l1', values='close')
                                            .pipe(pd.DataFrame.pct_change))

        # 获取相对于行业的超额
        excess_ret: pd.DataFrame = month_stock_pct - month_industry_pct
        cols: pd.Index = excess_ret.columns.get_level_values(1)
        # 将MultiIndex还原成股票代码
        excess_ret.columns = cols
        month_stock_pct.columns = cols

        # stack数据
        stack_excess: pd.DataFrame = excess_ret.stack()
        stack_pct: pd.DataFrame = month_stock_pct.stack()

        stack_excess.name = 'industry_excess'
        stack_pct.name = 'next_ret'

        rets_df: pd.DataFrame = pd.concat((stack_excess, stack_pct), axis=1)
        rets_df = rets_df.reset_index()
        rets_df.rename(columns={'time': 'monthEnd'}, inplace=True)

        self.forward_returns = rets_df
        self.next_returns = month_stock_pct

    def full_data(self) -> pd.DataFrame:

        return pd.merge(self.gold_stock_frame, self.forward_returns, on=['code', 'monthEnd'])

    def _add_tradeday_monthend(self) -> None:
        """添加交易日的每月日期"""
        mapping_date = self.td.get_tradedays_month_end().set_index('MonthEnd(all)')[
            'MonthEnd'].to_dict()
        self.gold_stock_frame['monthEnd'] = self.gold_stock_frame['end_date'].map(
            mapping_date)


def get_author_proba(all_df: pd.DataFrame, returns_name: str = 'next_ret', window: int = 12, threshold: int = 5, beta_window: int = 12) -> pd.Series:
    """获取分析师概率

    使用 Beta 分布定量记录分析师金股推荐历史。假设，对于分析师的真实选股能
    力，没有先验知识。因此，每个分析师初始的 Beta 分布中，α = β = 1,此情况下,分析
    师推荐成功率在[0,1]上均匀分布。当分析师推荐金股成功时，即推荐月份的股票涨幅>0
    时，参数𝛼更新为𝛼 + 1。反之,当推荐失败时,参数𝛽更新为𝛽 + 1。

    Parameters
    ----------
    all_df : pd.DataFrame
        prepare_data的full_data结果
    returns_name : str
        统计股票自身涨跌->next_ret,相对于行业的超额->industry_excess, by default next_ret
    window : int, optional
        对分析师推荐次数及胜率的统计期窗口, by default 12
    threshold : int, optional
        分析师近window期推荐次数的阈值, by default 5
    beta_window : int, optional
        beta分布的计算期, by default 12

    Returns
    -------
    pd.Series 
        MultiIndex level0-date level1-code values
    """
    # 统计分析师推票情况
    status_author: pd.DataFrame = pd.pivot_table(
        all_df, index='monthEnd', columns='author', values='sec_short_name', aggfunc='count')

    # 如果有推荐则标记为1
    # 统计近12月推荐次数
    sel_author: pd.DataFrame = (~status_author.isna()).rolling(window).sum()

    # 过滤前序期
    filter_author: pd.DataFrame = sel_author.iloc[window-1:]
    # 筛选近一年推荐次数大于等于threshold日的分析师
    filter_author: pd.DataFrame = (filter_author >= threshold)

    # 统计股票自身涨跌->next_ret,相对于行业的超额->industry_excess
    sign_ret: pd.DataFrame = pd.pivot_table(all_df, index='monthEnd', columns=[
                                            'author', 'code'], values=returns_name, aggfunc=np.sign)

    # 统计近beta_window期分析师推票胜率
    a_params: pd.DataFrame = sign_ret[sign_ret > 0].fillna(
        0).rolling(beta_window).sum()
    b_params: pd.DataFrame = sign_ret[sign_ret < 0].fillna(
        0).rolling(beta_window).sum()

    a_params: pd.DataFrame = a_params.iloc[beta_window-1:]
    b_params: pd.DataFrame = b_params.iloc[beta_window-1:]

    # 将a,b值合并
    tmp: List = [[(a, abs(b)) for a, b in zip(a_values, b_values)]
                 for a_values, b_values in zip(a_params.values, b_params.values)]
    params: pd.DataFrame = pd.DataFrame(
        tmp, index=a_params.index, columns=b_params.columns)

    # 计算beta分布
    # 默认初始a=b=1所以这里+1
    beta_df: pd.DataFrame = params.applymap(
        lambda x: (x[0]+1)/(x[0]+x[1]+2) if x[0]+x[1] else 0)

    # 计算概率
    # 乘filter_author是过滤掉近一年推荐次数小于threshold次的分析师
    author_proba: pd.DataFrame = beta_df.groupby(
        level=0, axis=1).mean() * filter_author

    author_proba: pd.DataFrame = author_proba.where(author_proba != 0)
    author_proba: pd.Series = author_proba.stack()
    author_proba: pd.Series = author_proba.sort_index()
    author_proba: pd.Series = author_proba.dropna()
    return author_proba


def transform2stock_group(author_proba: pd.DataFrame, gold_stock_frame: pd.DataFrame, group_num: int = 5) -> pd.DataFrame:
    """将分析师概率分组并获取分析师当期所推股票

    Parameters
    ----------
    author_proba : pd.DataFrame
        get_author_proba的结果
    gold_stock_frame : pd.DataFrame
        金股表格
    group_num : int, optional
        分组, by default 5

    Returns
    -------
    pd.DataFrame
        columns-MultiIndex level-0 分组编号 level-1股票代码
    """

    def _get_group_stock(ser: pd.Series, gold_stock_frame: pd.DataFrame) -> List:
        """从gold_stock_frame按ser获取所推荐股票信息

        Parameters
        ----------
        ser : pd.Series
            MultiIndex level0-date level1-code values-proba
        gold_stock_frame : pd.DataFrame
            金股数据表

        Returns
        -------
        List
            标的
        """
        end_date, _ = ser.name
        # 去重
        author = ser.index.get_level_values(1).unique().tolist()

        codes = gold_stock_frame.query(
            'monthEnd == @end_date and author == @author')['code'].unique().tolist()
        if codes:
            return codes
        else:
            return np.nan

    # 分5组
    author_group: pd.Series = author_proba.groupby(level=0).apply(
        lambda x: pd.qcut(x, group_num, False))+1

    group = author_group.groupby([pd.Grouper(level=0), author_group.values])

    stock_group: pd.Series = group.apply(lambda x: pd.Series(
        _get_group_stock(x, gold_stock_frame)))
    stock_group.reset_index(level=2, drop=True, inplace=True)
    stock_group.index.names = ['monthEnd', 'group']
    stock_group.name = 'stock'
    stock_group: pd.DataFrame = stock_group.reset_index()
    stock_group: pd.DataFrame = stock_group.dropna(subset=['stock'])

    return stock_group


def get_stock_group_returns(stock_group: pd.DataFrame, next_returns: pd.DataFrame) -> pd.DataFrame:
    """获取分组股票组合收益率

    Parameters
    ----------
    stock_group : pd.DataFrame
        transform2stock_group的结果
    next_returns : pd.DataFrame
        未来期收益率

    Returns
    -------
    pd.DataFrame
        index-date columns-分组编号
    """
    stock_group['flag'] = 1

    flag = pd.pivot_table(stock_group, index='monthEnd', columns=[
                          'group', 'stock'], values='flag')

    return flag.groupby(level=0, axis=1).apply(lambda x: (x[x.name]*next_returns).mean(axis=1))

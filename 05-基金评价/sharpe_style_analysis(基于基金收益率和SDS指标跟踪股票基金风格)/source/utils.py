# -*- coding: utf-8 -*-
"""
工具函数模块 - 威廉·夏普风格分析

包含：
1. 日期处理工具
2. 数据预处理函数
3. 性能评估指标
4. 报告生成工具
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


def format_date(date_input, output_format: str = '%Y%m%d') -> str:
    """
    统一日期格式转换
    
    Parameters:
    -----------
    date_input : str, datetime, pd.Timestamp
        输入日期
    output_format : str
        输出格式
        
    Returns:
    --------
    str
        格式化后的日期字符串
    """
    if isinstance(date_input, str):
        # 尝试多种格式解析
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d-%m-%Y']:
            try:
                dt = datetime.strptime(date_input, fmt)
                return dt.strftime(output_format)
            except:
                continue
        raise ValueError(f"无法解析日期: {date_input}")
    
    elif isinstance(date_input, (datetime, pd.Timestamp)):
        return date_input.strftime(output_format)
    
    else:
        raise ValueError(f"不支持的日期类型: {type(date_input)}")


def get_trading_days(start_date: str, end_date: str, 
                     calendar: str = 'SSE') -> pd.DatetimeIndex:
    """
    获取交易日历
    
    Parameters:
    -----------
    start_date : str
        开始日期 'YYYYMMDD'
    end_date : str
        结束日期 'YYYYMMDD'
    calendar : str
        交易所代码，默认'SSE'（上交所）
        
    Returns:
    --------
    pd.DatetimeIndex
        交易日列表
    """
    try:
        import akshare as ak
        # 获取交易日历
        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        mask = (df['trade_date'] >= start) & (df['trade_date'] <= end)
        return df.loc[mask, 'trade_date']
    except:
        # 降级：使用工作日
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return pd.bdate_range(start=start, end=end)


def align_dates(data_dict: Dict[str, pd.DataFrame], 
                date_col: str = '日期') -> Dict[str, pd.DataFrame]:
    """
    对齐多个DataFrame的日期索引
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        数据字典
    date_col : str
        日期列名
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        日期对齐后的数据字典
    """
    # 找出共同的日期范围
    common_dates = None
    
    for name, df in data_dict.items():
        if date_col in df.columns:
            dates = set(pd.to_datetime(df[date_col]))
        else:
            dates = set(df.index)
        
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)
    
    common_dates = sorted(list(common_dates))
    
    # 过滤数据
    result = {}
    for name, df in data_dict.items():
        if date_col in df.columns:
            df = df[df[date_col].isin(common_dates)]
        else:
            df = df[df.index.isin(common_dates)]
        result[name] = df
    
    return result


def calculate_performance_metrics(returns: pd.Series, 
                                  risk_free_rate: float = 0.03) -> Dict:
    """
    计算收益风险指标
    
    Parameters:
    -----------
    returns : pd.Series
        收益率序列（日收益率）
    risk_free_rate : float
        无风险利率（年化），默认3%
        
    Returns:
    --------
    Dict
        绩效指标字典
    """
    if len(returns) == 0:
        return {}
    
    # 年化因子
    ann_factor = 252
    
    # 年化收益率
    ann_return = returns.mean() * ann_factor
    
    # 年化波动率
    ann_volatility = returns.std() * np.sqrt(ann_factor)
    
    # 夏普比率
    sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
    
    # 最大回撤
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 卡玛比率
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 胜率
    win_rate = (returns > 0).mean()
    
    # 盈亏比
    avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    profit_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 0
    
    return {
        'ann_return': round(ann_return, 4),
        'ann_volatility': round(ann_volatility, 4),
        'sharpe_ratio': round(sharpe_ratio, 4),
        'max_drawdown': round(max_drawdown, 4),
        'calmar_ratio': round(calmar_ratio, 4),
        'win_rate': round(win_rate, 4),
        'profit_loss_ratio': round(profit_loss_ratio, 4),
        'total_return': round((1 + returns).prod() - 1, 4)
    }


def compute_information_ratio(portfolio_returns: pd.Series,
                              benchmark_returns: pd.Series) -> float:
    """
    计算信息比率 (Information Ratio)
    
    IR = (Rp - Rb) / σ(Rp - Rb)
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        组合收益率
    benchmark_returns : pd.Series
        基准收益率
        
    Returns:
    --------
    float
        信息比率
    """
    # 对齐日期
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    p_ret = portfolio_returns.loc[common_dates]
    b_ret = benchmark_returns.loc[common_dates]
    
    # 超额收益
    excess_return = p_ret - b_ret
    
    # 信息比率
    ann_excess = excess_return.mean() * 252
    ann_tracking_error = excess_return.std() * np.sqrt(252)
    
    ir = ann_excess / ann_tracking_error if ann_tracking_error > 0 else 0
    return ir


def generate_style_report(fund_code: str,
                         fund_name: str,
                         style_result: Dict,
                         drift_result: Dict,
                         performance_metrics: Dict = None) -> str:
    """
    生成风格分析报告文本
    
    Parameters:
    -----------
    fund_code : str
        基金代码
    fund_name : str
        基金名称
    style_result : Dict
        风格分析结果
    drift_result : Dict
        风格漂移检测结果
    performance_metrics : Dict
        绩效指标
        
    Returns:
    --------
    str
        格式化的报告文本
    """
    lines = []
    lines.append("=" * 70)
    lines.append(" " * 20 + "基金风格分析报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"基金代码: {fund_code}")
    lines.append(f"基金名称: {fund_name}")
    lines.append(f"报告日期: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    
    # 风格分析结果
    lines.append("-" * 70)
    lines.append("【一、风格分析结果】")
    lines.append("-" * 70)
    
    if 'exposures' in style_result:
        lines.append("")
        lines.append("风格暴露系数:")
        exposures = style_result['exposures'].sort_values(ascending=False)
        for idx, exp in exposures.items():
            if exp > 0.01:
                lines.append(f"  {idx}: {exp:.4f} ({exp:.2%})")
    
    if 'r_squared' in style_result:
        lines.append("")
        lines.append(f"模型拟合优度 (R²): {style_result['r_squared']:.4f}")
        lines.append(f"解释力度: {'强' if style_result['r_squared'] > 0.7 else '中等' if style_result['r_squared'] > 0.4 else '弱'}")
    
    if 'tracking_error' in style_result:
        lines.append(f"跟踪误差 (年化): {style_result['tracking_error']:.4f}")
    
    lines.append("")
    
    # 风格漂移检测
    lines.append("-" * 70)
    lines.append("【二、风格漂移检测】")
    lines.append("-" * 70)
    
    if drift_result.get('sds_score'):
        lines.append("")
        lines.append(f"SDS风格漂移指标: {drift_result['sds_score']:.4f}")
        
        sds = drift_result['sds_score']
        if sds < 0.1:
            lines.append("  → 风格高度稳定")
        elif sds < 0.2:
            lines.append("  → 风格相对稳定")
        elif sds < 0.3:
            lines.append("  → 风格存在一定波动")
        else:
            lines.append("  → 风格漂移风险较高，需谨慎关注")
    
    if 'style_history' in drift_result:
        lines.append("")
        lines.append("历史风格演变:")
        for i, style in enumerate(drift_result['style_history']):
            lines.append(f"  子区间 {i+1}: {style}")
    
    if drift_result.get('has_drift'):
        lines.append("")
        lines.append("⚠️ 检测到风格漂移事件")
        if drift_result.get('drift_periods'):
            for event in drift_result['drift_periods']:
                lines.append(f"  区间 {event['from_period']}: {event['from_style']} → {event['to_style']}")
    else:
        lines.append("")
        lines.append("✓ 风格保持稳定")
    
    lines.append("")
    
    # 绩效指标
    if performance_metrics:
        lines.append("-" * 70)
        lines.append("【三、绩效指标】")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"年化收益率: {performance_metrics.get('ann_return', 0):.2%}")
        lines.append(f"年化波动率: {performance_metrics.get('ann_volatility', 0):.2%}")
        lines.append(f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"最大回撤: {performance_metrics.get('max_drawdown', 0):.2%}")
        lines.append(f"卡玛比率: {performance_metrics.get('calmar_ratio', 0):.4f}")
        lines.append(f"胜率: {performance_metrics.get('win_rate', 0):.2%}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("报告生成完毕")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_results_to_json(results: Dict, filepath: str):
    """
    保存分析结果为JSON文件
    
    Parameters:
    -----------
    results : Dict
        结果字典
    filepath : str
        保存路径
    """
    # 转换不可序列化的对象
    def convert(obj):
        if isinstance(obj, pd.Series):
            # Convert Series to dict with string keys
            return {str(k): convert(v) for k, v in obj.to_dict().items()}
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    converted = convert(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] 结果已保存: {filepath}")


def load_index_name_mapping() -> Dict[str, str]:
    """
    加载指数代码到名称的映射
    
    Returns:
    --------
    Dict[str, str]
        指数代码到中文名称的映射
    """
    return {
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        '000906.SH': '中证800',
        '000918.SH': '沪深300成长',
        '000919.SH': '沪深300价值',
        '000920.SH': '中证500成长',
        '000921.SH': '中证500价值',
        '000044.SH': '上证超级大盘',
        '000045.SH': '上证中盘',
        '000046.SH': '上证小盘',
        '000901.SH': '中证100成长',
        '000902.SH': '中证100价值',
        '000016.SH': '上证50',
        '000010.SH': '上证180',
        '399006.SZ': '创业板指',
        '399005.SZ': '中小板指',
    }

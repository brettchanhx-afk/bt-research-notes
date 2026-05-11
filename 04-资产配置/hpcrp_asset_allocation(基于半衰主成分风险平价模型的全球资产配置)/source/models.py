"""
资产配置模型模块
实现各种风险平价和资产配置模型
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def equal_weight(n_assets):
    """
    等权重资产配置法 (Equal Weight, EW)
    w_i = 1/N
    
    Args:
        n_assets: 资产数量
    
    Returns:
        np.ndarray: 等权重向量
    """
    return np.ones(n_assets) / n_assets


def equal_volatility_weights(returns):
    """
    等波动率资产配置法 (Equal Volatility, EV)
    w_i = σ_i^(-1) / Σσ_j^(-1)
    
    Args:
        returns: 收益率数据 (DataFrame)
    
    Returns:
        np.ndarray: 等波动率权重向量
    """
    vol = returns.std()
    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()
    return weights.values


def minimum_variance_weights(returns):
    """
    最小方差资产配置法 (Minimum Variance, MV)
    min w'Σw  s.t. w'1=1, w≥0
    
    Args:
        returns: 收益率数据 (DataFrame)
    
    Returns:
        np.ndarray: 最小方差权重向量
    """
    cov = returns.cov().values
    n = len(cov)
    
    # 目标函数: w'Σw
    def objective(w):
        return w @ cov @ w
    
    # 约束条件: w'1=1, w≥0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    # 初始权重: 等权重
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x


def maximum_diversification_weights(returns):
    """
    最大分散化资产配置法 (Maximum Diversification, MD)
    max (σ'w) / sqrt(w'Σw)
    
    Args:
        returns: 收益率数据 (DataFrame)
    
    Returns:
        np.ndarray: 最大分散化权重向量
    """
    cov = returns.cov().values
    vol = returns.std().values
    n = len(vol)
    
    # 目标函数: 最大化分散化比率
    def objective(w):
        avg_vol = np.dot(vol, w)
        port_vol = np.sqrt(w @ cov @ w)
        return -avg_vol / port_vol  # 最小化负值
    
    # 约束条件: w'1=1, w≥0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    # 初始权重: 等权重
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x


def risk_contribution(weights, cov):
    """
    计算各资产的风险贡献
    RC_i = w_i * (Σw)_i / (w'Σw)
    
    Args:
        weights: 权重向量
        cov: 协方差矩阵
    
    Returns:
        np.ndarray: 各资产的风险贡献
    """
    port_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / port_var
    return risk_contrib


def risk_parity_weights(returns):
    """
    风险平价资产配置法 (Risk Parity, RP)
    min ΣΣ[RC_i - RC_j]^2  s.t. w'1=1, w≥0
    
    Args:
        returns: 收益率数据 (DataFrame)
    
    Returns:
        np.ndarray: 风险平价权重向量
    """
    cov = returns.cov().values
    n = len(cov)
    
    # 目标函数: 风险贡献差异平方和
    def objective(w):
        rc = risk_contribution(w, cov)
        total = 0
        for i in range(n):
            for j in range(n):
                total += (rc[i] - rc[j]) ** 2
        return total
    
    # 约束条件: w'1=1, w≥0
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    # 初始权重: 等权重
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    return result.x


def principal_component_risk_parity_weights(returns):
    """
    主成分风险平价资产配置法 (Principal Component Risk Parity, PCRP)
    
    步骤:
    1. 对收益率进行PCA变换，得到主成分
    2. 对主成分应用风险平价模型
    3. 将主成分权重反推回原资产权重
    
    Args:
        returns: 收益率数据 (DataFrame)
    
    Returns:
        np.ndarray: 主成分风险平价权重向量
    """
    # 计算协方差矩阵
    cov = returns.cov().values
    n = len(cov)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 主成分变换矩阵 E'
    E = eigenvectors
    
    # 对主成分应用风险平价
    # 主成分收益率的协方差是对角阵 Λ
    Lambda = np.diag(eigenvalues)
    
    # 目标函数: 主成分风险贡献差异平方和
    def objective(w_pc):
        # 将主成分权重转换为原资产权重
        w = E @ w_pc
        
        # 主成分因子的风险贡献
        # 对于主成分, Var(r_pc_i) = λ_i
        # RC_pc_i = w_pc_i * λ_i / (w_pc'Λw_pc)
        w_lambda = w_pc * eigenvalues
        total_risk = np.sum(w_lambda)
        
        rc_pc = w_lambda / total_risk if total_risk > 0 else np.zeros(n)
        
        # 风险贡献差异平方和
        total = 0
        for i in range(n):
            for j in range(n):
                total += (rc_pc[i] - rc_pc[j]) ** 2
        return total
    
    # 约束条件: w_pc'1=1, w_pc≥0
    constraints = {'type': 'eq', 'fun': lambda w_pc: np.sum(w_pc) - 1}
    bounds = [(0, 1) for _ in range(n)]
    
    # 初始权重: 等权重
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    # 将主成分权重反推回原资产权重
    w_pc = result.x
    w = E @ w_pc
    
    # 确保权重非负并归一化
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    
    return w


def half_life_weights(returns, half_life=120):
    """
    半衰加权 (Half-Life Weighting)
    近期数据权重更高，权重按指数衰减
    
    Args:
        returns: 收益率数据 (DataFrame)
        half_life: 半衰期 (交易日)
    
    Returns:
        pd.DataFrame: 加权后的收益率
    """
    n = len(returns)
    
    # 计算权重: w_t = 2^(-t/half_life)
    decay = np.log(2) / half_life
    t = np.arange(n)
    weights = np.exp(-decay * (n - 1 - t))  # 最近的权重最高
    
    # 归一化权重
    weights = weights / weights.sum()
    
    # 加权收益率
    weighted_returns = returns.copy()
    for col in weighted_returns.columns:
        weighted_returns[col] = returns[col].values * weights
    
    return weighted_returns


def hpcrp_weights(returns, half_life=120):
    """
    半衰主成分风险平价资产配置法 (Half-Life Principal Component Risk Parity, HPCRP)
    
    步骤:
    1. 对收益率应用半衰加权
    2. 计算加权后的协方差矩阵
    3. 对加权协方差应用主成分风险平价模型
    
    Args:
        returns: 收益率数据 (DataFrame)
        half_life: 半衰期 (交易日)
    
    Returns:
        np.ndarray: 半衰主成分风险平价权重向量
    """
    # 应用半衰加权
    weighted_returns = half_life_weights(returns, half_life)
    
    # 计算加权协方差矩阵
    cov = weighted_returns.cov().values
    n = len(cov)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 主成分变换矩阵
    E = eigenvectors
    
    # 目标函数
    def objective(w_pc):
        w = E @ w_pc
        w_lambda = w_pc * eigenvalues
        total_risk = np.sum(w_lambda)
        rc_pc = w_lambda / total_risk if total_risk > 0 else np.zeros(n)
        
        total = 0
        for i in range(n):
            for j in range(n):
                total += (rc_pc[i] - rc_pc[j]) ** 2
        return total
    
    constraints = {'type': 'eq', 'fun': lambda w_pc: np.sum(w_pc) - 1}
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)
    
    w_pc = result.x
    w = E @ w_pc
    
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    
    return w


def get_model_weights(model_name, returns, **kwargs):
    """
    获取指定模型的权重向量
    
    Args:
        model_name: 模型名称 ('EW', 'EV', 'MV', 'MD', 'RP', 'PCRP', 'HPCRP')
        returns: 收益率数据 (DataFrame)
        **kwargs: 其他参数 (如 half_life)
    
    Returns:
        np.ndarray: 权重向量
    """
    models = {
        'EW': lambda r: equal_weight(len(r.columns)),
        'EV': equal_volatility_weights,
        'MV': minimum_variance_weights,
        'MD': maximum_diversification_weights,
        'RP': risk_parity_weights,
        'PCRP': principal_component_risk_parity_weights,
        'HPCRP': lambda r: hpcrp_weights(r, half_life=kwargs.get('half_life', 120))
    }
    
    if model_name not in models:
        raise ValueError(f"未知的模型: {model_name}")
    
    return models[model_name](returns)


# 模型名称映射 (支持中英文)
MODEL_NAMES = {
    '1/N': 'EW',
    'EW': 'EW',
    'Equal Weight': 'EW',
    '等权重': 'EW',
    
    'EV': 'EV',
    'Equal Volatility': 'EV',
    '等波动率': 'EV',
    
    'MV': 'MV',
    'Minimum Variance': 'MV',
    '最小方差': 'MV',
    
    'MD': 'MD',
    'Maximum Diversification': 'MD',
    '最大分散化': 'MD',
    
    'RP': 'RP',
    'Risk Parity': 'RP',
    '风险平价': 'RP',
    
    'PCRP': 'PCRP',
    'Principal Component Risk Parity': 'PCRP',
    '主成分风险平价': 'PCRP',
    
    'HPCRP': 'HPCRP',
    'Half-Life PCRP': 'HPCRP',
    '半衰主成分风险平价': 'HPCRP'
}


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)
    test_returns = pd.DataFrame(
        np.random.randn(240, 4),
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4']
    )
    
    print("测试各种模型:")
    for model in ['EW', 'EV', 'MV', 'MD', 'RP', 'PCRP', 'HPCRP']:
        if model == 'HPCRP':
            w = get_model_weights(model, test_returns, half_life=120)
        else:
            w = get_model_weights(model, test_returns)
        print(f"{model}: {w.round(4)}")

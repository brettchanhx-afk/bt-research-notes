import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def load_data(file_path="data/投资者情绪指数数据.xlsx"):
    df = pd.read_excel(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def compute_gsisi_signal(df, window=20):
    data = df.copy()
    data["ma"] = data.iloc[:, 0].rolling(window=window).mean()
    data["signal"] = np.where(data.iloc[:, 0] > data["ma"], 1, 0)
    return data

def backtest(df_signal):
    df = df_signal.copy()
    df["ret"] = df.iloc[:, 0].pct_change()
    df["strategy"] = df["ret"] * df["signal"].shift(1)

    df["cum_index"] = (1 + df["ret"]).cumprod()
    df["cum_strategy"] = (1 + df["strategy"]).cumprod()

    annual_return = df["strategy"].mean() * 252
    sharpe = df["strategy"].mean() / df["strategy"].std() * np.sqrt(252) if df["strategy"].std() != 0 else 0

    print("===== 策略回测结果 =====")
    print(f"年化收益：{annual_return:.2%}")
    print(f"夏普比率：{sharpe:.2f}")
    print(f"基准累计收益：{df['cum_index'].iloc[-1]:.2%}")
    print(f"策略累计收益：{df['cum_strategy'].iloc[-1]:.2%}")
    return df

def plot_result(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df["cum_index"], label="沪深300 基准", color="#1f77b4")
    plt.plot(df["cum_strategy"], label="情绪指数择时策略", color="#ff7f0e")
    plt.title("国信投资者情绪指数(GSISI)择时模型", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
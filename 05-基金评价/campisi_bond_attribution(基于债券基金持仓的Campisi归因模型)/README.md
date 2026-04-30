# Campisi 债券基金业绩归因模型

基于华泰金工研报《基于债券基金持仓的Campisi归因模型》复现。

## 📚 研报来源

华泰证券研究所 - 金工研究/深度研究 | 2020年08月21日

## 🎯 模型原理

Campisi模型将债券基金收益率分解为三部分：

```
R = 票息收益 + 国债利率变化效应 + 信用利差变化效应
  = y × dt + (-MD) × dy_treasury + (-MD) × dy_credit
```

### 核心公式

**单只债券收益率分解：**
- **y**：期初债券到期收益率
- **dt**：期初距上一次付息的时间间隔比例
- **MD**：期初修正久期（Modified Duration）
- **dy_treasury**：期间国债利率变化
- **dy_credit**：期间信用利差变化

**债券基金收益率分解：**
```
Σ(w_i × R_i) = Σ(w_i × y_i × dt) + Σ(w_i × (-MD_i) × dy_treasury,i) + Σ(w_i × (-MD_i) × dy_credit,i)
```

### 三部分收益贡献

1. **票息效应** = Σ(w_i × y_i × dt) / Σ(w_i × R_i)
2. **国债利率变化效应** = Σ(w_i × (-MD_i) × dy_treasury,i) / Σ(w_i × R_i)
3. **信用利差变化效应** = Σ(w_i × (-MD_i) × dy_credit,i) / Σ(w_i × R_i)

## 📁 项目结构

```
campisi_bond_attribution/
├── data/                    # 数据目录
│   ├── bond_holdings/       # 债券持仓数据
│   └── yield_curves/        # 收益率曲线数据
├── source/                  # 核心源码
│   ├── __init__.py
│   ├── data_loader.py       # 数据获取模块
│   ├── bond_analytics.py    # 债券分析工具（久期、凸性等）
│   ├── yield_curve.py       # 收益率曲线处理
│   ├── campisi_model.py     # Campisi归因模型核心
│   └── plot.py              # 可视化模块
├── output/                  # 输出结果
├── ipynb/                   # Jupyter Notebook演示
│   └── campisi_demo.ipynb
├── config.py                # 配置文件
├── main.py                  # 主程序入口
└── README.md                # 项目说明
```

## 🔧 数据源

优先级（根据用户要求）：
1. **tushare** - 债券基本信息、收益率曲线
2. **efinance** - 债券行情数据
3. **akshare** - 债券久期、信用评级
4. **baostock** - 国债收益率曲线

## 🚀 快速开始

```python
# 运行完整归因分析
python main.py

# 或在Jupyter中交互式分析
jupyter notebook ipynb/campisi_demo.ipynb
```

## 📊 输出结果

- 票息效应贡献
- 国债利率变化效应贡献（久期配置能力）
- 信用利差变化效应贡献（券种配置+个债选择能力）
- 归因分解图表

## 📖 参考文献

- Wagner-Tito模型：债券归因基础模型
- 加权久期分析方法：Van Breukelen (2000)
- Campisi模型：本文核心方法

# HPCRP 全球资产配置项目 - 完成状态报告

## 项目概述
复现天风证券研报《基于半衰主成分风险平价模型的全球资产配置策略研究》

## 项目结构
```
hpcrp_asset_allocation/
├── source/
│   ├── data_loader.py    # 数据获取 (baostock/yfinance/efinance)
│   ├── models.py         # 7种资产配置模型
│   ├── backtest.py      # 回测引擎
│   └── plot.py          # 可视化
├── ipynb/
│   └── reproduce.ipynb  # Jupyter复现
├── config.py           # 配置
├── main.py            # 主程序
└── README.md         # 说明文档
```

## 已完成模块
1. **data_loader.py** - 全球指数数据获取 (支持A股/港股/美股/欧股)
2. **models.py** - 7种资产配置模型:
   - EW (等权重)
   - EV (等波动率)
   - MV (最小方差)
   - MD (最大分散化)
   - RP (风险平价)
   - PCRP (主成分风险平价)
   - HPCRP (半衰主成分风险平价)
3. **backtest.py** - 季度调仓回测引擎
4. **plot.py** - 净值曲线/热力图/风险贡献图
5. **reproduce.ipynb** - 完整复现notebook

## 运行状态
- 框架可运行
- API获取受限时使用模拟数据

## 待解决
1. YFinance限流问题 - 需要代理或配置
2. efinance接口变更 - 需要更新调用方式

## 使用方法
```bash
cd hpcrp_asset_allocation
python run_backtest.py
```

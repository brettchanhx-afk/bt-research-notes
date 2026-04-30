# 威廉·夏普风格分析模型 (Sharpe Style Analysis)

复现华泰证券2020年金工研报《威廉·夏普风格分析基于基金收益率，能实现对基金风格的高频跟踪》

## 核心算法

### 1. 风格暴露求解

通过带约束的二次规划最小化残差平方和：

```
min Σ(R - Σβ_j * x_j)²
s.t. Σβ_j = 1, 0 ≤ β_j ≤ 1
```

其中：
- **R**: 基金收益率序列
- **x_j**: 第j个风格指数收益率序列
- **β_j**: 基金在第j个风格指数上的暴露系数

### 2. SDS风格漂移指标 (Idzorek方法)

将研究期间划分为多个子区间，计算风格暴露的波动率：

```
SDS = √[Var(β₁) + Var(β₂) + ... + Var(βₙ)]
```

**SDS解读：**
- SDS < 0.1: 风格高度稳定 ✓
- 0.1 ≤ SDS < 0.2: 风格相对稳定
- 0.2 ≤ SDS < 0.3: 风格存在一定波动 ⚠
- SDS ≥ 0.3: 风格漂移风险较高 ⚠⚠

## 项目结构

```
sharpe_style_analysis/
├── data/                           # 数据存放目录
├── ipynb/
│   └── sharpe_style_analysis.ipynb # Jupyter完整复现
├── source/
│   ├── __init__.py                 # 包初始化
│   ├── data_loader.py              # 数据获取模块
│   ├── factor.py                   # 风格分析核心算法
│   ├── backtest.py                 # 风格漂移检测
│   ├── plot.py                     # 可视化模块
│   └── utils.py                    # 工具函数
├── main.py                         # 命令行入口
└── README.md                       # 项目说明
```

## 安装依赖

```bash
pip install pandas numpy matplotlib scipy

# 数据获取（按优先级）
pip install efinance akshare baostock
```

## 使用方法

### 1. 命令行使用

```bash
# 分析单只基金
python main.py 021181

# 指定时间范围
python main.py 021181 --start_date 20230101 --end_date 20241231

# 使用模拟数据测试
python main.py 021181 --mock

# 自定义风格指数
python main.py 021181 --style_indices 000300.SH 000905.SH 000918.SH 000919.SH
```

### 2. Jupyter Notebook

打开 `ipynb/sharpe_style_analysis.ipynb` 进行交互式分析：

```bash
jupyter notebook ipynb/sharpe_style_analysis.ipynb
```

### 3. Python API

```python
from source.data_loader import StyleDataLoader
from source.factor import SharpeStyleModel

# 加载数据
loader = StyleDataLoader()
fund_df = loader.get_fund_nav('021181', '20230101', '20241231')
index_df = loader.get_index_returns(['000300.SH', '000905.SH'], '20230101', '20241231')

# 风格分析
model = SharpeStyleModel(['000300.SH', '000905.SH'])
result = model.fit(fund_df['daily_return'], index_df)

print("风格暴露:", result['exposures'])
print("R²:", result['r_squared'])
```

## 风格指数选择

根据研报推荐，常用的风格指数组合：

| 指数代码 | 指数名称 | 风格属性 |
|---------|---------|---------|
| 000300.SH | 沪深300 | 大盘 |
| 000905.SH | 中证500 | 中盘 |
| 000918.SH | 沪深300成长 | 成长 |
| 000919.SH | 沪深300价值 | 价值 |
| 000920.SH | 中证500成长 | 中小盘成长 |
| 000921.SH | 中证500价值 | 中小盘价值 |

## 输出结果

运行后会生成以下文件：

```
output/{fund_code}/
├── style_exposure.png       # 风格暴露条形图
├── return_comparison.png    # 实际vs拟合收益对比
├── sds_analysis.png         # SDS风格漂移分析
├── style_report.txt         # 文本分析报告
└── results.json             # JSON格式结果
```

## 核心功能

### 1. 风格暴露分析
- 求解基金在各风格指数上的暴露系数
- 判定基金主要风格标签
- 计算模型拟合优度(R²)和跟踪误差

### 2. 风格漂移检测
- 划分多个子区间进行滚动分析
- 计算SDS风格漂移指标
- 识别风格漂移事件

### 3. 可视化
- 风格暴露条形图
- 实际vs拟合收益对比
- 风格暴露时序变化
- SDS分析图

### 4. 绩效评估
- 年化收益率/波动率
- 夏普比率
- 最大回撤
- 卡玛比率

## 应用场景

1. **基金风格识别**: 判断基金实际投资风格是否与宣称一致
2. **风格漂移监控**: 持续跟踪基金风格变化，及时发现漂移
3. **FOF组合构建**: 基于风格暴露进行组合配置和风险分散
4. **基金经理评价**: 评估风格稳定性和一致性

## 注意事项

1. **数据质量**: 确保基金净值和指数数据完整，缺失数据会影响分析结果
2. **指数选择**: 风格指数的选择会显著影响分析结果，需根据分析目的合理选择
3. **R²解释**: R²较低说明基金风格不明显或模型解释力不足，需谨慎解读
4. **时间窗口**: SDS指标需要足够长的历史数据（建议≥1年）才有统计意义
5. **约束条件**: 本模型采用Σβ=1且0≤β≤1的约束，确保暴露系数可解释为配置比例

## 参考研报

华泰证券 - 《威廉·夏普风格分析基于基金收益率，能实现对基金风格的高频跟踪》(2020年08月21日)

## License

MIT License

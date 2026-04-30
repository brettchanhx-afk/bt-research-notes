from typing import Sequence, Tuple, Union
import matplotlib.gridspec as plot_grid
import matplotlib.pyplot as visual_plot
import mplfinance as finance_plot
import pandas as price_dataframe
from mplfinance import create_market_colors, create_mpf_style


class LayoutGridChart:
    """
    简化网格子图的创建与管理，实现多子图快速布局
    """

    def __init__(self, row_count: int, col_count: int, figure_size: Tuple):
        self.row_count = row_count
        self.col_count = col_count
        self.chart_figure = visual_plot.figure(figsize=figure_size)
        self.grid_layout = plot_grid.GridSpec(row_count, col_count, wspace=0.4, hspace=0.3)
        self.current_row_idx = 0
        self.current_col_idx = 0

    def switch_full_row(self):
        """切换至下一行整行子图，重置列索引"""
        if self.current_col_idx != 0:
            self.current_row_idx = self.current_row_idx + 1
            self.current_col_idx = 0
        axis_subplot = visual_plot.subplot(self.grid_layout[self.current_row_idx, :])
        self.current_row_idx = self.current_row_idx + 1
        return axis_subplot

    def switch_single_cell(self):
        """切换至下一个单元格子图，自动换行"""
        if self.current_col_idx >= self.col_count:
            self.current_row_idx = self.current_row_idx + 1
            self.current_col_idx = 0
        axis_subplot = visual_plot.subplot(self.grid_layout[self.current_row_idx, self.current_col_idx])
        self.current_col_idx = self.current_col_idx + 1
        return axis_subplot

    def release_resources(self):
        """关闭图表并释放内存资源"""
        visual_plot.close(self.chart_figure)
        self.chart_figure = None
        self.grid_layout = None


# 定义A股风格配色：上涨红色，下跌绿色
CHINA_MARKET_STYLE = create_mpf_style(marketcolors=create_market_colors(up="r", down="g"))


def render_multi_symbol_charts(
    symbol_list: Union[Sequence, str],
    ohlc_dataset: price_dataframe.DataFrame,
    row_num: int,
    col_num: int,
    figure_dim: Tuple,
    recent_period: int = 60,
) -> visual_plot.Figure:
    """
    批量渲染多个标的的K线网格图
    参数：
        symbol_list: 标的代码列表/元组/单个代码
        ohlc_dataset: 包含多标的OHLC的数据集
        row_num: 网格行数
        col_num: 网格列数
        figure_dim: 图表尺寸
        recent_period: 展示最近N根K线
    返回：
        绘制完成的图表对象
    """
    grid_manager = LayoutGridChart(row_count=row_num, col_count=col_num, figure_size=figure_dim)

    # 遍历所有标的并绘制对应K线图
    for symbol_code in symbol_list:
        # 提取当前标的的行情数据
        filtered_data = ohlc_dataset.xs(symbol_code, level=1)
        # 绘制蜡烛图
        finance_plot.plot(
            filtered_data.iloc[-recent_period:],
            type="candle",
            axtitle=symbol_code,
            style=CHINA_MARKET_STYLE,
            datetime_format="%Y%m%d",
            ax=grid_manager.switch_single_cell(),
        )

    # 展示所有绘制完成的图表
    visual_plot.show()
    # 释放资源
    grid_manager.release_resources()
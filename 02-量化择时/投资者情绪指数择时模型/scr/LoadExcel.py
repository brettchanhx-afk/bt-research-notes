import pandas as pd
import os

class LoadData(object):
    def __init__(self):
        # 路径拼接逻辑保持不变，仅调整导入位置（可选，优化导入结构）
        this_file_path: str = os.path.split(os.path.dirname(__file__))[0]
        self._excel_file = pd.ExcelFile(f"{this_file_path}/data/投资者情绪指数数据.xlsx")

    def _parse_sheet(self, sheet_name: str, index_col: list = None, parse_dates: list = None) -> pd.DataFrame:
        """
        通用的Excel工作表解析方法，提取重复解析逻辑
        :param sheet_name: 工作表名称
        :param index_col: 作为索引的列
        :param parse_dates: 需要解析为日期的列
        :return: 解析后的DataFrame
        """
        # 设置默认值，避免重复传参
        # 空值替换为默认空列表
        index_col = index_col or []
        parse_dates = parse_dates or []
        return self._excel_file.parse(
            sheet_name,
            index_col=index_col,
            parse_dates=parse_dates
        )

    @property
    def index_price(self):
        return self._parse_sheet(
            sheet_name="index_price",
            index_col=[0],
            parse_dates=["trade_date"]
        )

    @property
    def sw_classify(self):
        return self._parse_sheet(
            sheet_name="sw_classify",
            index_col=[0]
        )

    @property
    def pivot_swprice(self):
        # 先解析数据，读取申万价格原始数据
        self.sw_price: pd.DataFrame = self._parse_sheet(
            sheet_name="sw_price",
            index_col=[0],
            parse_dates=["trade_date"]
        )
        # 透视表逻辑
        return pd.pivot_table(
            self.sw_price, index="trade_date", columns="code", values="close"
        )
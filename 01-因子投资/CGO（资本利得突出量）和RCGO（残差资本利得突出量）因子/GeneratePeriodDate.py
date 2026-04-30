from jqdata import *
import pandas as pd

# 日期相关工具依赖
import calendar
from dateutil.parser import parse
import datetime

# 迭代工具库
import itertools


###########################  日期周期处理 ###############################

class PeriodDateGenerator(object):
    """根据预设调仓规则，生成策略对应的调仓时间区间"""

    def __init__(self, start_time=None, end_time=None):
        # 初始化时校验并存储起止日期
        if start_time and end_time:
            self._validate_date_format(start_time, end_time)

    @property
    def generate_time_intervals(self):
        """生成连续的调仓时间区间列表"""
        position_dates = self.build_position_dates()
        paired_dates = list(zip(position_dates[:-1], position_dates[1:]))

        # 首段区间直接使用，后续区间起始日期向后偏移一个交易日
        formatted_intervals = []
        for idx, date_pair in enumerate(paired_dates):
            if idx == 0:
                formatted_intervals.append((date_pair[0], date_pair[1]))
            else:
                formatted_intervals.append((shift_trade_day(date_pair[0], 1), date_pair[1]))
        return formatted_intervals

    # 构建指定范围内的所有调仓日期
    def build_position_dates(self, config: dict = {"months": (6, 12), "weekday": "Friday", 'spec_weekday': "2nd"}) -> list:
        """
        依据配置参数生成年度调仓日期
        start: 格式 YYYY-MM-DD
        end: 格式 YYYY-MM-DD
        =================
        返回值：datetime.date 类型的日期列表
        """
        # 获取起止年份范围
        start_year = self.__start_date.year
        end_year = self.__end_date.year

        # 解析配置参数
        target_months = config['months']
        target_weekday = config['weekday']
        weekday_sequence = config['spec_weekday']

        position_date_list = []
        # 遍历所有年份与目标月份组合
        for year_val in range(start_year, end_year + 1):
            for month_val in target_months:
                target_date = self.locate_target_weekday(year_val, month_val, target_weekday, weekday_sequence)
                position_date_list.append(target_date)

        # 添加起止日期并排序
        position_date_list.append(self.__start_date)
        position_date_list.append(self.__end_date)
        position_date_list.sort()

        # 过滤出在有效时间范围内的日期
        valid_dates = []
        for date_item in position_date_list:
            if (date_item >= self.__start_date) and (date_item <= self.__end_date):
                valid_dates.append(date_item)
        return valid_dates

    def _validate_date_format(self, start_time, end_time):
        """校验并转换输入的日期格式为标准日期对象"""
        if isinstance(start_time, (str, int)):
            self.__start_date = parse(start_time).date()

        if isinstance(end_time, (str, int)):
            self.__end_date = parse(end_time).date()

    @staticmethod
    def locate_target_weekday(year, month, weekday_name, sequence) -> datetime.date:
        """
        定位指定年月的第N个指定星期几
        示例：locate_target_weekday(2019, 12, "Friday", "2nd")
        ================
        返回值：对应日期的 datetime.date 对象
        """
        week_day_mapping = [day for day in calendar.day_name]
        target_week_index = week_day_mapping.index(weekday_name)

        # 获取当月所有目标星期几的日期
        valid_days = []
        for week in calendar.monthcalendar(year, month):
            day_val = week[target_week_index]
            if day_val != 0:
                valid_days.append(day_val)

        # 根据规则匹配对应日期
        if sequence == 'teenth':
            for day_value in valid_days:
                if 13 <= day_value <= 19:
                    return datetime.date(year, month, day_value)

        if sequence == 'last':
            select_index = -1
        elif sequence == 'first':
            select_index = 0
        else:
            select_index = int(sequence[0]) - 1

        return datetime.date(year, month, valid_days[select_index])


def shift_trade_day(base_date: str, offset: int) -> datetime.date:
    """
    基于交易日进行日期偏移
    base_date: 基准交易日
    offset: 正数向后推移，负数向前推移
    -----------
    返回值：偏移后的交易日日期对象
    """
    # 获取基准交易日
    base_trade_date = get_trade_days(end_date=base_date, count=1)[0]

    if offset > 0:
        # 向后偏移指定交易日数量
        full_trade_calendar = get_all_trade_days().tolist()
        base_index = full_trade_calendar.index(base_trade_date)
        return full_trade_calendar[base_index + offset]

    if offset < 0:
        # 向前偏移指定交易日数量
        return get_trade_days(end_date=base_trade_date, count=abs(offset))[0]
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Indicator:
    name: str
    category: str
    logic_direction: int
    delay_months: int
    data_type: str
    source: str


INDICATOR_CATEGORIES = {
    "total": "总量类",
    "price": "价格类",
    "yoy_ratio": "同比类/比率类",
    "diffusion": "扩散类"
}


INDUSTRY_INDICATOR_LIB = {
    "石油石化": [
        {"name": "石油和天然气开采业:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "石油、煤炭及其他燃料加工业:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "进口平均单价:成品油(海关口径):当月值", "category": "price", "logic": 1, "delay": 1},
        {"name": "出口平均单价:柴油:当月值", "category": "price", "logic": 1, "delay": 1},
        {"name": "期货收盘价(连续):IPE布油", "category": "price", "logic": 1, "delay": 0},
        {"name": "石油和天然气开采业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPI:石油和天然气开采业:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPI:石油、煤炭及其他燃料加工业:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "秦皇岛港:平仓价:动力煤(Q5000K)", "category": "price", "logic": 1, "delay": 0},
        {"name": "Myspic综合钢价指数", "category": "price", "logic": 1, "delay": 0},
    ],
    "煤炭": [
        {"name": "PPI:煤炭及炼焦工业:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "秦皇岛港:平仓价:动力煤(Q5000K)", "category": "price", "logic": 1, "delay": 0},
        {"name": "坑口价(含税):无烟煤(A15-20%,V7-10%,0.6%S,Q5600):晋城:阳城", "category": "price", "logic": 1, "delay": 0},
        {"name": "出口价格指数(SITC2):煤、焦炭及煤砖", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "煤炭开采和洗选业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "煤炭开采和洗选业:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "原煤产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "发电量:当月值", "category": "total", "logic": 1, "delay": 1},
    ],
    "有色金属": [
        {"name": "期货收盘价:LME基本金属指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "PPIRM:有色金属材料类:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPI:金属制品业:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "期货官方价:LME3个月锡", "category": "price", "logic": 1, "delay": 0},
        {"name": "期货官方价:LME3个月铝", "category": "price", "logic": 1, "delay": 0},
        {"name": "有色金属冶炼及压延加工业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "期货官方价:LME3个月铜", "category": "price", "logic": 1, "delay": 0},
        {"name": "期货官方价:LME3个月锌", "category": "price", "logic": 1, "delay": 0},
        {"name": "有色金属冶炼及压延加工业:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
    ],
    "钢铁": [
        {"name": "钢铁行业采购经理人指数(PMI):全国", "category": "diffusion", "logic": 1, "delay": 0},
        {"name": "PPI:黑色金属冶炼及压延加工业:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPIRM:黑色金属材料类:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "中国大宗商品价格指数:钢铁类", "category": "price", "logic": 1, "delay": 0},
        {"name": "出口价格指数:黑色金属冶炼业及压延加工", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "黑色金属冶炼及压延加工业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "Myspic综合钢价指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "螺纹钢期货收盘价", "category": "price", "logic": 1, "delay": 0},
    ],
    "基础化工": [
        {"name": "PMI:原材料库存", "category": "diffusion", "logic": 1, "delay": 0},
        {"name": "PPI:化学纤维制造业:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPIRM:化工原料类:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "现货价:甲醛:国内", "category": "price", "logic": 1, "delay": 0},
        {"name": "现货价(中间价):磷酸一铵(散装):FOB波罗的海", "category": "price", "logic": 1, "delay": 0},
        {"name": "化学原料及化学制品制造业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "化工行业PPI:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "化学原料及化学制品制造业:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
    ],
    "建材": [
        {"name": "PPI:非金属矿物制品业:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "PPIRM:建筑材料类:环比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "水泥价格指数:全国", "category": "price", "logic": 1, "delay": 0},
        {"name": "浮法平板玻璃价格指数:全国", "category": "price", "logic": 1, "delay": 0},
        {"name": "非金属矿物制品业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "建材行业PPI:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "水泥产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "平板玻璃产量:当月值", "category": "total", "logic": 1, "delay": 1},
    ],
    "机械": [
        {"name": "产量:谷物收获机械:当月值", "category": "total", "logic": 1, "delay": 2},
        {"name": "期货收盘价:LME基本金属指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "销量:叉车:全行业:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "销量:内燃机:当月值", "category": "total", "logic": 1, "delay": 2},
        {"name": "销量:平地机:主要企业:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "专用设备制造业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "通用设备制造业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "机械行业PMI:当月值", "category": "diffusion", "logic": 1, "delay": 0},
    ],
    "电力设备及新能源": [
        {"name": "固定资产投资完成额:制造业:电气机械及器材制造业:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "电源基本建设投资完成额:核电:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "电源建设新增生产能力:全国:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "产量:高压开关设备(11万伏以上):当月值", "category": "total", "logic": 1, "delay": 2},
        {"name": "光伏行业综合价格指数(SPI):电池片", "category": "price", "logic": 1, "delay": 0},
        {"name": "现货价(周平均价):太阳能电池", "category": "price", "logic": 1, "delay": 0},
        {"name": "发电设备产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "太阳能电池板价格指数", "category": "price", "logic": 1, "delay": 0},
    ],
    "国防军工": [
        {"name": "工业增加值:铁路、船舶、航空航天和其他运输设备制造业:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "出口金额:稀土及其制品:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "出口金额:碳纤维(68159920):当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "进口金额:涡轮喷气发动机:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "进口均价:电容器:当月值", "category": "price", "logic": 1, "delay": 1},
        {"name": "铁路、船舶、航空航天和其他运输设备制造业:净资产营业利润率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "军工行业PMI:当月值", "category": "diffusion", "logic": 1, "delay": 0},
        {"name": "国防支出预算:当月值", "category": "total", "logic": 1, "delay": 1},
    ],
    "汽车": [
        {"name": "PMI:在手订单", "category": "diffusion", "logic": 1, "delay": 0},
        {"name": "二手车平均交易价格:当月值", "category": "price", "logic": 1, "delay": 1},
        {"name": "汽车制造:亏损企业单位数", "category": "total", "logic": -1, "delay": 1},
        {"name": "汽车制造:营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "汽车产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "汽车销量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "乘用车销量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "商用车销量:当月值", "category": "total", "logic": 1, "delay": 1},
    ],
    "家电": [
        {"name": "家电行业PPI:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "家电零售额:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "空调产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "冰箱产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "洗衣机产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "家用电器行业营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "家电行业净利润:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "家电行业毛利率", "category": "yoy_ratio", "logic": 1, "delay": 1},
    ],
    "酒类": [
        {"name": "白酒产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "啤酒产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "葡萄酒产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "酒类零售额:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "白酒价格指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "粮食价格指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "酒类行业营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "酒类行业净利润:累计值", "category": "total", "logic": 1, "delay": 1},
    ],
    "饮料": [
        {"name": "饮料产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "碳酸饮料产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "果汁饮料产量:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "饮料行业营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "饮料行业净利润:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "饮料行业毛利率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "软饮料价格指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "包装材料价格指数", "category": "price", "logic": 1, "delay": 0},
    ],
    "食品": [
        {"name": "食品行业PPI:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "食品制造业营业收入:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "食品制造业净利润:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "食品制造业毛利率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "粮食产量:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "食用油价格指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "猪肉价格指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "食品行业ROE_TTM", "category": "yoy_ratio", "logic": 1, "delay": 1},
    ],
    "房地产": [
        {"name": "房地产开发投资完成额:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "商品房销售面积:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "商品房销售额:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "房屋新开工面积:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "土地购置面积:累计值", "category": "total", "logic": 1, "delay": 1},
        {"name": "房地产行业资产负债率", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "房地产行业ROE", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "百城住宅价格指数", "category": "price", "logic": 1, "delay": 0},
    ],
    "电子": [
        {"name": "费城半导体指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "中国台湾电子行业指数", "category": "price", "logic": 1, "delay": 0},
        {"name": "市场价:浮法平板玻璃:4.8/5mm:全国", "category": "price", "logic": 1, "delay": 1},
        {"name": "市场价:单晶硅片(156mm×156mm,一线厂商):国内", "category": "price", "logic": 1, "delay": 1},
        {"name": "台股营收:IC封装测试:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "北美半导体设备制造商:出货额:当月值", "category": "total", "logic": 1, "delay": 1},
        {"name": "电子信息制造业PPI:当月同比", "category": "yoy_ratio", "logic": 1, "delay": 1},
        {"name": "电子行业净利润:累计值", "category": "total", "logic": 1, "delay": 1},
    ],
}


class IndicatorLibrary:
    def __init__(self):
        self.industry_indicators = INDUSTRY_INDICATOR_LIB

    def get_industry_indicators(self, industry_name: str) -> List[Dict]:
        return self.industry_indicators.get(industry_name, [])

    def get_all_industries(self) -> List[str]:
        return list(self.industry_indicators.keys())

    def get_indicator_count(self, industry_name: str) -> int:
        return len(self.industry_indicators.get(industry_name, []))

    def get_indicators_by_category(
        self,
        industry_name: str,
        category: str
    ) -> List[Dict]:
        indicators = self.get_industry_indicators(industry_name)
        return [ind for ind in indicators if ind["category"] == category]

    def validate_indicator_data(
        self,
        data: pd.Series,
        min_length: int = 36
    ) -> Tuple[bool, int]:
        valid_length = data.dropna().shape[0]
        is_valid = valid_length >= min_length
        return is_valid, valid_length

    def get_indicator_info(self, industry_name: str) -> pd.DataFrame:
        indicators = self.get_industry_indicators(industry_name)
        if not indicators:
            return pd.DataFrame()

        df = pd.DataFrame(indicators)
        df['category_name'] = df['category'].map(
            lambda x: INDICATOR_CATEGORIES.get(x, x)
        )
        return df


def get_citici_code_mapping() -> Dict[str, str]:
    return {
        "石油石化": "CI005001.WI",
        "煤炭": "CI005002.WI",
        "有色金属": "CI005003.WI",
        "钢铁": "CI005005.WI",
        "基础化工": "CI005006.WI",
        "建材": "CI005008.WI",
        "机械": "CI005010.WI",
        "电力设备及新能源": "CI005011.WI",
        "国防军工": "CI005012.WI",
        "汽车": "CI005013.WI",
        "家电": "CI005016.WI",
        "酒类": "CI005156.WI",
        "饮料": "CI005822.WI",
        "食品": "CI005823.WI",
        "房地产": "CI005023.WI",
        "电子": "CI005025.WI",
    }


def main():
    lib = IndicatorLibrary()
    print("Industry Indicator Library:")
    for industry in lib.get_all_industries():
        count = lib.get_indicator_count(industry)
        print(f"  {industry}: {count} indicators")

    info = lib.get_indicator_info("电子")
    print(f"\n电子行业指标库:\n{info}")


if __name__ == "__main__":
    main()

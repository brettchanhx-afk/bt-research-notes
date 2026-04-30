from typing import Dict


def get_value_from_traderanalyzerdict(dic: Dict, *args) -> float:
    """从嵌套字典中获取指定键对应的值"""
    # 若参数长度为1，直接获取对应值，默认返回0
    if len(args) == 1:
        return dic.get(args[0], 0.0)
    
    # 遍历参数中的键，逐层查找嵌套字典
    for key in args:
        # 获取当前层级的字典值，不存在则返回0
        nested_value = dic.get(key, None)
        if nested_value is not None:
            return get_value_from_traderanalyzerdict(nested_value, *args[1:])
        return 0.0
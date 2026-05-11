"""
Tushare API 初始化和连接管理
"""

import tushare as ts
from typing import Optional
from ..config import TUSHARE_TOKEN, TUSHARE_API_URL


class TushareClient:
    """
    Tushare API 客户端封装
    """

    _instance: Optional["TushareClient"] = None
    _pro = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化Tushare Pro API"""
        self._pro = ts.pro_api(self._get_token())
        self._pro._DataApi__http_url = TUSHARE_API_URL

    def _get_token(self) -> str:
        """获取token"""
        return TUSHARE_TOKEN

    @property
    def pro(self):
        """获取pro API对象"""
        return self._pro

    def query(self, api_name: str, params: dict = None, fields: str = None):
        """
        执行Tushare查询

        Args:
            api_name: API名称
            params: 查询参数
            fields: 返回字段

        Returns:
            DataFrame
        """
        if params is None:
            params = {}
        if fields:
            params["fields"] = fields
        return self._pro.query(api_name, **params)


def get_tushare_client() -> TushareClient:
    """
    获取Tushare客户端单例

    Returns:
        TushareClient实例
    """
    return TushareClient()


def get_pro_api():
    """
    获取Tushare Pro API对象

    Returns:
        pro API对象
    """
    client = get_tushare_client()
    return client.pro

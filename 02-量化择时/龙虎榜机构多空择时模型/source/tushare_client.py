import logging
import logging.handlers
# tuhsare自动延迟下载，防止频繁调取数据是报错
import time

import tushare as ts

from .config_loader import ts_token


class TuShare:
    
    def __init__(self,
                 token: str = ts_token,
                 logger: bool = None,
                 max_retry: int = 0):
        """构造函数,token:tushare的token;logger:日志对象,可以不传；
        max_retry:最大重试次数,默认为0意为无限重试,建议用10以上100以内。"""
        self.token = token
        if not logger:
            logger = logging.getLogger('TuShare')
            # CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
            logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s %(name)s %(pathname)s:%(lineno)d %(funcName)s %(levelname)s %(message)s'
            )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
        self.logger = logger
        self.max_retry = max_retry
        ts.set_token(token)
        self.pro = ts.pro_api()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            i = 0
            while True:
                try:
                    if name == 'pro_bar':
                        m = getattr(ts, name, None)
                    else:
                        m = getattr(self.pro, name, None)
                    if m is None:
                        self.logger.error('Attribute %s does not exist.', name)
                        return None
                    else:
                        return m(*args, **kwargs)
                except (Exception):
                    if self.max_retry > 0 and i >= self.max_retry:
                        raise
                    self.logger.exception(
                        'TuShare exec %s failed, args:%s, kwargs:%s, try again.',
                        name, args, kwargs)
                    time.sleep(1)
                i += 1

        return wrapper

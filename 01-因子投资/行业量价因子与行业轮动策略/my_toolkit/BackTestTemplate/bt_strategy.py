# 基于信号的交易策略类
import backtrader as bt
class SignalStrategy(bt.Strategy):

    # 策略参数配置
    params = (
        ("open_threshold", 0.301),    # 开仓阈值
        ("close_threshold", -0.301),  # 平仓阈值
        ("show_log", True),           # 是否输出日志
    )

    def record_log(self, log_content, trade_date=None, is_show: bool = True):
        """日志记录函数，格式化输出交易相关信息"""
        trade_date = trade_date if trade_date is not None else self.datas[0].datetime.date(0)
        if is_show:
            print(f"{trade_date.isoformat()}, {log_content}")

    def __init__(self):
        # 初始化核心数据引用
        self.close_price = self.data.close
        self.trade_signal = self.data.GSISI
        self.current_order = None

    def notify_order(self, order):
        """订单状态通知处理函数"""
        # 忽略已提交/已接受但未完成的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # 处理已完成/已取消/保证金不足的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.record_log(
                    "BUY EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,                # 订单唯一标识
                        order.executed.price,     # 实际成交价格
                        order.executed.value,     # 成交总金额
                        order.executed.comm,      # 交易佣金
                        order.executed.size,      # 成交数量
                        order.data._name,         # 标的名称
                    ),
                    is_show=self.p.show_log,
                )
            else:  # 卖出订单处理
                self.record_log(
                    "SELL EXECUTED, ref:%.0f, Price: %.2f, Cost: %.2f, Comm %.2f, Size: %.2f, Stock: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                    ),
                    is_show=self.p.show_log,
                )

    def next(self):
        # 撤销未执行的挂单
        if self.current_order:
            self.cancel(self.current_order)

        # 持有仓位时的平仓逻辑
        has_position = bool(self.position)
        close_signal_meet = (self.trade_signal[0] <= self.params.close_threshold) and (self.trade_signal[-1] <= self.params.close_threshold)
        if has_position and close_signal_meet:
            self.record_log(f"收盘价Close, {self.close_price[0]:.2f}", is_show=self.p.show_log)
            self.record_log(
                "设置卖单SELL CREATE, %.2f信号为:%.2f,阈值为:%.2f"
                % (self.close_price[0], self.trade_signal[0], self.params.close_threshold),
                is_show=self.p.show_log,
            )
            self.current_order = self.order_target_value(target=0.0)

        # 无仓位时的开仓逻辑
        open_signal_meet = (self.trade_signal[0] >= self.params.open_threshold) and (self.trade_signal[-1] >= self.params.open_threshold)
        elif not has_position and open_signal_meet:
            self.record_log(f"收盘价Close, {self.close_price[0]:.2f}", is_show=self.p.show_log)
            self.record_log(
                "设置买单 BUY CREATE, %.2f,信号为:%.2f,阈值为:%.2f"
                % (self.close_price[0], self.trade_signal[0], self.params.open_threshold),
                is_show=self.p.show_log,
            )
            self.current_order = self.order_target_percent(target=0.95)
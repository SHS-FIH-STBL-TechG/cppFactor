from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform, min_forward_adj


class m_vpc_mut_ty_log7(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # fix_times = ["1000"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.open_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag=5
    reform_window=1

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation = ["drop1", "drop4"])

        # 播放的数据通过database.depend_data字典获取
        minute_close = database.depend_data['FactorData.Basic_factor.close_minute']
        minute_close = min_forward_adj(minute_close)
        minute30_close = minute_close.resample("30T", label="right").last().dropna(axis=0, how='all').iloc[-8*(self.lag):,:].values
        minute_open = database.depend_data['FactorData.Basic_factor.open_minute']
        minute_open = min_forward_adj(minute_open)
        minute30_open = minute_open.resample("30T", label="right").first().dropna(axis=0, how="all").iloc[-8*(self.lag):,:].values

        stock_ret = minute30_close / minute30_open - 1
        stock_ret = abs(stock_ret)
        stock_ret = pd.Series(stock_ret.mean(axis=0), index=minute_close.columns)
        #stock_ret = stock_ret.rolling(window=8*(self.lag)).mean()
        return stock_ret


    def reform(self, temp_result):
        
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = factor_values.rolling(window=self.reform_window, min_periods=1).mean()
        return factor_values


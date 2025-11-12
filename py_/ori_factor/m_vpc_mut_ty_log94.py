from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform,min_forward_adj
from xfactor.Util import *


class m_vpc_mut_ty_log94(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.high_minute","FactorData.Basic_factor.low_minute", "FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag=0
    reform_window=3

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])

        # 播放的数据通过database.depend_data字典获取
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        minute_high = database.depend_data['FactorData.Basic_factor.high_minute']
        minute_high = data_filter(minute_high, limit_status, method='minute')
        minute_high = min_forward_adj(minute_high)
        minute_low = database.depend_data['FactorData.Basic_factor.low_minute']
        minute_low = data_filter(minute_low, limit_status, method='minute')
        minute_low = min_forward_adj(minute_low)
        minute_high5 = minute_high.rolling(5).max()
        minute_low5 = minute_low.rolling(5).min()
        beta_high_low = minute_high5.std(axis=0) / minute_low5.std(axis=0) * minute_high5.corrwith(minute_low5, axis=0)
        return beta_high_low

    def reform(self, temp_result):
        
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = factor_values.rolling(self.reform_window, min_periods=1).mean()
        return factor_values


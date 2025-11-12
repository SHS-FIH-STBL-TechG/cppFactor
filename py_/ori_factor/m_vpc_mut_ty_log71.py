from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform, min_forward_adj
from xfactor.Util import *


class m_vpc_mut_ty_log71(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.amt_minute", "FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 0
    reform_window = 5
    n = 5

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        # 分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        # 处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation=["merge1", "merge1"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']

        # 播放的数据通过database.depend_data字典获取
        minute_amt = database.depend_data['FactorData.Basic_factor.amt_minute']
        minute_amt = min_forward_adj(minute_amt)
        minute_amt = data_filter(minute_amt, limit_status, method='minute')
        minute_amt_pct = minute_amt / minute_amt.shift(1)
        ans = minute_amt_pct.mean(axis=0) / minute_amt_pct.std(axis=0)
        return ans

    def reform(self, temp_result):
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = factor_values.rolling(self.n, min_periods=1).mean()
        return factor_values


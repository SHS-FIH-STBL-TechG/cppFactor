from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.Util import *
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform,min_forward_adj


class m_vpc_mut_ty_log80(BaseFactor):
    #  定义因子参数
#    ['FactorMinAmtSR', {'ptype': 'mean', 'Data_Base': ['play_minute_amt'], 'play_day_lag': None, 'play_min_lag': 2, 'generator_lag': 10, 'type': 1430, 'drop_callauction': True}, 'F_D_Min1430_AmtStdRet_mean10.h5'], 


    # 因子频率，默认为日频因子， 可不设置 
    factor_type = "FIX"
#    fix_times = ["1000", "1030", "1300", "1330", "1400", "1430"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.amt_minute","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    minute_lag = 1
#    lag = 1
    generator_lag = 10
    reform_window = 10
    ptype = 'mean'

    # 每次播放的计算具体方法。必须实现。8
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        amt_df = data_filter(database.depend_data['FactorData.Basic_factor.amt_minute'],limit_status,method='minute')
        minute30_amt = amt_df.replace([0], 0.001).resample("30T", label="right", fill_method=None).sum().\
        replace([0], np.nan).dropna(axis=0, how="all").iloc[-8 * self.minute_lag - 1:, :]
        delta = (minute30_amt -minute30_amt.shift(1)).iloc[1:,:]
        df_factor = delta.std(axis=0) / abs(delta).sum(axis=0)
        return df_factor


    def reform(self, temp_result):        
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        if self.ptype == "regrmse":
            min_periods = self.generator_lag
        else:
            min_periods = 1
        # 数据通过self.data_base字典获取
        factor_values = Util.rolling_process(factor_values, self.ptype, window=self.generator_lag, min_periods=min_periods)
        return factor_values



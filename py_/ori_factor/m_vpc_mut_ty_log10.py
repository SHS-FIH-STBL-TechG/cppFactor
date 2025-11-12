from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.Util import *
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform,min_forward_adj


class m_vpc_mut_ty_log10(BaseFactor):
    #  定义因子参数
#    ['FactorMinAccelerateStdRE', {'ptype': 'meandivstd', 'Data_Base': ['play_minute_close'], 'play_day_lag': None, 'play_min_lag': 6, 'generator_lag': 10, 'type': 1430, 'drop_callauction': True}, 'F_D_Min1430_AccelerateStdRE_meandivstd10.h5'], 


    # 因子频率，默认为日频因子， 可不设置 
    factor_type = "FIX"
#    fix_times = ["1000", "1030", "1100", "1300", "1330", "1400", "1430"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute", "FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    minute_lag = 5
    lag = 0
    generator_lag = 10
    reform_window = 10
    ptype = 'meandivstd'

    # 每次播放的计算具体方法。必须实现。8
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        minute_close = data_filter(database.depend_data['FactorData.Basic_factor.close_minute'],limit_status,method='minute')       
        minute_close = min_forward_adj(minute_close)

        close = minute_close.resample("5T", label="right").last().dropna(axis=0, how='all').iloc[
                       -48  * self.minute_lag - 1:, :]  
        speed = ((close - close.shift(1)) / close.shift(1)).iloc[1:,:]
        ans_df = (speed - speed.shift(1)).iloc[1:,:]
        ans_df = pd.DataFrame(ans_df.values - ans_df.mean(axis=1).values.reshape(ans_df.shape[0], 1), index=ans_df.index, columns=ans_df.columns)
        return 100 * abs(ans_df).std(axis=0)
       
    def reform(self, temp_result):
        
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = Util.rolling_process(factor_values, self.ptype, window=self.reform_window, min_periods=self.reform_window)
        return factor_values



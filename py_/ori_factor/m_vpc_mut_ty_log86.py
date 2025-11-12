from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time


class m_vpc_mut_ty_log86(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.volume_minute',
                   'FactorData.Basic_factor.amt_minute',
                   'FactorData.Basic_factor.close_adj_minute']
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 1
    minute_lag = 1
    reform_window = 5
    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteVolume = database.depend_data['FactorData.Basic_factor.volume_minute']
        MinuteTurnover = database.depend_data['FactorData.Basic_factor.amt_minute']
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_adj_minute']

        avg_price = MinuteClose.mean()
        vwap = pd.Series(MinuteTurnover.sum().values / MinuteVolume.sum().values, index=MinuteVolume.columns)
        ans = pd.Series(avg_price.values / vwap.values, index=vwap.index)
        return ans


    def  reform(self, temp_result):
        A = temp_result.rolling(5, min_periods=1).mean() / temp_result.rolling(5, min_periods=1).std()
        # A = pd.DataFrame(-1.*A.values, index=A.index, columns=A.columns,)
        return A
    
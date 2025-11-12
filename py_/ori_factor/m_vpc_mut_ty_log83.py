from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log83(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'
    depend_data = ['FactorData.Basic_factor.close_minute', 'FactorData.Basic_factor.open_minute',
                'FactorData.Basic_factor.volume_minute', 'FactorData.Basic_factor.amt_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=1
    # fix_times = ["1500"]
    reform_window = 15
    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        min_close = database.depend_data['FactorData.Basic_factor.close_minute'].iloc[120:]
        min_open = database.depend_data['FactorData.Basic_factor.open_minute'].iloc[120:]
        min_turn = database.depend_data['FactorData.Basic_factor.amt_minute'].iloc[120:]
        min_volume = database.depend_data['FactorData.Basic_factor.volume_minute'].iloc[120:]

        min_return = min_close.values / min_open.values - 1
        min_return[min_return<0] = np.nan
        
        min_RV = np.abs(min_return) / min_volume.values
        RV_flag = (min_RV >= np.nanpercentile(min_RV,90,axis=0))
        vwap_RV = np.nansum(min_turn.values * RV_flag, axis=0) / np.nansum(min_volume.values * RV_flag, axis=0)
        vwap_allday = min_turn.sum() / min_volume.sum()
#        vwap_pos = (min_turn*(min_return>0)).sum() / (min_volume*(min_return>0)).sum()
        df_ratio = vwap_RV / vwap_allday
        return df_ratio


    
    def reform(self, temp_result):
        A = temp_result.rolling(15,1).mean()
        return A
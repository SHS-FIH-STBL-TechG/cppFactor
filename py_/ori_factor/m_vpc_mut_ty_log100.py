from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log100(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.volume_minute', 'FactorData.Basic_factor.amt_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=0
    # fix_times = ["1300"]
    # reform_window = 20

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteVolume = database.depend_data['FactorData.Basic_factor.volume_minute']
        MinuteTurnover = database.depend_data['FactorData.Basic_factor.amt_minute']
    
        vwap = MinuteTurnover / MinuteVolume
        # r = vwap.pct_change()
        r = vwap.diff() / vwap.shift()
        m = vwap.median()
        f1 = pd.DataFrame((vwap.values < np.tile(m.values,(vwap.shape[0],1))), index=r.index, columns=r.columns)
        f2 = pd.DataFrame((vwap.values > np.tile(m.values,(vwap.shape[0],1))), index=r.index, columns=r.columns)
        s_0, s_1 = vwap[f1].std(), vwap[f2].std()
        low, high = vwap.min(), vwap.max()
        f3 = pd.DataFrame((vwap.values < np.tile((low+s_0).values,(vwap.shape[0],1))), index=r.index, columns=r.columns)
        f4 = pd.DataFrame((vwap.values > np.tile((high-s_1).values,(vwap.shape[0],1))), index=r.index, columns=r.columns)
        # return r[vwap < low + s_0].std() / r[vwap > high - s_1].std()
        alpha = r[f3].std() / r[f4].std()
        alpha[np.isinf(alpha.values)]=np.nan
        return alpha



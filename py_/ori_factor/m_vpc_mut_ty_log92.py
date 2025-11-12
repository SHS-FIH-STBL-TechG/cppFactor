from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log92(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.close_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=0
    # fix_times = ["1300"]
    # reform_window = 20

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        # MinuteVolume = database.depend_data['FactorData.Basic_factor.volume_minute']
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_minute']

        MinuteClose = MinuteClose.iloc[-90:,:]
        BAStrength = MinuteClose.diff(5) / MinuteClose.rolling(5,1).std() 
        BAStrength[np.isinf(BAStrength.values)]=np.nan
        f = BAStrength.quantile(0.95) + BAStrength.quantile(0.05)
        # f = np.nanpercentile(BAStrength,95,axis=0) + np.nanpercentile(BAStrength,5,axis=0)
        # f = pd.Series(f, index = MinuteClose.columns)
        return  f 

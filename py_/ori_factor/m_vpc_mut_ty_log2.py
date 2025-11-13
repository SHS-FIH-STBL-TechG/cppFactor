from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log2(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.volume_adj_minute', 'FactorData.Basic_factor.close_adj_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=1
    # fix_times = ["1300"]
    reform_window = 20

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteVolume = database.depend_data['FactorData.Basic_factor.volume_adj_minute']
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_adj_minute']

        MinuteVolume = MinuteVolume.iloc[-240:,:] 
        MinuteClose = MinuteClose.iloc[-240:,:]

        re = (MinuteClose - MinuteClose.shift())/MinuteClose.shift() #增量
        rank = re.rank(pct=True) 
        flag = pd.DataFrame((rank.values>0.9), index = re.index, columns=re.columns)
        coef = Util.array_coef(MinuteClose[flag], MinuteVolume[flag])

        return coef

       
    def  reform(self, temp_result):
        A = (temp_result- temp_result.rolling(20).mean())/temp_result.rolling(20).std()
        return -A
        
        
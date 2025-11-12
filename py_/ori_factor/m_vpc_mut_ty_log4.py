# # -*- coding: utf-8 -*-
# import numpy as np
# import sys
# sys.path.insert(0, '/data/group/800020/AlphaFramework/AnalysisTool/')
# from HFactor import *
# import statsmodels.api as sm 


# class AbnormalVolumePVCorr_13h(HFactor):

#     def definition(self, MinuteVolume,MinuteClose):
#         factor = self.minute_help(self.minute, 'AbnormalVolumePVCorr_13hHelp',MinuteVolume, MinuteClose)
#         return -factor
    
#     def minute(self ,MinuteVolume,MinuteClose ):
#         date_list = sorted(np.unique(MinuteClose.index.strftime('%Y-%m-%d')))
#         date = date_list[-1]
#         if date=='2016-01-07':
#             return pd.Series(np.nan, index = MinuteClose.columns)
        
#         MinuteVolume = MinuteVolume.iloc[-240:,:] 
#         MinuteClose = MinuteClose.iloc[-240:,:]

#         rank = MinuteVolume.rank(pct=True)

#         return MinuteClose[rank>0.9].corrwith(MinuteVolume[rank>0.9])
        
        

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log4(BaseFactor):  # 派生一个因子类
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

        # date_list = sorted(np.unique(MinuteClose.index.strftime('%Y-%m-%d')))
        # date = date_list[-1]
        # if date=='2016-01-07':
        #     return pd.Series(np.nan, index = MinuteClose.columns)
        MinuteVolume = MinuteVolume.iloc[-240:,:] 
        MinuteClose = MinuteClose.iloc[-240:,:]
        # re = MinuteClose.pct_change()
        rank = MinuteVolume.rank(pct=True)
        flag = pd.DataFrame((rank.values>0.9), index = rank.index, columns=rank.columns)
        coef = Util.array_coef(MinuteClose[flag], MinuteVolume[flag])

        return -coef
from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log26(BaseFactor):  # 派生一个因子类
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

        # r = (MinuteTurnover / MinuteVolume).pct_change()
        t1=time.time()
        r = (MinuteTurnover / MinuteVolume)
        r = (r-r.shift()) / r.shift()
        r = r.sub(r.mean(axis=1), axis=0).rolling(10).mean()  # 滚动10分钟平均超额收益率
        # amt = MinuteTurnover.div(MinuteTurnover.sum(axis=1), axis=0).pct_change()
        amt = MinuteTurnover.div(MinuteTurnover.sum(axis=1), axis=0)
        amt = (amt-amt.shift()) / amt.shift()
        roll = amt.rolling(10)
        amt_sharpe = roll.mean() / roll.std()  # 滚动10分钟成交额占比增长率Sharpe
        # print('cost', time.time()-t1)
        flag = pd.DataFrame(r.values == np.tile(r.min().values,(r.shape[0],1)), index=r.index, columns=r.columns)
        alpha = amt_sharpe[flag].mean()
        return alpha  # 超额收益最低时成交额占比增长率的Sharpe
from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log27(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.amt_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 1
    minute_lag=1
    # fix_times = ["1300"]
    # reform_window = 20

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteTurnover = database.depend_data['FactorData.Basic_factor.amt_minute']

        MinuteTurnover = MinuteTurnover.iloc[-240:].resample('30min').sum().dropna(how='all', axis=0) # 取距离当前时刻最近的240个分钟成交额数据,resample成30分钟成交额
        mt = MinuteTurnover
        amount_mkt = MinuteTurnover.sum(axis=1)
        amount_mkt = amount_mkt / amount_mkt.sum()  # 市场成交额分布
        amount = mt / pd.DataFrame(np.tile(mt.sum().values,(mt.shape[0],1)),index=mt.index,columns=mt.columns)  # 成交额分布
        e = amount.sub(amount_mkt, axis=0)
        a = np.arange(1, 1+len(e))
        a = a / a.sum()  # 线性加权
        m = e.abs().mul(a, axis=0).sum()
        return -m
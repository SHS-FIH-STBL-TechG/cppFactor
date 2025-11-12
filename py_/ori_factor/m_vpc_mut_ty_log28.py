from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log28(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.amt_minute',]    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=0
    # fix_times = ["1300"]
    # reform_window = 20

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteTurnover = database.depend_data['FactorData.Basic_factor.amt_minute']

        MinuteTurnover = MinuteTurnover.resample('20min').sum().dropna(how='all', axis=0)  # resample成20分钟成交额
        growth = MinuteTurnover.div(MinuteTurnover.sum(axis=1), axis=0)
        # .pct_change()  # 成交额占比同时刻市场总成交额比例的变化率
        # growth = (growth-growth.shift())/growth.shift()
        growth = growth.diff()/growth.shift()
        a = np.e ** (np.arange(len(growth)-1, -1, -1) * np.log(0.5) / len(growth) * 2)
        a = a / a.sum()  # 指数加权权重
        m = growth.mul(a, axis=0).sum()  # 指数加权均值
        d = growth - self.S2D(m, growth)
        d *= d
        s = d.mul(a, axis=0).sum()  # 指数加权方差
        return m / s  # 取Sharpe高的股票

    def S2D(self, S, D):
        return pd.DataFrame(np.tile(S.values,(D.shape[0],1)),index=D.index,columns=D.columns)
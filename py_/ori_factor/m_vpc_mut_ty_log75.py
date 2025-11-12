from xfactor.BaseFactor import BaseFactor
import numpy as np
from xfactor.FixUtil import minute_data_transform,min_forward_adj

# ["FactorDailyStableRet",{'n': 20, 'Data_Base': ['play_day_minute_close'], 'play_day_lag': 20,'play_min_lag': None,
#     'generator_lag': 1,'type': 1500}, "F_D_StableRet.h5"]

class m_vpc_mut_ty_log75(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    #fix_times = ["1000", "1030",'1100','1300','1330','1400','1430']
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ['FactorData.Basic_factor.amt_minute']
    #依赖的个人因子库的因子，默认为空，可不设置
    depend_factors = []
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 1
    # 播放后得到的结果，可按照该长度进行rolling等计算，具体rolling方法需要在reform方法中定义。 默认为1，可不设置。
    reform_window = 3

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        minute_data_transform(database.depend_data,['drop','drop'])
        period = '5min'
        amt = database.depend_data["FactorData.Basic_factor.amt_minute"].iloc[-237*self.lag:,:].copy()
        amt_min = amt.resample(period).sum()
        amt_min[amt_min.values == 0] = np.nan
        amt_min = amt_min.dropna(how='all')
        ans = -1 * np.sqrt(amt_min).skew()
        return ans

    # 针对播放后的结果，进行相关的rolling等操作。所用的前序数据长度应为reform_window。默认不修改temp_result， 可不重写。
    def reform(self, temp_result):
        return temp_result.rolling(self.reform_window,min_periods=int(self.reform_window/2)).mean()

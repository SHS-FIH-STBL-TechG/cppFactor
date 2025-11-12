from xfactor.BaseFactor import BaseFactor
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform,min_forward_adj
from xfactor.Util import data_filter


class m_vpc_mut_ty_log99(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    fix_times = ["1000",'1030','1100','1300','1330','1400','1430']
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ['FactorData.Basic_factor.high_minute',
                   'FactorData.Basic_factor.close_minute',
                   'FactorData.Basic_factor.limit_status_minute']
    #依赖的个人因子库的因子，默认为空，可不设置
    depend_factors = []
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 1
    # 播放后得到的结果，可按照该长度进行rolling等计算，具体rolling方法需要在reform方法中定义。 默认为1，可不设置。
    reform_window = 3

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        period = '30min'
        minute_data_transform(database.depend_data,['drop1','drop4'])
        limit_status = database.depend_data["FactorData.Basic_factor.limit_status_minute"].copy()
        close = database.depend_data["FactorData.Basic_factor.close_minute"].copy()
        high = database.depend_data["FactorData.Basic_factor.high_minute"].copy()
        close =  data_filter(min_forward_adj(close),limit_status,method='minute')
        high =  data_filter(min_forward_adj(high),limit_status,method='minute')
        ret = np.log(close/close.shift(1))
        ret.iloc[237,:] = np.nan
        ret = ret.iloc[-237*self.lag:,:].copy()
        filter_index = close.iloc[-237*self.lag:,:].resample(period,how='last').dropna(axis=0,how='all').index.to_list()
        ret_min = ret.resample(period).sum().reindex(filter_index)
        high_min = high.resample(period).max().reindex(filter_index)
        big = ((ret_min.values - ret_min.max().values) == 0)
        small = ((ret_min.values - ret_min.min().values) == 0)
        ans = np.log((high_min*big/big).mean()/(high_min*small/small).mean())
        # ans = ans.rank()
        return ans

    # 针对播放后的结果，进行相关的rolling等操作。所用的前序数据长度应为reform_window。默认不修改temp_result， 可不重写。
    def reform(self, temp_result):
        fac = temp_result.rolling(self.reform_window, min_periods=int(self.reform_window/2)).mean()
        fac_quantile = fac.quantile(0.25, axis=1)
        fac_trans = abs(pd.DataFrame(fac.values - fac_quantile.values.reshape((fac.shape[0], 1)), index=fac.index,
                                     columns=fac.columns))
        alpha = fac_trans
        return alpha

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform, min_forward_adj

class m_vpc_mut_ty_log9(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    #fix_times = ["1000", "1030", "1100", "1300", "1330", "1400", "1430"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute", "FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    minute_lag = 1
    lag = 0


    def calc_single(self, database):
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        #读取分钟数据
        close_minute = database.depend_data['FactorData.Basic_factor.close_minute']
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        #读取adj
        # adj_factor = database.depend_data['FactorData.Basic_factor.adjfactor']
        min_close = Util.data_filter(min_forward_adj(close_minute ), limit_status, method="minute")
        close = min_close.resample("5T", label="right").last().dropna(axis=0, how='all').iloc[
                -48 * (self.minute_lag) - 1:, :]  
        speed = ((close - close.shift(1)) / close.shift(1)).iloc[1:,:]
        ans_df = (speed - speed.shift(1)).iloc[1:,:]
        ans_df = pd.DataFrame(ans_df.values - ans_df.mean(axis=1).values.reshape(ans_df.shape[0], 1), index=ans_df.index, columns=ans_df.columns)
        ans = 100 * abs(ans_df).std(axis=0)
        return ans

    # 针对播放后的结果，进行相关的rolling等操作。所用的前序数据长度应为reform_window。默认不修改temp_result， 可不重写。
    def reform(self, temp_result):
        return temp_result

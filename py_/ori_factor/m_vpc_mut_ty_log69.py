from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform, min_forward_adj

class m_vpc_mut_ty_log69(BaseFactor):
    #  定义因子参数
    ptype = 'regrmse'

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    #fix_times = ["1000", "1030", "1100", "1300", "1330", "1400", "1430"]
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.amt_minute", "FactorData.Basic_factor.limit_status_minute",
                   "FactorData.Basic_factor.amt", "FactorData.Basic_factor.Data_limit_pctg"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    minute_lag = 1
    lag = 9
    reform_window = 5


    def calc_single(self, database):
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        #读取分钟数据
        amt_minute = database.depend_data['FactorData.Basic_factor.amt_minute']
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        #读取adj
        # adj_factor = database.depend_data['FactorData.Basic_factor.adjfactor']
        min_amt = Util.data_filter(amt_minute, limit_status, method="minute")
        amt_day = database.depend_data['FactorData.Basic_factor.amt']
        limit_pctg = database.depend_data['FactorData.Basic_factor.Data_limit_pctg'].astype("float64")
        day_amt = Util.data_filter(amt_day, limit_pctg, method="day")
        day_amt = day_amt.mean()
        ans_df = np.abs(min_amt.iloc[-237:,:].sum() - day_amt) / day_amt
        return ans_df

    # 针对播放后的结果，进行相关的rolling等操作。所用的前序数据长度应为reform_window。默认不修改temp_result， 可不重写。
    def reform(self, temp_result):
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = Util.rolling_process(factor_values, self.ptype, window=self.reform_window, min_periods=self.reform_window)
        return factor_values

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from copy import deepcopy
from xfactor.FixUtil import minute_data_transform,min_forward_adj
import datetime as dt

class m_vpc_mut_ty_log6(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.volume_minute","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 1
    reform_window=5

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation = ["drop1", "drop4"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']

        close_df= database.depend_data['FactorData.Basic_factor.close_minute']
        close_df = min_forward_adj(close_df)
        close_df = self.data_filter2(close_df, limit_status, method='minute')
        volume_df = database.depend_data['FactorData.Basic_factor.volume_minute']
        volume_df = self.data_filter2(volume_df, limit_status, method='minute')
        volume_df_lastday = volume_df.iloc[-237:,:]

        ret_df = pd.DataFrame(close_df.values/close_df.shift(1).values-1,index = close_df.index,columns=close_df.columns)
        ret_df_lastday = ret_df.iloc[-237:,:]

        # logvolume_lastday = np.log(volume_df_lastday)
        volume_df_std= volume_df_lastday.std()
        volume_df_avg = volume_df_lastday.mean()
        minvolume_uplimit = volume_df_avg+volume_df_std
        # minvolume_downlimit = volume_df_avg-volume_df_std
        isbeyonduplimit = pd.DataFrame(volume_df_lastday.values>minvolume_uplimit.values,index = volume_df_lastday.index,columns = volume_df_lastday.columns)
        isbeyonduplimit = isbeyonduplimit[isbeyonduplimit]
        retdfbeyonduplimit = pd.DataFrame(isbeyonduplimit.values*ret_df_lastday.values,index = isbeyonduplimit.index,columns=isbeyonduplimit.columns)

        alphastat=retdfbeyonduplimit.std()
        return alphastat

    def reform(self, temp_result):
        factor_values = temp_result
        factor_values_std = factor_values.rolling(window=self.reform_window, min_periods=1).std()
        factor_values_mean = factor_values.rolling(window=self.reform_window, min_periods=1).mean()
        factor_values_ms = factor_values_mean/factor_values_std
        return factor_values_ms


    @staticmethod
    def data_filter2(data_df, filter_df, method='day'):
        """
        将停牌/涨跌停过滤掉
        """
        ans_df = data_df.copy()
        if method == 'day':
            threshold = 0.5
            ans_df[filter_df.reindex(index=ans_df.index, columns=ans_df.columns).values > threshold] = np.nan
            ans_df[np.isnan(filter_df)] = np.nan
        elif method == 'minute':
            ans_df[pd.DataFrame((abs(filter_df).values > np.exp(-10)),index = ans_df.index,columns = ans_df.columns)|np.isnan(filter_df)] = np.nan
        else:
            raise Exception("method only suport for day or minute!")
        return ans_df
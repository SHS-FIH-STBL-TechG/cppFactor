from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.Util import *
import numpy as np
import pandas as pd
from xfactor.FixUtil import minute_data_transform

def min_forward_adj(df, date=None, precision_num = 4):
    #### 给分钟级别的因子进行复权 #########
    #### 如果传入dataframe,则不需要传入时间######
    #### 如果传入Series,则需要传入时间######
    #### precision_num为复权因子的精度位数，当为None时则不作精度处理######
    import datetime as dt
    import copy
    df = copy.deepcopy(df)
    assert isinstance(precision_num,int) or precision_num==None,'wrong precision type'
    if date == None:
        stock_list = df.columns.tolist()
        startdate = df.index[0]
        enddate = df.index[-1]
        start_date = startdate.year * 10000 + startdate.month * 100 + startdate.day
        end_date = enddate.year * 10000 + enddate.month * 100 + enddate.day

        df_adjfactor = pd.read_hdf('/data/group/800080/Apollo/AlphaDataBase/Data_adjfactor.h5', '/factor')
        if precision_num!=None:
            df_adjfactor = df_adjfactor.round(precision_num)

        df_adjfactor = df_adjfactor.loc[start_date:end_date]
        df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"), df_adjfactor.index))
        df_adjfactor = df_adjfactor.reindex(columns=stock_list)

        # 将分钟行情的df设为双重索引，其第1重索引为日期
        df['date'] = df.index.date
        df.index.name = 'datetime'
        df = df.reset_index()
        df = df.set_index(['date', 'datetime'])
        # 将分钟行情的df与复权因子按日期相乘
        df = df.multiply(df_adjfactor, level=0)
        # 去除分钟行情的df的双重索引中日期的索引，仅保留datetime这一索引
        df = df.reset_index()
        df = df.set_index('datetime')
        df = df.reindex(columns=stock_list)
    else:
        stock_list = df.index.tolist()
        start_date = date.year * 10000 + date.month * 100 + date.day

        # 暂时用666889中的数据代替get_factor_value
        df_adjfactor = pd.read_hdf('/data/group/800080/Apollo/AlphaDataBase/Data_adjfactor.h5', '/factor')
        if precision_num!=None:
            df_adjfactor = df_adjfactor.round(precision_num)
        df_adjfactor = df_adjfactor.loc[start_date].to_frame().T
        df_adjfactor.index = list(map(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"), df_adjfactor.index))
        df_adjfactor = df_adjfactor.reindex(columns=stock_list)

        df_adjfactor = df_adjfactor.iloc[-1, :]
        df = df * df_adjfactor
    return df
    
class m_vpc_mut_ty_log91(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute",
    "FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 0
    minute_lag = 5
    generator_lag=10
    reform_window = 10
    ptype='regrmse'

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        minute_data_transform(database.depend_data, operation = ["drop1", "drop4"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        minute_close = data_filter(database.depend_data['FactorData.Basic_factor.close_minute'],limit_status,method='minute')

        minute_close = min_forward_adj(minute_close)
        minute30_open = minute_close.resample("5T", label="right").first().dropna(axis=0, how='all').iloc[
                       -48 * self.minute_lag - 1:, :]
        minute30_close = minute_close.resample("5T", label="right").last().dropna(axis=0, how='all').iloc[
                       -48 * self.minute_lag -1:, :]
        avg = (minute30_open + minute30_close) / 2
        delta = (avg - avg.shift(1)).iloc[1:,:]
        df_factor = delta.std(axis=0) / abs(delta).sum(axis=0)
        return df_factor


    def reform(self, temp_result):        
        factor_values = temp_result  # 传入这里的函数每天都会调用一次播放数据计算中间量
        factor_values = Util.rolling_process(factor_values, self.ptype, window=self.reform_window, min_periods=self.reform_window)
        return factor_values

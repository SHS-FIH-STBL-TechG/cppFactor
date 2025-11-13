import numpy as np
import pandas as pd
import time


# 实现Util.array_coef函数：计算两个DataFrame每列的协方差
def array_coef(x, y):
    x_values = np.array(x, dtype=float)
    y_values = np.array(y, dtype=float)
    x_values[np.isinf(x_values)] = np.nan
    y_values[np.isinf(y_values)] = np.nan
    nan_index = np.isnan(x_values) | np.isnan(y_values)
    x_values[nan_index] = np.nan
    y_values[nan_index] = np.nan
    delta_x = x_values - np.nanmean(x_values, axis=0)
    delta_y = y_values - np.nanmean(y_values, axis=0)
    multi = np.nanmean(delta_x * delta_y, axis=0) / (np.nanstd(delta_x, axis=0) * np.nanstd(delta_y, axis=0))
    multi[np.isinf(multi)] = np.nan
    return pd.Series(multi, index=x.columns)


class Util:
    """工具类，提供array_coef方法"""
    @staticmethod
    def array_coef(df_x, df_y):
        return array_coef(df_x, df_y)


class m_vpc_mut_ty_log2:  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.volume_adj_minute', 'FactorData.Basic_factor.close_adj_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 0
    minute_lag=1
    # fix_times = ["1300"]
    reform_window = 20
    updateNum = -1
    
    def calc_single(self, database):
        # 兼容amt_minute和volume_adj_minute两种字段名
        if 'FactorData.Basic_factor.volume_adj_minute' in database.depend_data:
            MinuteVolume = database.depend_data['FactorData.Basic_factor.volume_adj_minute']
        elif 'FactorData.Basic_factor.amt_minute' in database.depend_data:
            MinuteVolume = database.depend_data['FactorData.Basic_factor.amt_minute']
        else:
            raise KeyError("找不到volume_adj_minute或amt_minute字段")
        
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_adj_minute']

        MinuteVolume = MinuteVolume.iloc[-240:,:] 
        MinuteClose = MinuteClose.iloc[-240:,:]

        re = (MinuteClose - MinuteClose.shift())/MinuteClose.shift() #增量
        rank = re.rank(pct=True) 
        flag = pd.DataFrame((rank.values>0.9), index = re.index, columns=re.columns)
        coef = Util.array_coef(MinuteClose[flag], MinuteVolume[flag])

        # debug
        self.updateNum += 1
        if self.updateNum == 1:
            print("MinuteVolume")
            print(MinuteVolume)
            print("MinuteClose")
            print(MinuteClose)
            print("re")
            print(re)
            print("rank")
            print(rank)
            print("MinuteVolume[flag]")
            print(MinuteVolume[flag])
            print("MinuteClose[flag]")
            print(MinuteClose[flag])

        return coef

       
    def  reform(self, temp_result):
        A = (temp_result- temp_result.rolling(20).mean())/temp_result.rolling(20).std()
        return -A
        
        

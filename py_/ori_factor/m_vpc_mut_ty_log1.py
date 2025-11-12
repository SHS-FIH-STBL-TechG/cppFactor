from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log1(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.amt_minute', 'FactorData.Basic_factor.close_adj_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 2
    minute_lag=2
    fix_times = ["1300"]
    # reform_window = 5


    '''
    * 因子名：AbnormalPriceDiff_13h
    * 描述：前日到今日异常交易后45分钟价格波动的绝对值的偏度负数，异常交易为价格的变化与波动率比值大于2
    * 逻辑：因子值越大表示低价格变动越大表示价格大幅变动之后变化越小，股票优势，建议持有
    * 因子参数：分钟数据的收高低、量、换手
    * 函数修改日期：尚未修改
    * 修改人：尚未修改
    * 修改原因：尚未修改
    '''
    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteTurnover = database.depend_data['FactorData.Basic_factor.amt_minute']
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_adj_minute']

        std_period = 60
        add_signal = MinuteTurnover.diff() /  MinuteTurnover.rolling(std_period).std()
        add_signal = (add_signal.values > 2.)
        
        rollingtime = 45
        UPDO = MinuteClose.diff(rollingtime).shift(-rollingtime).abs()
        f = (UPDO * add_signal).skew()
        return -f

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
from xfactor.FixUtil import minute_data_transform
import numpy as np
import pandas as pd
import time




class m_vpc_mut_ty_log93(BaseFactor):  # 派生一个因子类
    factor_type = 'FIX'             # 声明因子类型为FIX
    depend_data = ['FactorData.Basic_factor.close_minute']    # 声明因子计算需要依赖的数据字段，必需设置
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 当lag = n时，每次播放时将提供 242 * (n+1) 根分钟线数据，默认lag=0，可不设置
    lag = 1
    minute_lag=1
    # fix_times = ["1300"]
    reform_window = 5

    
    def calc_single(self, database):

        minute_data_transform(database.depend_data, operation = ["drop", "merge"])
        MinuteClose = database.depend_data['FactorData.Basic_factor.close_minute']

        t1=time.time()
        date_list = sorted(np.unique(MinuteClose.index.strftime('%Y-%m-%d')))
        # print('cost',time.time()-t1)
        date = date_list[-1]
        pre_date = date_list[-2]

        re1 = MinuteClose.loc[pre_date]
        re1 = re1.diff()/re1.shift()
        re2 = MinuteClose.loc[date]
        re2 = re2.diff()/re2.shift()
        re = pd.concat([re1,re2])
        score = (re-self.S2D(re.mean(),re))/self.S2D(re.std(),re)
        flag = pd.DataFrame((score.values< -2)|(score.values>2), index=re.index, columns=re.columns)
        alpha = re[flag].sum()
        return alpha

        # re1 = MinuteClose.loc[pre_date].pct_change()
        # re2 = MinuteClose.loc[date].pct_change()
        # re = pd.concat([re1,re2])
        # score = (re-re.mean())/re.std()
        # return re[(score< -2)|(score>2)].sum()
        


    def  reform(self, temp_result):
        A = temp_result.rolling(5,min_periods=1).mean()/temp_result.rolling(5,min_periods=1).std()
        # A = pd.DataFrame(-1.*A.values, index=A.index, columns=A.columns,)
        return -A

    def S2D(self, S, D):
        return pd.DataFrame(np.tile(S.values,(D.shape[0],1)),index=D.index,columns=D.columns)
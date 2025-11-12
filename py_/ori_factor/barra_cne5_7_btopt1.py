from xfactor.BaseFactor import BaseFactor
import numpy as np
import pandas as pd
import copy
import time
from sklearn.preprocessing import scale


class barra_cne5_7_btopt1(BaseFactor):
    factor_type = "DAY"
    depend_data = ['FactorData.Basic_factor.s_val_pb_new', 'FactorData.Basic_factor.is_valid_test', 'FactorData.Basic_factor.sw_indcode1']
    financial_lag = 270
    lag = 0
    reform_window = 1

    def calc_single(self, database):
        btop = 1 / database.depend_data['FactorData.Basic_factor.s_val_pb_new'].replace(0,np.nan)
        v = database.depend_data['FactorData.Basic_factor.is_valid_test']
        sw_indcode1 = database.depend_data['FactorData.Basic_factor.sw_indcode1'].iloc[-1:].T.reset_index()
        sw_indcode1.columns = ['index', 'swl1']

        ans = btop.iloc[-1]

        def demean(ans):
            ans1 = ans.reset_index()
            ans1.columns = ['index', 'ans']
            ans_ind = pd.merge(sw_indcode1, ans1, on=['index'], how='left')
            ind = ans_ind.groupby('swl1').mean().reset_index()
            ind1 = (ans_ind.groupby('swl1').quantile(0.683) - ans_ind.groupby('swl1').quantile(0.317)).reset_index() # 1倍标准差的分布范围
            ans_ind1 = pd.merge(ans_ind, ind, on=['swl1'], how='left')
            ans_ind1 = pd.merge(ans_ind1, ind1, on=['swl1'], how='left')
            ans_ind1['ans_zscore'] = (ans_ind1[ans_ind1.columns[-3]] - ans_ind1[ans_ind1.columns[-2]])
            ans = ans_ind1.set_index('index')['ans_zscore']
            return ans
        return demean(ans) * v.iloc[-1]

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A
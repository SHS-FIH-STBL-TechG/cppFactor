from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_7_btop1(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.s_val_pb_new", "FactorData.Basic_factor.a_mkt_cap",'FactorData.SUNTIME_con_forecast_roll_stk',
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.pcf_ncf_ttm"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 500
    reform_window = 1

    def calc_single(self, database):
        pb = database.depend_data["FactorData.Basic_factor.s_val_pb_new"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        def winsor(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            # beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1
        def winsor1(beta):
            # beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        v = ((pb.count() > 400) + 0).replace(0, np.nan)
        bp = 1 / winsor(pb.iloc[-1] * v)
        ans = winsor(bp)
        # ans = zscore(winsor(mom)) * valid.mean()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

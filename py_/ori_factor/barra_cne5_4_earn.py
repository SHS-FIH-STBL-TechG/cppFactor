from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_4_earn(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pe_ttm", "FactorData.Basic_factor.a_mkt_cap",'FactorData.SUNTIME_con_forecast_roll_stk',
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.pcf_ncf_ttm"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 0
    reform_window = 1

    def calc_single(self, database):
        def winsor(beta):
            beta1 = np.where(beta < beta.quantile(0.995),np.where(beta > beta.quantile(0.005), beta, beta.quantile(0.005)),beta.quantile(0.995))
            beta1 = pd.Series(beta1, index = beta.index)  * valid.mean()
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        try:
            pe = database.depend_data["FactorData.Basic_factor.pe_ttm"]
            pcf = database.depend_data["FactorData.Basic_factor.pcf_ncf_ttm"]
            con = database.depend_data["FactorData.SUNTIME_con_forecast_roll_stk"]['con_np_roll'].unstack()
            float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
            valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

            epfwd = con.iloc[-1] / float.iloc[-1].replace(0,np.nan)
            cetop = 1 / pcf.iloc[-1].replace(0,np.nan)
            etop = 1 / pe.iloc[-1].replace(0,np.nan)

            earn = 0.68 * epfwd + 0.21 * cetop + 0.11 * etop
            ans = zscore(earn) * valid.iloc[-1]
        except:
            ans = np.nan
        # ans = zscore(winsor(mom)) * valid.mean()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

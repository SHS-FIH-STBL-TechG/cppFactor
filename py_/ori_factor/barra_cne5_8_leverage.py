from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_8_leverage(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.WIND_AShareBalanceSheet", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 180 # financial_lag需保证至少获取到一个季度的财度数据
    lag = 0
    reform_window = 1

    def calc_single(self, database):
        bs = database.depend_data['FactorData.WIND_AShareBalanceSheet']
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"] * 10000
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        def fill(fdm): # 最新财报未披露时，用上一期的填充
            fdm.iloc[-1] = np.where(np.isnan(fdm.iloc[-1]), fdm.iloc[-2], fdm.iloc[-1])
            return fdm

        ld = bs[bs.STATEMENT_TYPE == 408001000]['TOT_NON_CUR_LIAB'].unstack() # 非流动负债
        ld = fill(ld).iloc[-1]
        me = float.iloc[-1]
        mlev = (me + ld) / me

        ta = bs[bs.STATEMENT_TYPE == 408001000]['TOT_ASSETS'].unstack()
        tl = bs[bs.STATEMENT_TYPE == 408001000]['TOT_LIAB'].unstack()
        dtoa = fill(tl / ta).iloc[-1]

        be = bs[bs.STATEMENT_TYPE == 408001000]['TOT_SHRHLDR_EQY_INCL_MIN_INT'].unstack()
        blev = (fill(be).iloc[-1] + ld) / fill(be).iloc[-1]


        def winsor(beta):
            # beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            beta1 = beta.mask((beta < beta.mean() - 1 * beta.std()) | (beta > beta.mean() + 1 * beta.std()))
            return beta1

        # 标准化
        def zscore(beta):
            beta = beta * valid.mean()
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans

        lev = 0.38 * mlev + 0.35 * dtoa + 0.3 * blev
        # ans = zscore(lev)  # 极端值会影响pearson相关系数
        ans = zscore(winsor(lev))
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

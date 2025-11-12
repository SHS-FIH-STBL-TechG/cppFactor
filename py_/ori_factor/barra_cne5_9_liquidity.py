from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_9_liquidity(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.free_turn", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.turn"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 256
    reform_window = 1

    def calc_single(self, database):
        turn = database.depend_data["FactorData.Basic_factor.turn"]
        # turn = database.depend_data["FactorData.Basic_factor.free_turn"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        stom = np.log(turn.iloc[-21:].sum().replace(0,np.nan))
        stoq = np.log(turn.iloc[-63:].sum().replace(0,np.nan) / 3)
        stoa = np.log(turn.sum().replace(0,np.nan) / 12)

        def winsor(beta):
            beta1 = np.where(beta < beta.quantile(0.995),np.where(beta > beta.quantile(0.005), beta, beta.quantile(0.005)),beta.quantile(0.995))
            beta1 = pd.Series(beta1, index = beta.index)  * valid.iloc[-1]
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans

        liq = 0.35 * stom + 0.35 * stoq + 0.3 * stoa
        ans = zscore(liq) * valid.mean()
        # ans = zscore(winsor(mom)) * valid.mean()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

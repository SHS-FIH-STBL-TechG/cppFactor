from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np
import statsmodels.api as sm  # 导入OLS模型库


class barra_cne5_9_liquidity2(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.volume", "FactorData.Basic_factor.mkt_cap_ard",
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.float_a_shares"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 251
    reform_window = 1

    def calc_single(self, database):
        shares = database.depend_data["FactorData.Basic_factor.float_a_shares"]
        volume = database.depend_data["FactorData.Basic_factor.volume"]
        float = database.depend_data["FactorData.Basic_factor.mkt_cap_ard"]
        # valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        turn = volume / shares / 100

        def winsor(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            # beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1
        turn = turn * ((turn.count() > 250) + 0).replace(0, np.nan)
        turn = turn.clip(turn.mean() - 3 * turn.std(), turn.mean() + 3 * turn.std(), axis=1)  # 个股turn做3std winsor

        turn = turn.T
        turn = turn.clip(turn.mean() - 3 * turn.std(), turn.mean() + 3 * turn.std(), axis=1)  # 个股turn做3std winsor
        turn = turn.T

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        stom = np.log(turn.iloc[-21:].sum().replace(0, np.nan))
        stoq = np.log(turn.iloc[-63:].sum().replace(0, np.nan) / 3)
        stoa = np.log(turn.sum().replace(0, np.nan) / 12)
        liq = (0.35 * stom + 0.35 * stoq + 0.3 * stoa).dropna()
        size = winsor(np.log(winsor(float.iloc[-1]))).dropna() #计算size，用作正交化

        y = liq.loc[liq.index & size.index]
        x = sm.add_constant(size.loc[liq.index & size.index])
        # model = sm.OLS(y, x).fit()
        weight = np.sqrt(float.iloc[-1]).loc[x.index]
        # weight = size.loc[x.index]
        model = sm.WLS(y, x, weights = weight).fit()
        ans = zscore(winsor(model.resid))
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np
import statsmodels.api as sm  # 导入OLS模型库

class barra_cne5_10_sizenl2(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 126
    reform_window = 1

    def calc_single(self, database):
        # ret = database.depend_data["FactorData.Basic_factor.pct_chg"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        # valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        cap = float.iloc[-1]
        # cap = cap * ((float.count() > 100) + 0).replace(0, np.nan)  # 剔除前一年交易天数少于200的
        # cap = cap.clip(cap.mean() - 3 * cap.std(), cap.mean() + 3 * cap.std())

        size = np.log(cap)
        def winsor(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            # beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1

        def winsor1(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans

        # size = (size * valid.iloc[-1]).dropna()
        size = size.dropna()
        size_cube = winsor(size ** 3)
        size = winsor(size)

        # size_cube = size_cube.clip(size_cube.mean() - 3 * size_cube.std(), size_cube.mean() + 3 * size_cube.std())
        y = size_cube
        x = sm.add_constant(size)
        # model = sm.OLS(y, x).fit()
        weight = np.sqrt(winsor(float.iloc[-1])).loc[x.index]
        model = sm.WLS(y, x, weights = weight).fit()
        ans = -zscore(winsor(model.resid))
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

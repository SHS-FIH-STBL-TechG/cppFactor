from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_2_momentum3(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_badj", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 524
    reform_window = 1

    def calc_single(self, database):
        close = database.depend_data["FactorData.Basic_factor.close_badj"].iloc[:-20]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        # valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        ret = np.log(close / close.shift(1))

        ret = ret.clip(ret.mean() - 3 * ret.std(), ret.mean() + 3 * ret.std(), axis=1)  # 先对行做winsor

        ret = ret.T
        ret = ret.clip(ret.mean() - 3 * ret.std(), ret.mean() + 3 * ret.std(), axis=1)  # 再对列做
        ret = ret.T


        # ret = ret.clip(-0.25,0.25)  # 收益去极值
        ret = ret * ((close.count() > 100) + 0).replace(0, np.nan) #加上后相关性降低
        mom = np.log((ret / 100 + 1)).ewm(halflife=126).sum().iloc[-1]  # 转化为对数收益率：可加性
        def winsor(beta):
            # beta1 = beta.mask((beta < beta.mean() - 3 * beta.std()) | (beta > beta.mean() + 3 * beta.std())) # trimming
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())  # winsor
            beta1 = pd.Series(beta1, index = beta.index)
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans

        ans = winsor(mom)
        # ans = zscore(ans)
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

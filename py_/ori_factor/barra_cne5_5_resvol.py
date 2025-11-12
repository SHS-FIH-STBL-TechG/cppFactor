from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_5_resvol(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 252
    reform_window = 1

    def calc_single(self, database):
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']
        def zscore(beta):
            beta = beta * valid.iloc[-1]
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        # 1. 对原始数据进行winsor处理
        rett = ret.T
        rett = rett.where(rett > -20, -20)  # 涨跌幅超过上限的winsor一下
        rett = rett.where(rett < 20, 20) * (ret.T / ret.T)
        ret = np.log(rett.T / 100 + 1)  #使用对数收益
        # ret = np.log(rett.T / 100 + 1)  #使用对数收益

        dastd = ret.ewm(halflife = 42).std().iloc[-1]
        ret_cum = ret.cumsum()  # 对数收益可加
        cmra = ret_cum.max() - ret_cum.min()

        rm = (ret * float).sum(axis=1) / float.sum(axis=1)  # 用流通市值加权计算指数收益率
        beta = ret.corrwith(rm) * ret.std() / rm.std()  # 用等价公式计算beta
        retp = pd.DataFrame(np.dot(pd.DataFrame(beta), pd.DataFrame(rm).T).T,columns = [ret.columns & beta.index],index = ret.index)
        hsigma = np.subtract(ret[ret.columns & beta.index], retp).std()

        # 标准化
        resvol = 0.74 * zscore(dastd) + 0.16 * zscore(cmra) + 0.1 * zscore(hsigma)
        ans = resvol
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

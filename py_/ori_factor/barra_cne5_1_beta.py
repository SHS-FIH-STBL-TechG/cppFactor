from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_1_beta(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 149
    reform_window = 1

    def calc_single(self, database):
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        # 1. 对原始数据进行winsor处理
        rett = ret.T
        rett = rett.where(rett > -20, -20)  # 涨跌幅超过上限的winsor一下
        rett = rett.where(rett < 20, 20) * (ret.T / ret.T)
        ret = rett.T

        rm = (ret * float).sum(axis=1) / float.sum(axis=1)  # 用流通市值加权计算指数收益率
        beta = ret.corrwith(rm) * ret.std() / rm.std()  # 用等价公式计算beta
        beta = beta.clip(beta.quantile(0.005), beta.quantile(0.995))
        beta = pd.Series(beta, index = ret.columns)  * valid.iloc[-1]

        # 标准化
        beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
        ans = (beta - beta_mean) / beta.std()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_1_beta2(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.pct_chg-000905.SH"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 252
    reform_window = 1

    def calc_single(self, database):
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"] / 100
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']
        rm = database.depend_data["FactorData.Basic_factor.pct_chg-000905.SH"].iloc[:,0] / 100

        # 1. 对原始数据进行winsor处理
        rett = (ret * valid.iloc[-1]).T
        rett = rett.where(rett > -20, -20)  # 涨跌幅超过上限的winsor一下
        rett = rett.where(rett < 20, 20) * (ret.T / ret.T)
        ret = rett.T

        # 衰减加权和除法的先后问题？
        cov = (ret - ret.mean()).ewm(halflife=63).cov(rm - rm.mean())
        vari = ret.ewm(halflife=63).var()
        beta = cov.iloc[-1] / vari.iloc[-1]
        beta = beta.clip(beta.quantile(0.005), beta.quantile(0.995))   # winsor

        # 标准化
        beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
        ans = (beta - beta_mean) / beta.std()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

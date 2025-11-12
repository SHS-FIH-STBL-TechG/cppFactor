from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_11_betas(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 251
    reform_window = 1

    def calc_single(self, database):
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"] / 100
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        # 1. 对原始数据进行winsor处理
        ret = ret.clip(-0.21,0.21)

        smb = ret.T.where((float.T <= float.T.median()) & (float.T > float.T.quantile(0.1)), np.nan).mean() - ret.T.where((float.T >= float.T.median()) & (float.T < float.T.quantile(0.9)), np.nan).mean() #剔除两端10%的样本
        corr = (ret - ret.mean()).ewm(halflife=63).corr(smb - smb.mean())
        beta = corr.iloc[-1] * ret.ewm(halflife=63).std().iloc[-1] / smb.ewm(halflife=63).std().iloc[-1]

        beta = beta * ((ret.count() > 250) + 0).replace(0,  np.nan)   # 样本数太少的剔掉，临界参数需要尝试
        beta = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())   # winsor

        # 标准化
        beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
        ans = (beta - beta_mean) / beta.std()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

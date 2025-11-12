from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm  # 导入OLS模型库

class barra_cne5_5_resvol1(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.pct_chg", "FactorData.Basic_factor.a_mkt_cap",
                   "FactorData.Basic_factor.is_valid_test","FactorData.Basic_factor.pct_chg-000985.CSI"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 251
    reform_window = 1

    def calc_single(self, database):
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"] / 100
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        # valid = database.depend_data['FactorData.Basic_factor.is_valid_test']
        rm = database.depend_data["FactorData.Basic_factor.pct_chg-000985.CSI"].iloc[:, 0] / 100

        def zscore(beta):
            beta = beta
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        def winsor(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            # beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1

        # 对原始数据进行winsor处理
        ret = ret.clip(-0.21,0.21)
        lgret = np.log(ret + 1)  #对数收益

        # 1. dastd
        dastd = ret.ewm(halflife = 42).std().iloc[-1]

        # 2. cmra
        retm = lgret.reset_index(drop=True)
        retm.index = ((retm.index) / 21).astype(int) # index变成月频
        retm1 = retm.groupby(retm.index).sum()   # 计算月收益
        ret_cum = retm1.cumsum()  # 对数收益可加
        cmra = ret_cum.max() - ret_cum.min()

        # 3. hsigma
        corr = (ret - ret.mean()).ewm(halflife=63).corr(rm - rm.mean())
        beta = corr.iloc[-1] * ret.ewm(halflife=63).std().iloc[-1] / rm.ewm(halflife=63).std().iloc[-1]
        beta = beta * ((ret.count() > 200) + 0).replace(0,  np.nan)   # 样本数太少的剔掉，临界参数需要尝试
        beta = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())   # winsor

        retp = pd.DataFrame(np.dot(pd.DataFrame(beta), pd.DataFrame(rm).T).T,columns = [ret.columns & beta.index],index = ret.index)
        hsigma = np.subtract(ret[ret.columns & beta.index], retp).std()

        # 标准化
        resvol = 0.74 * zscore(dastd) + 0.16 * zscore(cmra) + 0.1 * zscore(hsigma)
        size = winsor(np.log(winsor(float.iloc[-1])))  # 计算市值因子

        # resvol对size和beta取残差
        df = pd.merge(pd.merge(resvol.reset_index(),size.reset_index(),on=['index']), beta.reset_index(),on=['index']).dropna(how='any').set_index(['index'])

        y = df.iloc[:,0]
        x = sm.add_constant(df.iloc[:,-2:])
        # model = sm.OLS(y, x).fit()
        weight = np.sqrt(float.iloc[-1]).loc[x.index]
        model = sm.WLS(y, x, weights = weight).fit()

        ans = winsor(model.resid)
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

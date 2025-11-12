from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np
import statsmodels.api as sm  # 导入OLS模型库

class barra_cne5_6_growth(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.WIND_AShareIncome", "FactorData.Basic_factor.a_mkt_cap",
                   'FactorData.SUNTIME_con_forecast_roll_stk', "FactorData.Basic_factor.is_valid_test"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 1500 # financial_lag需保证至少获取到一个季度的财度数据
    lag = 0
    reform_window = 1

    def calc_single(self, database):
        inc = database.depend_data['FactorData.WIND_AShareIncome']
        profit = inc[inc.STATEMENT_TYPE == 408001000]['TOT_PROFIT'].unstack()
        # profit = profit[profit.date.astype(str).str[5:7] == '12']
        revenue = inc[inc.STATEMENT_TYPE == 408001000]['OPER_REV'].unstack()
        # revenue = revenue[revenue.date.astype(str).str[5:7] == '12']
        def winsor(beta):
            beta1 = np.where(beta < beta.quantile(0.99),np.where(beta > beta.quantile(0.01), beta, beta.quantile(0.01)),beta.quantile(0.99))
            beta1 = pd.Series(beta1, index = beta.index)  * valid.iloc[-1]
            return beta1

        # 标准化
        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans
        # def t_ols(df):
        #     df = df.dropna(how='any')
        #     if len(df > 5):
        #         df['i'] = 1
        #         df['i'] = df['i'].cumsum()
        #         y = df.iloc[:,2] / abs(df.iloc[:,2]).mean() # 用平均绝对净利润归一化处理？
        #         x = sm.add_constant(df[['i']])
        #         model = sm.OLS(y, x).fit()
        #         res = model.params[1]
        #     else:
        #         res = np.nan
        #     return res

        egro = (profit - profit.shift(1)).mean() / abs(profit).mean() * 4
        sgro = (revenue - revenue.shift(1)).mean() / abs(revenue).mean() * 4 #乘以4 年华一下，跟预期数据可比

        egrlf = database.depend_data["FactorData.SUNTIME_con_forecast_roll_stk"]['con_npcgrate_2y_roll'].unstack().iloc[-1] / 100
        egrsf = database.depend_data["FactorData.SUNTIME_con_forecast_roll_stk"]['con_np_yoy_roll'].unstack().iloc[-1] / 100
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        growth = 0.18 * egrlf + 0.11 * egrsf + 0.24 * egro + 0.47 * sgro
        # ans = zscore(growth) * valid.mean()
        ans = zscore(winsor(growth)) * valid.mean()
        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import pandas as pd
import numpy as np


class barra_cne5_4_earnyild3(BaseFactor):
    factor_type = "DAY"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ['FactorData.WIND_AShareIncome.[STATEMENT_TYPE, NET_PROFIT_EXCL_MIN_INT_INC]', "FactorData.Basic_factor.a_mkt_cap",'FactorData.SUNTIME_con_forecast_roll_stk',
                   "FactorData.Basic_factor.is_valid_test",'FactorData.WIND_AShareCashFlow.[STATEMENT_TYPE,STOT_CASH_INFLOWS_OPER_ACT]']
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    financial_lag = 450  # financial_lag需保证至少获取到一个季度的财度数据
    lag = 0
    reform_window = 1

    def calc_single(self, database):
        inc = database.depend_data['FactorData.WIND_AShareIncome.[STATEMENT_TYPE, NET_PROFIT_EXCL_MIN_INT_INC]']
        cf = database.depend_data['FactorData.WIND_AShareCashFlow.[STATEMENT_TYPE,STOT_CASH_INFLOWS_OPER_ACT]']

        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']
        con = database.depend_data["FactorData.SUNTIME_con_forecast_roll_stk"]['con_np_roll'].unstack()

        def renew(panel):  # 根据财务数据面板，返回最新季度的数据
            panel.iloc[-1] = np.where(np.isnan(panel.iloc[-1]), panel.iloc[-2], panel.iloc[-1])  # 最新季报未出时，数据平移
            return panel.iloc[-1]

        earn = inc[inc.STATEMENT_TYPE==408002000]['NET_PROFIT_EXCL_MIN_INT_INC'].unstack()
        earn_ttm = earn.rolling(4).sum()
        pe_ttm = float.iloc[-1][float.columns & earn_ttm.columns] / renew(earn_ttm)[float.columns & earn_ttm.columns] * 10000

        ocf_ttm = (cf[cf.STATEMENT_TYPE==408002000]['STOT_CASH_INFLOWS_OPER_ACT'].unstack()).rolling(4).sum()
        pcf_ttm = float.iloc[-1][float.columns & ocf_ttm.columns] / renew(ocf_ttm)[float.columns & ocf_ttm.columns] * 10000

        def winsor(beta):
            beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            # beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1
        def winsor1(beta):
            # beta1 = beta.clip(beta.mean() - 3 * beta.std(), beta.mean() + 3 * beta.std())
            beta1 = beta.mask((beta < beta.mean() - 2 * beta.std()) | (beta > beta.mean() + 2 * beta.std()))
            return beta1

        def zscore(beta):
            beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
            ans = (beta - beta_mean) / beta.std()
            return ans

        epfwd = winsor(con.iloc[-1] / float.iloc[-1].replace(0, np.nan))
        cetop = winsor(1 / pcf_ttm.replace(0, np.nan))
        etop = winsor(1 / pe_ttm.replace(0, np.nan))

        earn = 0.68 * epfwd + 0.21 * cetop + 0.11 * etop
        ans = winsor1(earn)

        return ans

    def reform(self, temp_result):
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A

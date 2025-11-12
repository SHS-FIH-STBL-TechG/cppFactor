from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from copy import deepcopy
from xfactor.FixUtil import minute_data_transform,min_forward_adj
import datetime as dt

class m_vpc_mut_ty_log829(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.citics_indcode2","FactorData.Basic_factor.mkt_cap_ard","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag = 1
    reform_window=5

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation = ["drop1", "drop4"]);limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']

        close_df = database.depend_data['FactorData.Basic_factor.close_minute']
        close_df = min_forward_adj(close_df)
        close_df = self.data_filter2(close_df, limit_status, method='minute')
        ret_df = pd.DataFrame(close_df.values/close_df.shift(1).values-1,index = close_df.index,columns = close_df.columns).fillna(0)
        ret_df_lastday = ret_df.iloc[-237:,:]
        ret_df_lastday_clipped = self.outlier_filter(ret_df_lastday,method='3Std')

        citic_ind = database.depend_data['FactorData.Basic_factor.citics_indcode2']
        ind = self.industry_convert(citic_ind)
        ind_tile = pd.DataFrame(np.tile(ind,[ret_df_lastday_clipped.shape[0],1]),index = ret_df_lastday_clipped.index,columns=ret_df_lastday_clipped.columns)
        mktcap = database.depend_data['FactorData.Basic_factor.mkt_cap_ard']
        mktcap_tile = pd.DataFrame(np.tile(mktcap,[ret_df_lastday_clipped.shape[0],1]),index = ret_df_lastday_clipped.index,columns=ret_df_lastday_clipped.columns)
        pure_ret=self.factor_neutralizer(ret_df_lastday_clipped, ind_tile,mktcap_tile, neutral_factor_set={'size', 'industry3'})
        ans =pure_ret.skew()
        return ans



    def reform(self, temp_result):
        factor_values = temp_result
        factor_values = factor_values.rolling(window=self.reform_window, min_periods=1).std()
        return factor_values

    @staticmethod
    def outlier_filter(value_df, method="MAD", parameter=3.14826, lower_clip=True, upper_clip=True):
        if method == "3Std":
            factor_mean = value_df.mean(axis=1)
            factor_std = value_df.std(axis=1)
            upper_limit = factor_mean + 3 * factor_std
            lower_limit = factor_mean - 3 * factor_std
        else:
            # 如有全为nan的行，则drop之
            factor_max = value_df.max(axis=1)
            factor_max = factor_max.dropna()
            value_df = value_df.reindex(factor_max.index)
            factor_median = value_df.median(axis=1)
            factor_deviation_from_median = value_df.sub(factor_median, axis=0)
            factor_mad = factor_deviation_from_median.abs().median(axis=1)
            lower_limit = factor_median - parameter * factor_mad
            upper_limit = factor_median + parameter * factor_mad
        lower_limit = lower_limit.fillna(method='ffill')
        upper_limit = upper_limit.fillna(method='ffill')
        if lower_clip:
            value_df = value_df.clip_lower(lower_limit, axis='index')
        if upper_clip:
            value_df = value_df.clip_upper(upper_limit, axis='index')
        return value_df

    @classmethod
    def factor_neutralizer(cls,factor_df, industry_df,mkt_cap_ard_df, neutral_factor_set={'size', 'industry3'}):
        """
        注释 - neutral_regressor_backward_shift: 中性化因子（行业、市值）是否要取前一天的
        """
        if neutral_factor_set == {'size', 'industry3'} or neutral_factor_set == {'size'}:
            pass
        else:
            neutralized_factor_df = factor_df
            return neutralized_factor_df
        stock_list = list(factor_df.columns)
        factor_date_list = list(factor_df.index)
        if neutral_factor_set == {'size', 'industry3'}:
            t1 = dt.datetime.now()
            factor_start_line = list(mkt_cap_ard_df.index).index(factor_df.index[0])
            factor_end_line = list(mkt_cap_ard_df.index).index(factor_df.index[-1])
            # 要生成一个从factor_start_line 到 factor_end_line的自然数数列，前后双闭；range是前闭后开，因此区间后半部分要+1
            repeat_lines_list = list(range(factor_start_line, factor_end_line + 1))
            factor_array = factor_df.values
            industry_df2 = industry_df.fillna(0)  # 将行业的缺失值替换为0，以方便后续用np创造one_hot矩阵
            industry_array = industry_df2.values
            mkt_cap_ard_array = mkt_cap_ard_df.values
            stock_col_num = stock_list.__len__()
            residual_list = []
            residual_date_list = []
            if repeat_lines_list.__len__() > 10000:
                division_multiplier = 1000
            elif repeat_lines_list.__len__() > 5000:
                division_multiplier = 500
            else:
                division_multiplier = 100
            for j, i_line in enumerate(repeat_lines_list):

                y0 = factor_array[j]
                x1_0 = industry_array[i_line].astype(np.int)
                x1_0 = cls.make_one_hot(x1_0)  # 构造one_hot矩阵
                x1_0 = x1_0[:, 1:]  # 去掉第0列，也就是去掉原来无行业的值
                x2_0 = mkt_cap_ard_array[i_line]
                x2_0 = x2_0.reshape([stock_col_num, 1])
                x0 = np.hstack([x1_0, x2_0])
                y0_isnan = np.isnan(y0)
                x0_isnan = np.isnan(np.max(x0, axis=1))
                y0_isvalid = 1 - y0_isnan
                x0_isvalid = 1 - x0_isnan
                valid_rows = x0_isvalid * y0_isvalid
                valid_stock_list = [stock_list[i] for i in range(stock_list.__len__()) if valid_rows[i] == 1]
                x0_valid = x0[valid_rows == 1]
                y0_valid = y0[valid_rows == 1]
                ind_check = np.sum(x0_valid, axis=0)
                empty_industry = []
                for i_col in range(x1_0.shape[1]):
                    if ind_check[i_col] < 1:
                        empty_industry.append(i_col)
                if empty_industry.__len__() > 0:
                    x0_valid = np.delete(x0_valid, empty_industry, 1)
                if x0_valid.__len__() > 0:
                    b = np.linalg.inv(x0_valid.T.dot(x0_valid)).dot(x0_valid.T).dot(y0_valid)
                    residual = y0_valid - x0_valid.dot(b)  # 求残差
                    residual_dict = dict(zip(valid_stock_list, residual))  # 将残差与股票代码关联起来
                    residual_list.append(residual_dict)
                    residual_date_list.append(factor_date_list[j])
            # 将残差list一次性转变为DataFrame
            neutralized_factor_df = pd.DataFrame(residual_list, index=residual_date_list)
            t2 = dt.datetime.now()
            #print("neutralizing costs", t2 - t1)
            return neutralized_factor_df
        elif neutral_factor_set == {'size'}:
            t1 = dt.datetime.now()
            factor_start_line = list(mkt_cap_ard_df.index).index(factor_df.index[0])
            factor_end_line = list(mkt_cap_ard_df.index).index(factor_df.index[-1])
            # 要生成一个从factor_start_line 到 factor_end_line的自然数数列，前后双闭；range是前闭后开，因此区间后半部分要+1
            repeat_lines_list = list(range(factor_start_line, factor_end_line + 1))
            factor_array = factor_df.values
            mkt_cap_ard_array = mkt_cap_ard_df.values
            stock_col_num = stock_list.__len__()
            residual_list = []
            residual_date_list = []
            if repeat_lines_list.__len__() > 10000:
                division_multiplier = 1000
            elif repeat_lines_list.__len__() > 5000:
                division_multiplier = 500
            else:
                division_multiplier = 100
            for j, i_line in enumerate(repeat_lines_list):

                y0 = factor_array[j]
                x0 = mkt_cap_ard_array[i_line]
                x0 = x0.reshape([stock_col_num, 1])
                y0_isnan = np.isnan(y0)
                x0_isnan = np.isnan(np.max(x0, axis=1))
                y0_isvalid = 1 - y0_isnan
                x0_isvalid = 1 - x0_isnan
                valid_rows = x0_isvalid * y0_isvalid
                valid_stock_list = [stock_list[i] for i in range(stock_list.__len__()) if valid_rows[i] == 1]
                x0_valid = x0[valid_rows == 1]
                y0_valid = y0[valid_rows == 1]
                empty_industry = []
                if empty_industry.__len__() > 0:
                    x0_valid = np.delete(x0_valid, empty_industry, 1)
                if x0_valid.__len__() > 0:
                    b = np.linalg.inv(x0_valid.T.dot(x0_valid)).dot(x0_valid.T).dot(y0_valid)
                    residual = y0_valid - x0_valid.dot(b)  # 求残差
                    residual_dict = dict(zip(valid_stock_list, residual))  # 将残差与股票代码关联起来
                    residual_list.append(residual_dict)
                    residual_date_list.append(factor_date_list[j])
            # 将残差list一次性转变为DataFrame
            neutralized_factor_df = pd.DataFrame(residual_list, index=residual_date_list)
            t2 = dt.datetime.now()
            #print("neutralizing costs", t2 - t1)
            return neutralized_factor_df

    @staticmethod
    def make_one_hot(input_data):
        max_value = np.max(input_data) + 1
        result = (np.arange(max_value) == input_data[:, None]).astype(np.int)
        return result

    @staticmethod
    def data_filter2(data_df, filter_df, method='day'):
        """
        将停牌/涨跌停过滤掉
        """
        ans_df = data_df.copy()
        if method == 'day':
            threshold = 0.5
            ans_df[filter_df.reindex(index=ans_df.index, columns=ans_df.columns).values > threshold] = np.nan
            ans_df[np.isnan(filter_df)] = np.nan
        elif method == 'minute':
            ans_df[pd.DataFrame((abs(filter_df).values > np.exp(-10)),index = ans_df.index,columns = ans_df.columns)|np.isnan(filter_df)] = np.nan
        else:
            raise Exception("method only suport for day or minute!")
        return ans_df


    @staticmethod
    def industry_convert(ind):
        citics_to_industy3_dict = {'b101': 1, 'b102': 2, 'b103': 3, 'b104': 4, 'b105': 5, 'b106': 6, 'b107': 7,
                                   'b108': 8, 'b109': 9, 'b10a': 10, 'b10b': 11, 'b10c': 12, 'b10d': 13,
                                   'b10e': 14, 'b10f': 15, 'b10g': 16, 'b10h': 17, 'b10i': 18, 'b10j': 19,
                                   'b10k': 20, 'b10l': 21, 'b10n': 22, 'b10o': 23, 'b10p': 24, 'b10q': 25,
                                   'b10r': 26, 'b10s': 27, 'b10t': 28, 'b10m01': 29, 'b10m02': 30, 'b10m03': 31}
        def single_indcode_trans(indcode):
            if type(indcode) == float:  # item若有取值，则为str，若无取值，则类型为float
                return np.nan
            else:
                if indcode[0:4] == 'b10m':
                    return citics_to_industy3_dict[indcode]
                elif indcode[0:4] == 'b10u':
                    return citics_to_industy3_dict['b10m03']
                else:
                    return citics_to_industy3_dict[indcode[0:4]]

        ind_stack = ind.stack(dropna=False)
        transvalue=[single_indcode_trans(i) for i in ind_stack.values.tolist()]
        ind3_stack = deepcopy(ind_stack)
        ind3_stack[:] = transvalue
        ind3 = ind3_stack.unstack()
        return ind3
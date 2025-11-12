from xfactor.Util import *
from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
from xfactor.FixUtil import minute_data_transform,min_forward_adj
import pandas as pd
from copy import deepcopy

class m_vpc_mut_ty_log264(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.volume_minute","FactorData.Basic_factor.limit_status_minute",
                   "FactorData.Basic_factor.low_minute","FactorData.Basic_factor.high_minute","FactorData.Basic_factor.amt_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置

    reform_window=1

    batch_info = {'sub_window': 60,
   'daily_split_way': 'nt',
   'intraday_stat': 'kurt',
   'daily_stat': 'mean'}
    lag=10
    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']

        # 抛出异常的函数
        def raise_error():
            raise Exception

        # 获取dataframe列均值的函数
        def get_mean(df):
            return pd.Series(np.nanmean(df,axis=0),index = df.columns)

        # 获取dataframe列标准差的函数
        def get_std(df):
            return pd.Series(np.nanstd(df,axis=0),index = df.columns)

        # 获取dataframe列峰度的函数
        def get_kurt(df):
            return df.kurt(axis=0)

        # 获取dataframe列偏度的函数
        def get_skew(df):
            return df.skew(axis=0)

        # 获取dataframe列（均值/标准差）的函数
        def get_ms(df):
            avg = pd.Series(np.nanmean(df,axis=0),index = df.columns)
            std = pd.Series(np.nanstd(df,axis=0),index = df.columns)
            cv = pd.Series(std.values/avg.values, index = avg.index)
            std[pd.Series(cv.values<0.00001,index = cv.index)] = np.nan
            ms = pd.Series(avg.values/std.values, index = avg.index)
            return ms

        # 获取dataframe列自相关系数的函数
        def get_scm(df):
            def get_corresponding_corr(x_df, y_df):
                x_df.dropna(how='all', inplace=True)
                y_df.dropna(how='all', inplace=True)
                common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
                x_df = x_df.reindex(common_idx)
                y_df = y_df.reindex(common_idx)
                common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
                x_df = x_df[common_columns]
                y_df = y_df[common_columns]

                subdf1_array = x_df.values
                subdf2_array = y_df.values
                subcorr = np.nanmean(
                    (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (
                                subdf2_array - np.nanmean(subdf2_array, axis=0)),
                    axis=0) / (np.nanstd(subdf1_array, axis=0) * np.nanstd(subdf2_array, axis=0))
                subcorr = pd.Series(subcorr, index=x_df.columns)
                return subcorr
            half_length = df.shape[0]//2;
            idx_fh = df.index[:half_length];
            idx_lh = df.index[half_length:];
            df_fh = df.loc[idx_fh,:].reset_index(drop=True);
            df_lh = df.loc[idx_lh,:].reset_index(drop=True);
            self_corr = get_corresponding_corr(df_fh,df_lh)
            scm = self_corr
            return scm

        # 获取dataframe列时序beta的函数，由于各列解释变量x一样，此处直接简化求的协方差
        def get_tb(df):
            # 由于解释变量x一样，所以用协方差替代
            def get_corresponding_cov(x_df, y_df):
                x_df.dropna(how='all', inplace=True)
                y_df.dropna(how='all', inplace=True)
                common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
                x_df = x_df.reindex(common_idx)
                y_df = y_df.reindex(common_idx)
                common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
                x_df = x_df[common_columns]
                y_df = y_df[common_columns]

                subdf1_array = x_df.values
                subdf2_array = y_df.values
                subcov = np.nanmean(
                    (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (subdf2_array - np.nanmean(subdf2_array, axis=0)),
                    axis=0)
                subcov = pd.Series(subcov, index=x_df.columns)
                return subcov
            time_idx = deepcopy(df)
            time_idx[:] = np.tile(np.array(range(time_idx.shape[0])),(time_idx.shape[1],1)).T
            time_cov = get_corresponding_cov(df,time_idx)
            return time_cov

        # 获取dataframe列最小值的函数
        def get_min(df):
            return pd.Series(np.nanmin(df, axis=0),index = df.columns)

        # 获取dataframe列最大值的函数
        def get_max(df):
            return pd.Series(np.nanmax(df, axis=0),index = df.columns)

        # 获取dataframe每列差分的函数
        def get_dm(df):
            df_delta = df.values - df.shift(1).values
            return pd.Series(np.nanmean(df_delta,axis=0),index = df.columns)

        # 获取dataframe列排序后的自相关系数的函数
        def get_srcm(df):
            def get_corresponding_corr(x_df, y_df):
                x_df.dropna(how='all', inplace=True)
                y_df.dropna(how='all', inplace=True)
                common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
                x_df = x_df.reindex(common_idx)
                y_df = y_df.reindex(common_idx)
                common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
                x_df = x_df[common_columns]
                y_df = y_df[common_columns]

                subdf1_array = x_df.values
                subdf2_array = y_df.values
                subcorr = np.nanmean(
                    (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (
                                subdf2_array - np.nanmean(subdf2_array, axis=0)),
                    axis=0) / (np.nanstd(subdf1_array, axis=0) * np.nanstd(subdf2_array, axis=0))
                subcorr = pd.Series(subcorr, index=x_df.columns)
                return subcorr
            df = df.rank(axis=1, pct=True)
            half_length = df.shape[0]//2;
            idx_fh = df.index[:half_length];
            idx_lh = df.index[half_length:];
            df_fh = df.loc[idx_fh,:].reset_index(drop=True);
            df_lh = df.loc[idx_lh,:].reset_index(drop=True);
            self_corr = get_corresponding_corr(df_fh,df_lh)
            return self_corr

        # 获取dataframe列排序后的时序beta的函数，由于各列解释变量x一样，此处直接简化求的协方差
        def get_trb(df):
            def get_corresponding_cov(x_df, y_df):
                x_df.dropna(how='all', inplace=True)
                y_df.dropna(how='all', inplace=True)
                common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
                x_df = x_df.reindex(common_idx)
                y_df = y_df.reindex(common_idx)
                common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
                x_df = x_df[common_columns]
                y_df = y_df[common_columns]

                subdf1_array = x_df.values
                subdf2_array = y_df.values
                subcov = np.nanmean(
                    (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (subdf2_array - np.nanmean(subdf2_array, axis=0)),
                    axis=0)
                subcov = pd.Series(subcov, index=x_df.columns)
                return subcov
            df = df.rank(axis=1, pct=True)
            time_idx = deepcopy(df)
            time_idx[:] = np.tile(np.array(range(time_idx.shape[0])),(time_idx.shape[1],1)).T
            time_cov = get_corresponding_cov(df,time_idx)
            return time_cov

        # 获取dataframe每列排序后差分的函数
        def get_rdm(df):
            df = df.rank(axis=1, pct=True)
            df_delta = df.values - df.shift(1).values
            return pd.Series(np.nanmean(df_delta,axis=0),index = df.columns)

        # 日内局部信号计算函数，此处的灵活性最高，可进行自定义的特征刻画
        def calc_intra_subinfo(group,min_ret):
            subidx = group['datetime'].values
            min_ret_sub = min_ret.loc[subidx]
            mrd_sub = min_ret_sub.values*(min_ret_sub.values<0)
            subinfo = pd.Series(np.nanstd(mrd_sub,axis=0),index = min_ret_sub.columns)
            return subinfo

        # 提取数据

        close = database.depend_data['FactorData.Basic_factor.close_minute']
        min_close = min_forward_adj(close)
        min_ret = pd.DataFrame(min_close.values / min_close.shift(1).values - 1, index=min_close.index,
                     columns=min_close.columns)
        database.depend_data['FactorData.Basic_factor.ret_minute'] = min_ret
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        min_ret = data_filter(min_ret, limit_status, method='minute')
        min_ret = min_ret.iloc[237:,:]
        min_close = data_filter(database.depend_data['FactorData.Basic_factor.close_minute'], limit_status, method='minute')
        min_close= min_forward_adj(min_close)
        min_close = min_close.iloc[237:,:]


        fordate = deepcopy(min_close)
        intraday_stat = self.batch_info['intraday_stat'] # 要计算的单日统计特征
        daily_stat = self.batch_info['daily_stat'] # 要计算的日间统计特征
        sub_window = self.batch_info['sub_window'] # 日内局部信息的窗口
        daily_split_way=self.batch_info['daily_split_way'] # 划分单日的方法

        # 获取日期和局部信号对应的index
        if daily_split_way == 'nt':
            date_splited = list(map(lambda x: x.date(), fordate.index.tolist()))
        elif daily_split_way == 'bw':
            date_splited =np.array(range(fordate.shape[0]-1,-1,-1))//237
        else:
            raise Exception('Wrong daily_split_way')
        datetime_df = pd.DataFrame(
            {'datetime': fordate.index.tolist(), 'date': date_splited})
        datetime_df['time_id'] = datetime_df['datetime'].groupby(datetime_df['date']).apply(lambda x:pd.Series(np.array(range(len(x)))//sub_window,index = x.index)).values

        # 计算局部信号
        subinfo_df =datetime_df.groupby(['date','time_id']).apply(lambda x:calc_intra_subinfo(x,min_ret))

        # 计算单日日内特征
        intrafun_choice = {
            'mean':get_mean,
            'std':get_std,
            'skew':get_skew,
            'kurt':get_kurt,
            'ms':get_ms,
            'scm':get_scm,
            'tb':get_tb,
            'max':get_max,
            'min':get_min,
            'dm':get_dm,
            'srcm':get_srcm,
            'trb':get_trb,
            'rdm':get_rdm
        }
        date_sidx = subinfo_df.reset_index()[['date','time_id']].set_index(subinfo_df.index) #提取局部信号的multiindex信息，为下一步进行日间的聚合
        dailyinfo_df = subinfo_df.groupby(date_sidx['date']).apply(lambda x:intrafun_choice.get(intraday_stat,raise_error)(x)) #计算单日的信号

        # 计算日间信号
        dailyfun_choice = intrafun_choice
        alpha = dailyfun_choice.get(daily_stat,raise_error)(dailyinfo_df)
        if alpha.count()==0 or alpha[alpha!=0].count()==0:
            raise Exception('skip')

        return alpha

    def reform(self, temp_result):
        alpha = temp_result

        return alpha



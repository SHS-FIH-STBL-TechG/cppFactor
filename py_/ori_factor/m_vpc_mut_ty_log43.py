from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from xfactor.FixUtil import minute_data_transform,min_forward_adj


class m_vpc_mut_ty_log43(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.amt_minute","FactorData.Basic_factor.volume_minute","FactorData.Basic_factor.high_minute","FactorData.Basic_factor.low_minute","FactorData.Basic_factor.close-index_minute","FactorData.Basic_factor.citics_indcode1","FactorData.Basic_factor.close","FactorData.Basic_factor.mkt_cap_ard","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag=3
    reform_window=1
    batch_info={'mini_factor':'ret_top','stat':'skew','sub_window':5}

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation = ["drop1", "drop4"])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']
        # 提取数据
        amt_df = database.depend_data['FactorData.Basic_factor.amt_minute']
        amt_df = self.data_filter2(amt_df, limit_status, method='minute')
        amt_df = amt_df.iloc[-237*(self.lag):,:]
        fordate = deepcopy(amt_df)
        sub_window = self.batch_info['sub_window']
        self.mini_factor = self.batch_info['mini_factor']
        self.stat = self.batch_info['stat']
        dt_series = pd.Series([i // sub_window for i in range(fordate.shape[0])], index=fordate.index)

        # 计算局部信号
        # 日内局部信号计算函数，此处的灵活性最高，可进行自定义的特征刻画
        def calc_intra_subinfo(group,min_amt):
            min_amt_sub = min_amt.loc[group.index]
            subinfo = pd.Series(np.nanmean(min_amt_sub,axis=0),index = min_amt_sub.columns)
            return subinfo

        subinfo_df =dt_series.groupby(dt_series).apply(lambda x:calc_intra_subinfo(x,amt_df)).unstack()
        base = pd.Series(list(range(subinfo_df.shape[0])),index = subinfo_df.index)
        base_tile = pd.DataFrame(np.tile(base,(subinfo_df.shape[1],1)).T,index = subinfo_df.index,columns = subinfo_df.columns)

        # 计算单日日内特征
        if self.mini_factor=='raw':
            if self.stat=='avg':
                ans = subinfo_df.mean()
            elif self.stat=='std':
                ans = subinfo_df.std()
            elif self.stat=='kurt':
                ans = subinfo_df.kurt()
            elif self.stat=='skew':
                ans = subinfo_df.skew()
            elif self.stat=='corr':
                ans = subinfo_df.corrwith(base)
            elif self.stat=='cov':
                ans = subinfo_df.corrwith(base)*subinfo_df.std()
            else:
                raise Exception('Wrong_stat')
            return ans

        if self.mini_factor[-3:] == 'top':
            mini_factor_name = self.mini_factor[:-4]
        elif self.mini_factor[-4:] == 'tail':
            mini_factor_name = self.mini_factor[:-5]

        #1. 自身的边缘
        if mini_factor_name == 'ret':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            close_5 = close.groupby(dt_series).last()
            minifactor = pd.DataFrame(close_5.values/close_5.shift(1).values-1,index = close_5.index,columns = close_5.columns).fillna(0)
            minifactor=minifactor.mean(axis=1)

        if mini_factor_name == 'amt_std':
            minifactor = amt_df.sum(axis=1).groupby(dt_series).std()

        if mini_factor_name == 'ret_std':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret = pd.DataFrame(close.values/close.shift(1).values -1, index = close.index, columns = close.columns).mean(axis=1)
            minifactor = ret.groupby(dt_series).std()

        if mini_factor_name == 'illq':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            close_5 = close.groupby(dt_series).last()
            ret_5 = pd.DataFrame(close_5.values/close_5.shift(1).values-1,index = close_5.index,columns = close_5.columns).fillna(0)
            volume = database.depend_data['FactorData.Basic_factor.volume_minute']
            volume = self.data_filter2(volume, limit_status, method='minute')
            volume_5 = volume.groupby(dt_series).sum()
            minifactor = ret_5.mean(axis=1)/volume_5.mean(axis=1)*100000


        if mini_factor_name == 'hl':
            high = database.depend_data['FactorData.Basic_factor.high_minute']
            high = self.data_filter2(high, limit_status, method='minute')
            high = min_forward_adj(high)
            low = database.depend_data['FactorData.Basic_factor.low_minute']
            low = self.data_filter2(low, limit_status, method='minute')
            low = min_forward_adj(low)
            high_5 = high.groupby(dt_series).max()
            low_5 = low.groupby(dt_series).min()
            minifactor = high_5/low_5-1
            minifactor = minifactor.mean(axis=1)


        if mini_factor_name == 'ret_skew':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret = pd.DataFrame(close.values/close.shift(1).values -1, index = close.index, columns = close.columns)
            minifactor = ret.mean(axis=1).groupby(dt_series).skew()

        if mini_factor_name == 'sizestyle':
            idxclose = database.depend_data['FactorData.Basic_factor.close-index_minute']
            idxret = pd.DataFrame(idxclose.values/idxclose.shift(1).values-1,index = idxclose.index,columns = idxclose.columns)
            idxret_diff = idxret['399101.SZ']-idxret['000300.SH']
            minifactor = idxret_diff.groupby(dt_series).mean()

        if mini_factor_name == 'mktskew':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret = pd.DataFrame(close.values/close.shift(1).values -1, index = close.index, columns = close.columns)
            minifactor = ret.groupby(dt_series).mean().skew(axis=1)

        if mini_factor_name == 'indamtpctstd':
            stock_industry = database.depend_data['FactorData.Basic_factor.citics_indcode1']
            idx_convert = pd.DataFrame({'datetime': amt_df.index,
                                        'mindate': [int(datetime.strftime(i, '%Y%m%d')) for i in
                                                    amt_df.index]}).set_index('datetime')
            stock_industry.index = [int(i) for i in stock_industry.index]
            stock_industry = stock_industry.reindex(idx_convert['mindate'].drop_duplicates().values.tolist())
            stock_industry_corres = stock_industry.reindex(idx_convert['mindate'])
            stock_industry_corres.index = amt_df.index
            ind_amt = amt_df.apply(lambda x: x.groupby(stock_industry_corres.loc[x.name]).sum(), axis=1)
            tot_amt = ind_amt.sum(axis=1)
            tot_amt_tile = pd.DataFrame(np.tile(tot_amt, (ind_amt.shape[1], 1)).T, index=ind_amt.index, columns=ind_amt.columns)
            ind_amt_perc = ind_amt / tot_amt_tile
            ind_amt_perc_5 = ind_amt_perc.groupby(dt_series).mean()
            minifactor = ind_amt_perc_5.std(axis=1)

        if mini_factor_name == 'hlimitnum':
            thrsh = 0.097
            daily_close = database.depend_data['FactorData.Basic_factor.close']
            adjfactor = database.depend_data['FactorData.Basic_factor.adjfactor']
            daily_close = daily_close*adjfactor
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            idx_convert = pd.DataFrame({'datetime':close.index,'mindate':[int(datetime.strftime(i,'%Y%m%d')) for i in close.index]}).set_index('datetime')
            daily_close.index = idx_convert['mindate'].drop_duplicates().values.tolist()
            daily_close_corres=daily_close.reindex(idx_convert['mindate'])
            daily_close_corres.index = close.index
            cross_ret = pd.DataFrame(close.values/daily_close_corres.values-1,index = close.index,columns = close.columns)
            hlimit = pd.DataFrame(cross_ret.values>thrsh,index = cross_ret.index,columns=cross_ret.columns).sum(axis=1)
            hlimit_5 = hlimit.groupby(dt_series).mean()
            minifactor = deepcopy(hlimit_5)
            pass

        if mini_factor_name == 'llimitnum':
            thrsh = -0.097
            daily_close = database.depend_data['FactorData.Basic_factor.close']
            adjfactor = database.depend_data['FactorData.Basic_factor.adjfactor']
            daily_close = daily_close*adjfactor
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            idx_convert = pd.DataFrame({'datetime':close.index,'mindate':[int(datetime.strftime(i,'%Y%m%d')) for i in close.index]}).set_index('datetime')
            daily_close.index = idx_convert['mindate'].drop_duplicates().values.tolist()
            daily_close_corres=daily_close.reindex(idx_convert['mindate'])
            daily_close_corres.index = close.index
            cross_ret = pd.DataFrame(close.values/daily_close_corres.values-1,index = close.index,columns = close.columns)
            llimit = pd.DataFrame(cross_ret.values<thrsh,index = cross_ret.index,columns=cross_ret.columns).sum(axis=1)
            llimit_5 = llimit.groupby(dt_series).mean()
            minifactor = deepcopy(llimit_5)
            pass

        if mini_factor_name == 'timenearopen':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            timenearopen = deepcopy(close.mean(axis=1))
            timenearopen[:]=0
            timenearopen_list = [i for i in timenearopen.index if int(datetime.strftime(i,'%H%M'))<1030 and int(datetime.strftime(i,'%H%M'))>=930]
            timenearopen.loc[timenearopen_list]=1
            minifactor = timenearopen.groupby(dt_series).min()
            pass

        if mini_factor_name == 'timenearclose':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            timenearclose = deepcopy(close.mean(axis=1))
            timenearclose[:]=0
            timenearclose_list = [i for i in timenearclose.index if int(datetime.strftime(i,'%H%M'))<1500 and int(datetime.strftime(i,'%H%M'))>=1400]
            timenearclose.loc[timenearclose_list]=1
            minifactor = timenearclose.groupby(dt_series).min()
            pass

        if mini_factor_name =='idxmadiff':
            idxclose = database.depend_data['FactorData.Basic_factor.close-index_minute']['399006.SZ']
            idxclose_mal = idxclose.rolling(window=30,min_periods=1).mean()
            idxclose_mas = idxclose.rolling(window=5, min_periods=1).mean()
            idxmadiff = idxclose_mas-idxclose_mal
            minifactor = idxmadiff.groupby(dt_series).mean()

        if mini_factor_name =='watchbreakdiff':
            daily_close = database.depend_data['FactorData.Basic_factor.close']
            adjfactor = database.depend_data['FactorData.Basic_factor.adjfactor']
            daily_close = daily_close*adjfactor
            high = database.depend_data['FactorData.Basic_factor.high_minute']
            high = self.data_filter2(high, limit_status, method='minute')
            high = min_forward_adj(high)
            idx_convert = pd.DataFrame({'datetime':high.index,'mindate':[int(datetime.strftime(i,'%Y%m%d')) for i in high.index]}).set_index('datetime')
            daily_close.index = idx_convert['mindate'].drop_duplicates().values.tolist()
            daily_close_corres=daily_close.reindex(idx_convert['mindate'])
            daily_close_corres.index = high.index
            cross_ret = pd.DataFrame(high.values/daily_close_corres.values-1,index = high.index,columns = high.columns)
            wbh = pd.DataFrame(cross_ret.values>0,index = cross_ret.index,columns = cross_ret.columns)
            wbl = pd.DataFrame(cross_ret.values < 0, index=cross_ret.index, columns=cross_ret.columns)
            wbdiff = wbh.sum(axis=1)-wbl.sum(axis=1)
            minifactor = wbdiff.groupby(dt_series).mean()
            pass

        if mini_factor_name == 'mktconsist':
            idxclose = database.depend_data['FactorData.Basic_factor.close-index_minute'][['399101.SZ','000300.SH','399006.SZ']]
            mkt_consist = idxclose.groupby(dt_series).corr()
            minifactor = mkt_consist.mean(axis=1).unstack().min(axis=1)

        if mini_factor_name == 'ret_crsstd':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret_crsstd = pd.DataFrame(close.values/close.shift(1).values -1, index = close.index, columns = close.columns).std(axis=1)
            minifactor =ret_crsstd.groupby(dt_series).mean()

        if mini_factor_name == 'stkret_scl50':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret = close / close.shift(1) - 1
            mkt_cap = database.depend_data['FactorData.Basic_factor.mkt_cap_ard']
            idx_convert = pd.DataFrame({'datetime':close.index,'mindate':[int(datetime.strftime(i,'%Y%m%d')) for i in close.index]}).set_index('datetime')
            mkt_cap.index = idx_convert['mindate'].drop_duplicates().values.tolist()
            mkt_cap_corres=mkt_cap.reindex(idx_convert['mindate'])
            mkt_cap_corres.index = ret.index
            mkt_cap_l50 = pd.DataFrame(mkt_cap_corres.values < 500000,index=mkt_cap_corres.index,columns = mkt_cap_corres.columns)
            ret_l50 = ret[mkt_cap_l50]
            ret_l50_selfcorr = ret_l50.corrwith(ret_l50.shift(1), axis=1)
            minifactor = ret_l50_selfcorr.groupby(dt_series).mean()

        if mini_factor_name == 'indadjrstd':
            close = database.depend_data['FactorData.Basic_factor.close_minute']
            close = self.data_filter2(close, limit_status, method='minute')
            close = min_forward_adj(close)
            ret = close / close.shift(1) - 1
            ret = ret.fillna(0)
            stock_industry = database.depend_data['FactorData.Basic_factor.citics_indcode1']
            idx_convert = pd.DataFrame({'datetime': ret.index,
                                        'mindate': [int(datetime.strftime(i, '%Y%m%d')) for i in
                                                    ret.index]}).set_index('datetime')
            stock_industry.index = [int(i) for i in stock_industry.index]
            stock_industry = stock_industry.reindex(idx_convert['mindate'].drop_duplicates().values.tolist())
            stock_industry_corres = stock_industry.reindex(idx_convert['mindate'])
            stock_industry_corres.index = ret.index
            max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            ind_ret_std = ret.apply(lambda x: x.groupby(stock_industry_corres.loc[x.name]).apply(max_min_scaler).std(), axis=1)
            minifactor =ind_ret_std.groupby(dt_series).mean()
        minifactor = pd.DataFrame(np.tile(minifactor, (amt_df.shape[1], 1)).T, index=minifactor.index, columns=amt_df.columns)
        if self.mini_factor.split('_')[-1] == 'top':
            minifactor_rank = minifactor.rank(pct=True,ascending=False)
            if minifactor_rank.shape[0]//5>=5:
                subinfo_df_new = subinfo_df[pd.DataFrame(minifactor_rank.values<0.25 ,index = minifactor_rank.index,columns = minifactor_rank.columns)]
            else:
                subinfo_df_new = subinfo_df[pd.DataFrame(minifactor_rank.values<0.5 ,index = minifactor_rank.index,columns = minifactor_rank.columns)]
        else:
            minifactor_rank = minifactor.rank(pct=True, ascending=True)
            if minifactor_rank.shape[0]//5>=5:
                subinfo_df_new = subinfo_df[pd.DataFrame(minifactor_rank.values<0.25 ,index = minifactor_rank.index,columns = minifactor_rank.columns)]
            else:
                subinfo_df_new = subinfo_df[pd.DataFrame(minifactor_rank.values<0.5 ,index = minifactor_rank.index,columns = minifactor_rank.columns)]

        if self.stat == 'avg':
            ans_new = subinfo_df_new.mean()
        elif self.stat == 'std':
            ans_new = subinfo_df_new.std()
        elif self.stat == 'skew':
            ans_new = subinfo_df_new.skew()
        elif self.stat == 'corr':
            ans_new = self.get_corresponding_corr(subinfo_df_new,base_tile)
        elif self.stat == 'cov':
            ans_new = self.get_corresponding_corr(subinfo_df_new,base_tile) * subinfo_df_new.std()
        else:
            raise Exception('Wrong_stat')
        alpha= deepcopy(ans_new)
        if alpha.count()==0 or alpha[alpha!=0].count()==0:
            raise Exception('skip')
        return alpha

    def reform(self, temp_result):
        alpha = temp_result
        return alpha

    @staticmethod
    def get_corresponding_corr(x_df, y_df):
        x_df.dropna(how='all', inplace=True)
        y_df.dropna(how='all', inplace=True)
        common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
        x_df = x_df.reindex(common_idx)
        y_df = y_df.reindex(common_idx)
        subdf1_array = x_df.values
        subdf2_array = y_df.values
        subcorr = np.nanmean(((subdf1_array - np.nanmean(subdf1_array,axis=0)) * (subdf2_array - np.nanmean(subdf2_array))),axis=0)/ (np.nanstd(subdf1_array,axis=0) * np.nanstd(subdf2_array,axis=0))
        subcorr = pd.Series(subcorr, index=x_df.columns)
        return subcorr


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
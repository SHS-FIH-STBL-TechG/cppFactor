from xfactor.Util import *
from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
from xfactor.FixUtil import minute_data_transform,min_forward_adj
import statsmodels.api as sm
from copy import deepcopy
import pandas as pd

class m_vpc_mut_ty_log13(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.volume_minute","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag =4
    # 播放后得到的结果，可按照该长度进行rolling等计算，具体rolling方法需要在reform方法中定义。 默认为1，可不设置。
    reform_window = 1

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']

        close = data_filter(database.depend_data['FactorData.Basic_factor.close_minute'],limit_status,method='minute')
        close_adj=min_forward_adj(close)
        ret=pd.DataFrame(close_adj.values / close_adj.shift(1).values - 1, index=close_adj.index,
                     columns=close_adj.columns)
        database.depend_data['FactorData.Basic_factor.ret_minute'] = ret
        minute_data_transform(database.depend_data, operation=["drop1", "drop4"])
        ret_adj = database.depend_data['FactorData.Basic_factor.ret_minute'].copy()
        ret_adj.fillna(0, inplace=True)
        def get_corresponding_corr(x_df, y_df):
            x_df.dropna(how='all', inplace=True)
            y_df.dropna(how='all', inplace=True)
            common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
            x_df = x_df.reindex(common_idx)
            y_df = y_df.reindex(common_idx)
            subdf1_array = x_df.values
            subdf2_array = y_df.values
            subcorr = np.nanmean(
                (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (subdf2_array - np.nanmean(subdf2_array, axis=0)),
                axis=0) / (np.nanstd(subdf1_array, axis=0) * np.nanstd(subdf2_array, axis=0))
            subcorr = pd.Series(subcorr, index=x_df.columns)
            return subcorr

        def numpy_ewma_vectorized(data, window):
            data_np = data.values
            alpha = 2 / (window + 1.0)
            alpha_rev = 1 - alpha
            n = data_np.shape[0]
            pows = alpha_rev ** (np.arange(n + 1))
            scale_arr = 1 / pows[:-1]
            pw0 = alpha * alpha_rev ** (n - 1)
            mult = data_np * np.tile(pw0 * scale_arr, (data_np.shape[1], 1)).T
            out = pd.DataFrame(mult, index=data.index, columns=data.columns).sum()
            return out

        MA = numpy_ewma_vectorized(ret_adj, len(ret_adj)) * 100
        volume_adj = data_filter(database.depend_data['FactorData.Basic_factor.volume_minute'],limit_status,method='minute')
        corr = get_corresponding_corr(close_adj,volume_adj)
        adj_MA = MA.rank() * corr.rank()
        return adj_MA



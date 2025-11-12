from xfactor.Util import *
from xfactor.BaseFactor import BaseFactor
import xfactor.Util as Util
import numpy as np
from xfactor.FixUtil import minute_data_transform,min_forward_adj


class m_vpc_mut_ty_log95(BaseFactor):
    #  定义因子参数

    # 因子频率，默认为日频因子， 可不设置
    factor_type = "FIX"
    # 依赖的平台原始数据，包括FactorData和MarketData接口中的数据。 默认为空，必须设置
    depend_data = ["FactorData.Basic_factor.close_minute","FactorData.Basic_factor.close-index_minute","FactorData.Basic_factor.limit_status_minute"]
    # 计算每个时点的因子所需要前移的数据窗口大小
    # 例如，为日频因子，lag=3表示计算某一日的因子值需要依赖前三个交易日和当日的数据，默认为0，可不设置
    lag=0
    reform_window=1
    benchmark='000300.SH'

    # 每次播放的计算具体方法。必须实现。
    def calc_single(self, database):
        #分钟线242转换为240或者241根，operation为list，默认为["merge", "merge"],第一位表示对925时刻处理，第二位表示对1500处理
        #处理方式可分为"merge"、"drop"和"",分别表示合并、删除、和不操作。优化后单次播放时分钟线转换速度为毫秒级
        minute_data_transform(database.depend_data, operation = ["", ""])
        limit_status = database.depend_data['FactorData.Basic_factor.limit_status_minute']


        close = data_filter(database.depend_data['FactorData.Basic_factor.close_minute'],limit_status,method='minute')
        close = min_forward_adj(close)
        close_bench = data_filter(database.depend_data['FactorData.Basic_factor.close-index_minute'],limit_status,method='minute')
        close_bench = close_bench.loc[:, self.benchmark:self.benchmark]
        ret = np.log(close/close.shift(1))
        ret_benchmark = np.log(close_bench/close_bench.shift(1))
        beta_square = ((ret.mul(np.array(ret_benchmark)))**2).sum(axis=0) / np.array((ret_benchmark**4).sum(axis=0))
        ans = np.sqrt(beta_square)
        ans[ans == 0] = np.nan
        return ans





    def reform(self, temp_result):
        alpha = temp_result
        return alpha



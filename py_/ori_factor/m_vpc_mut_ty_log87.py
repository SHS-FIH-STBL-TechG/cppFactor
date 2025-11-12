import numpy as np
import pandas as pd
from xfactor.BaseFactor import BaseFactor
from xfactor.FixUtil import minute_data_transform


class m_vpc_mut_ty_log87(BaseFactor):
    depend_data = ['FactorData.Basic_factor.close_minute', 'FactorData.Basic_factor.amt_minute',
                   'FactorData.Basic_factor.volume_minute']
    factor_type = 'FIX'
    lag = 1
    reform_window = 5

    def calc_single(self, database):
        minute_data_transform(database.depend_data, ['drop', 'merge'])
        close = database.depend_data['FactorData.Basic_factor.close_minute']
        amt = database.depend_data['FactorData.Basic_factor.amt_minute']
        vol = database.depend_data['FactorData.Basic_factor.volume_minute']
        stk_code = close.columns
        twap = np.nanmean(close.values, axis=0)
        vwap = np.nansum(amt.values, axis=0) / np.nansum(vol.values, axis=0)
        result = pd.Series(twap / vwap, index=stk_code)
        return result

    def reform(self, temp_result):
        alpha = -temp_result.rolling(5, 1).std()
        return alpha

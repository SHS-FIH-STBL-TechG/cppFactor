#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 m_vpc_mut_ty_log1 因子实现（无平台依赖）
"""

from __future__ import annotations
from re import escape

import pandas as pd
import pdb

class m_vpc_mut_ty_log1:
    """
    分钟级异常交易后波动偏度因子
    """

    def __init__(self, lag: int, minute_lag: int = 0):
        self.lag = lag
        self.minute_lag = minute_lag
        self.std_period = 3
        self.rolling_time = 2

        self.updatenum = -1

    def calc_single(self, database) -> pd.Series:
        """
        在固定窗口内计算因子值

        参数:
            database: 拥有 depend_data 字段的对象，至少包含
                - FactorData.Basic_factor.amt_minute
                - FactorData.Basic_factor.close_adj_minute
        返回:
            pandas.Series: 每只股票的因子值
        """
        MinuteTurnover = database.depend_data["FactorData.Basic_factor.amt_minute"]
        MinuteClose = database.depend_data["FactorData.Basic_factor.close_adj_minute"]

        #不许改动
        std_period = 5
        add_signal = MinuteTurnover.diff() / MinuteTurnover.rolling(std_period).std()
        add_signal = (add_signal.values > 2.)

        #pdb.set_trace()

        rollingtime = 2
        UPDO = MinuteClose.diff(rollingtime).shift(-rollingtime).abs()

        f = (UPDO * add_signal).skew()
        #不许改动

        # #debug
        # self.updatenum += 1
        # if self.updatenum == 5:
        #     print("MinuteTurnover:")
        #     print(MinuteTurnover)
        #     print("MinuteClose:")
        #     print(MinuteClose)
        #     print("UPDO:")
        #     print(UPDO)
        #     print("add_signal:")
        #     print(add_signal)
        #     print("UPDO * add_signal:")
        #     print(UPDO * add_signal)

        return -f



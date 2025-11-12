# -*- coding: utf-8 -*-
"""
Barra CNE5 Beta因子计算（简化版，移除平台依赖）
"""

import pandas as pd
import numpy as np


class barra_cne5_1_beta:
    """Barra CNE5 Beta因子计算类"""
    
    def __init__(self, lag, reform_window):
        """
        初始化因子计算类
        
        参数:
            lag: 计算每个时点的因子所需要前移的数据窗口大小（必须提供）
            reform_window: 结果平滑窗口大小（必须提供）
        """
        self.lag = lag
        self.reform_window = reform_window

    def calc_single(self, database):
        """
        计算单点因子值
        
        参数:
            database: 包含depend_data字典的对象，包含以下键：
                - "FactorData.Basic_factor.pct_chg": 收益率数据 (DataFrame, 时间 x 股票)
                - "FactorData.Basic_factor.a_mkt_cap": 流通市值数据 (DataFrame, 时间 x 股票)
                - "FactorData.Basic_factor.is_valid_test": 有效性标记 (DataFrame, 时间 x 股票)
        
        返回:
            Series: 每只股票的beta值（标准化后）
        """
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"]
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        # 1. 对原始数据进行winsor处理
        rett = ret.T
        rett = rett.where(rett > -20, -20)  # 涨跌幅超过上限的winsor一下
        rett = rett.where(rett < 20, 20) * (ret.T / ret.T)
        ret = rett.T

        rm = (ret * float).sum(axis=1) / float.sum(axis=1)  # 用流通市值加权计算指数收益率
        beta = ret.corrwith(rm) * ret.std() / rm.std()  # 用等价公式计算beta
        beta = beta.clip(beta.quantile(0.005), beta.quantile(0.995))
        beta = pd.Series(beta, index = ret.columns)  * valid.iloc[-1]

        # 标准化
        beta_mean = (beta * float.iloc[-1]).sum() / float.iloc[-1].sum()
        ans = (beta - beta_mean) / beta.std()
        return ans

    def reform(self, temp_result):
        """
        对结果进行平滑处理
        
        参数:
            temp_result: 历史结果序列 (Series或DataFrame)
        
        返回:
            平滑后的结果
        """
        A = temp_result.rolling(self.reform_window, 1).mean()
        return A


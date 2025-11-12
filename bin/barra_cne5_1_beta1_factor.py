# -*- coding: utf-8 -*-
"""
Barra CNE5 Beta1因子计算（简化版，移除平台依赖）
"""

import pandas as pd
import numpy as np


class barra_cne5_1_beta1:
    """Barra CNE5 Beta1因子计算类"""
    
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
        ret = database.depend_data["FactorData.Basic_factor.pct_chg"] / 100
        float = database.depend_data["FactorData.Basic_factor.a_mkt_cap"]
        valid = database.depend_data['FactorData.Basic_factor.is_valid_test']

        # 1. 对原始数据进行winsor处理
        rett = (ret * valid.iloc[-1]).T
        rett = rett.where(rett > -20, -20)  # 涨跌幅超过上限的winsor一下
        rett = rett.where(rett < 20, 20) * (ret.T / ret.T)
        ret = rett.T
        
        rm = (ret * float).sum(axis=1) / float.sum(axis=1)
        cov = (ret - ret.mean()).ewm(halflife=2).cov(rm - rm.mean())
        vari = ret.ewm(halflife=2).var()
        beta = cov.iloc[-1] / vari.iloc[-1]


        # debug
        # 提高打印精度，输出时保留10位小数
        with pd.option_context('display.float_format', lambda x: f'{x:.10f}'):
            print("cov = ")
            print(cov.iloc[-1])
            print("vari = ")
            print(vari.iloc[-1])
            print("beta = ")
            print(beta)

        beta = beta.clip(beta.quantile(0.005), beta.quantile(0.995))   # winsor

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


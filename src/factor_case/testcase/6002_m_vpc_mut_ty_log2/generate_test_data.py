#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 6001_m_vpc_mut_ty_log1 因子测试用例数据
"""

import os
import numpy as np
import pandas as pd


def write_matrix(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, header=False, index=False, float_format="%.6f")


def main():
    np.random.seed(2048)

    num_timepoints = 600
    num_stocks = 300

    # 生成数据
    base_amt = np.random.lognormal(mean=12, sigma=0.5, size=(num_timepoints, num_stocks))
    noise = np.random.normal(scale=1e5, size=(num_timepoints, num_stocks))
    amt = pd.DataFrame(base_amt * 1e3 + noise, columns=[f"stock_{i}" for i in range(num_stocks)])

    price_moves = np.random.normal(loc=0.0, scale=0.5, size=(num_timepoints, num_stocks))
    close = pd.DataFrame(100 + np.cumsum(price_moves, axis=0), columns=[f"stock_{i}" for i in range(num_stocks)])

    # 保存输入
    root = os.path.dirname(__file__)
    write_matrix(amt, os.path.join(root, "input_amt.csv"))
    write_matrix(close, os.path.join(root, "input_close.csv"))

    print("数据生成完成。")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封装 m_vpc_mut_ty_log2 因子计算：
- 从 config.ini 读取路径与窗口参数
- 仅负责滑动拼接固定窗口数据
- 调用 m_vpc_mut_ty_log2 原算子输出结果
"""

import configparser
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from m_vpc_mut_ty_log2_factor import m_vpc_mut_ty_log2


class SimpleDatabase:
    """封装原算子所需的数据接口"""

    def __init__(self, amt: pd.DataFrame, close: pd.DataFrame):
        self.depend_data = {
            "FactorData.Basic_factor.amt_minute": amt,
            "FactorData.Basic_factor.close_adj_minute": close,
        }


def read_config(config_path: str) -> dict:
    """读取配置文件，仅提取必要字段"""
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    section = "6002_m_vpc_mut_ty_log2"
    if not parser.has_section(section):
        raise ValueError(f"配置文件缺少节: {section}")

    result = {}
    str_fields = {
        "input_amt": "input_amt",
        "input_close": "input_close",
        "output_csv": "output_csv",
    }
    int_fields = {
        "precision": "precision",
        "lag": "lagWindow",
        "step_size": "stepSize",
    }

    for key, option in str_fields.items():
        if not parser.has_option(section, option):
            raise ValueError(f"配置文件缺少必需项: [{section}] {option}")
        result[key] = parser.get(section, option)

    for key, option in int_fields.items():
        if not parser.has_option(section, option):
            raise ValueError(f"配置文件缺少必需项: [{section}] {option}")
        result[key] = parser.getint(section, option)

    return result


def read_csv_data(file_path: str) -> pd.DataFrame:
    """读取无表头CSV数据"""
    print(f"正在读取: {file_path}")
    data = pd.read_csv(file_path, header=None, dtype=float)
    print(f"  维度: {data.shape[0]} x {data.shape[1]}")
    return data


def format_value(value: float, precision: int) -> str:
    """输出格式化数值（兼容 NaN）"""
    if pd.isna(value) or np.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.ini")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] === 6002_m_vpc_mut_ty_log2 因子计算（Python封装） ===")
        print(f"[{timestamp}] 正在读取配置文件: {config_path}")

        config = read_config(config_path)

        # 读取输入数据
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在读取输入数据...")
        amt = read_csv_data(config["input_amt"])
        close = read_csv_data(config["input_close"])

        if amt.shape != close.shape:
            raise ValueError(f"输入数据维度不一致: amt {amt.shape}, close {close.shape}")

        num_timepoints, num_stocks = amt.shape
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 数据维度: {num_timepoints} x {num_stocks}")

        lag_window = config["lag"]
        step_size = config["step_size"]
        precision = config["precision"]

        if num_timepoints < lag_window:
            raise ValueError(f"时间点数({num_timepoints})小于窗口大小({lag_window})")

        # 因子实例仅负责计算，不再做配置解析（原始代码不接受参数）
        factor = m_vpc_mut_ty_log2()

        # 输出路径追加 _py
        original_output_path = config["output_csv"]
        if "." in os.path.basename(original_output_path):
            name, ext = os.path.splitext(original_output_path)
            output_path = name + "_py" + ext
        else:
            output_path = original_output_path + "_py"

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置参数:")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   lagWindow = {lag_window}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   stepSize = {step_size}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   precision = {precision}")

        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            header = "time_index,step,operation,elapsed_ms"
            for stock_idx in range(num_stocks):
                header += f",stock_{stock_idx}_mut_ty_log2"
            output_file.write(header + "\n")

            # 初始化阶段：取前 lagWindow 个时间点
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始初始化...")
            init_start = time.time()
            init_amt = amt.iloc[0:lag_window, :]
            init_close = close.iloc[0:lag_window, :]
            init_db = SimpleDatabase(init_amt, init_close)
            init_result = factor.calc_single(init_db)
            init_elapsed = (time.time() - init_start) * 1000
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 初始化完成，耗时: {init_elapsed:.3f} ms")

            init_row = f"{lag_window - 1},0,init,{init_elapsed:.3f}"
            for stock_idx in range(num_stocks):
                value = init_result.iloc[stock_idx] if stock_idx < len(init_result) else np.nan
                init_row += f",{format_value(value, precision)}"
            output_file.write(init_row + "\n")

            # 更新阶段：按 stepSize 滑动窗口
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始持续更新...")
            update_count = 0
            current_time = lag_window

            while current_time < num_timepoints:
                update_size = min(step_size, num_timepoints - current_time)
                window_end = current_time + update_size
                window_start = max(0, window_end - lag_window)

                window_amt = amt.iloc[window_start:window_end, :]
                window_close = close.iloc[window_start:window_end, :]
                database = SimpleDatabase(window_amt, window_close)

                update_start = time.time()
                update_result = factor.calc_single(database)
                update_elapsed = (time.time() - update_start) * 1000
                update_count += 1

                time_index = window_end - 1
                update_row = f"{time_index},{update_count},update,{update_elapsed:.3f}"
                for stock_idx in range(num_stocks):
                    value = update_result.iloc[stock_idx] if stock_idx < len(update_result) else np.nan
                    update_row += f",{format_value(value, precision)}"
                output_file.write(update_row + "\n")

                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"更新 #{update_count}: 时间点 {time_index}/{num_timepoints-1}, "
                    f"耗时: {update_elapsed:.3f} ms"
                )

                current_time += update_size

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 计算完成 ===")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 结果已保存到: {output_path}")

    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()



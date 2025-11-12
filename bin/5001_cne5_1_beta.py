#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
封装barra_cne5_1_beta因子计算，从config.ini读取配置，调用原算子计算，输出C++格式结果
"""

import sys
import os
import configparser
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 导入本地算子（简化版，无平台依赖）
from barra_cne5_1_beta_factor import barra_cne5_1_beta


class SimpleDatabase:
    """简单的数据库对象，用于封装原算子的接口"""
    def __init__(self, pct_chg, a_mkt_cap, is_valid):
        self.depend_data = {
            "FactorData.Basic_factor.pct_chg": pct_chg,
            "FactorData.Basic_factor.a_mkt_cap": a_mkt_cap,
            "FactorData.Basic_factor.is_valid_test": is_valid
        }


def read_config(config_path):
    """读取配置文件"""
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    
    section = "5001_cne5_1_beta"
    if not config.has_section(section):
        raise ValueError(f"配置文件中缺少节: {section}")
    
    # 读取必需配置项，如果不存在则报错
    required_keys = {
        'input_pct_chg': 'input_pct_chg',
        'input_a_mkt_cap': 'input_a_mkt_cap',
        'input_is_valid': 'input_is_valid',
        'output_csv': 'output_csv',
    }
    
    result = {}
    for key, config_key in required_keys.items():
        if not config.has_option(section, config_key):
            raise ValueError(f"配置文件中缺少必需的配置项: [{section}] {config_key}")
        result[key] = config.get(section, config_key)
    
    # 读取整数配置项，如果不存在则报错
    int_keys = {
        'precision': 'precision',
        'lag': 'lagWindow',
        'step_size': 'stepSize',
        'reform_window': 'reformWindow',
    }
    
    for key, config_key in int_keys.items():
        if not config.has_option(section, config_key):
            raise ValueError(f"配置文件中缺少必需的配置项: [{section}] {config_key}")
        result[key] = config.getint(section, config_key)
    
    return result


def read_csv_data(file_path):
    """读取CSV数据文件（无表头，纯数据）"""
    print(f"正在读取: {file_path}")
    data = pd.read_csv(file_path, header=None, dtype=float)
    print(f"  维度: {data.shape[0]} x {data.shape[1]}")
    return data


def format_value(value, precision):
    """格式化数值输出"""
    if pd.isna(value) or np.isnan(value):
        return "nan"
    return f"{value:.{precision}f}"


def main():
    """主函数"""
    try:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.ini")
        
        # 读取配置
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 5001_cne5_1_beta 因子计算（Python封装） ===")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在读取配置文件: {config_path}")
        config = read_config(config_path)
        
        # 读取输入数据
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 正在读取输入数据...")
        pct_chg = read_csv_data(config['input_pct_chg'])
        a_mkt_cap = read_csv_data(config['input_a_mkt_cap'])
        is_valid = read_csv_data(config['input_is_valid'])
        
        # 验证数据维度
        num_timepoints = pct_chg.shape[0]
        num_stocks = pct_chg.shape[1]
        
        if a_mkt_cap.shape != (num_timepoints, num_stocks):
            raise ValueError(f"市值数据维度不匹配: {a_mkt_cap.shape} vs ({num_timepoints}, {num_stocks})")
        if is_valid.shape != (num_timepoints, num_stocks):
            raise ValueError(f"有效性数据维度不匹配: {is_valid.shape} vs ({num_timepoints}, {num_stocks})")
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 数据维度: {num_timepoints} x {num_stocks}")
        
        # 从配置读取窗口参数
        lag_window = config['lag']
        step_size = config['step_size']
        reform_window = config['reform_window']
        
        # 创建因子实例，传入配置参数
        factor = barra_cne5_1_beta(lag=lag_window, reform_window=reform_window)
        
        # 检查数据量是否足够
        if num_timepoints < lag_window:
            raise ValueError(f"时间点数({num_timepoints})小于lag窗口大小({lag_window})")
        
        # 准备输出文件（添加_py后缀）
        original_output_path = config['output_csv']
        # 在文件名后添加_py后缀
        if '.' in os.path.basename(original_output_path):
            # 有扩展名的情况：output.csv -> output_py.csv
            name, ext = os.path.splitext(original_output_path)
            output_path = name + '_py' + ext
        else:
            # 无扩展名的情况：output -> output_py
            output_path = original_output_path + '_py'
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 配置参数:")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   lagWindow = {lag_window}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   stepSize = {step_size}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   reformWindow = {reform_window}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   precision = {config['precision']}")
        
        # 打开输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入注释行
            f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            
            # 写入表头
            header = "time_index,step,operation,elapsed_ms"
            for i in range(num_stocks):
                header += f",stock_{i}_beta"
            f.write(header + "\n")
            
            # ========== 初始化阶段：使用前lagWindow个时间点的数据 ==========
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始初始化...")
            init_start = time.time()
            
            # 提取前lagWindow个时间点的数据（固定窗口大小）
            init_ret = pct_chg.iloc[0:lag_window, :]
            init_cap = a_mkt_cap.iloc[0:lag_window, :]
            init_valid = is_valid.iloc[lag_window-1:lag_window, :]  # 使用最后一个时间点的有效性
            
            # 构造database对象
            init_database = SimpleDatabase(init_ret, init_cap, init_valid)
            
            # 调用原算子的calc_single方法（全量计算，但输入窗口固定为lagWindow）
            init_result = factor.calc_single(init_database)
            
            init_elapsed = (time.time() - init_start) * 1000
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 初始化完成，耗时: {init_elapsed:.3f} ms")
            
            # 输出初始化结果
            time_index = lag_window - 1
            row = f"{time_index},0,init,{init_elapsed:.3f}"
            for i in range(num_stocks):
                if i < len(init_result):
                    value = init_result.iloc[i]
                    row += f",{format_value(value, config['precision'])}"
                else:
                    row += ",nan"
            f.write(row + "\n")
            
            # ========== 持续更新阶段：按stepSize逐步处理后续数据 ==========
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始持续更新...")
            update_count = 0
            current_time = lag_window
            
            while current_time < num_timepoints:
                # 计算本次更新的数据量（不超过stepSize）
                update_size = min(step_size, num_timepoints - current_time)
                
                # 计算窗口的结束位置（当前时间点 + 本次更新的数据量）
                window_end = current_time + update_size
                
                # 计算窗口的起始位置（保持固定窗口大小为lagWindow）
                # 窗口应该是：从 window_end - lag_window 到 window_end
                window_start = max(0, window_end - lag_window)
                
                # 提取窗口数据（固定窗口大小，最多为lagWindow，如果不足则从0开始）
                window_ret = pct_chg.iloc[window_start:window_end, :]
                window_cap = a_mkt_cap.iloc[window_start:window_end, :]
                window_valid = is_valid.iloc[window_end-1:window_end, :]  # 使用最后一个时间点的有效性
                # 构造database对象
                update_database = SimpleDatabase(window_ret, window_cap, window_valid)
                
                # 记录更新时间
                update_start = time.time()
                
                
                # 调用原算子的calc_single方法（全量计算，但输入窗口固定大小）
                update_result = factor.calc_single(update_database)
                
                update_elapsed = (time.time() - update_start) * 1000
                update_count += 1
                
                # 输出更新结果
                time_index = current_time + update_size - 1
                row = f"{time_index},{update_count},update,{update_elapsed:.3f}"
                for i in range(num_stocks):
                    if i < len(update_result):
                        value = update_result.iloc[i]
                        row += f",{format_value(value, config['precision'])}"
                    else:
                        row += ",nan"
                f.write(row + "\n")
                
                # 每次更新都输出耗时信息
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 更新 #{update_count}: 时间点 {time_index}/{num_timepoints-1}, 耗时: {update_elapsed:.3f} ms")
                
                current_time += update_size
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === 计算完成 ===")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


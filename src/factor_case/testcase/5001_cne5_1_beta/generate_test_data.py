#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成5001_cne5_1_beta测试案例的输入数据
生成2000个股票、300000个时间点的数据
优化版本：使用numpy直接写入，分批处理，多线程并行生成，提升生成速度
"""

import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def write_array_to_csv_fast(data, filename, chunk_size=50000):
    """快速将numpy数组写入CSV文件，分批写入避免内存溢出"""
    print(f"正在写入 {filename}...")
    start_time = time.time()
    
    num_rows = data.shape[0]
    num_cols = data.shape[1]
    
    # 使用文本模式和大缓冲区，提升I/O速度
    with open(filename, 'w', encoding='utf-8', buffering=1024*1024) as f:  # 1MB缓冲区
        # 分批写入
        for i in range(0, num_rows, chunk_size):
            end_idx = min(i + chunk_size, num_rows)
            chunk = data[i:end_idx]
            
            # 使用numpy的savetxt直接写入文件，这是最快的方法
            # 根据数据类型选择格式化字符串
            if filename == 'input_valid.csv':
                fmt_str = '%.0f'  # 有效性数据是整数，不需要小数
            else:
                fmt_str = '%.6e'  # 其他数据使用科学计数法
            np.savetxt(f, chunk, delimiter=',', fmt=fmt_str, newline='\n')
            
            if (i // chunk_size + 1) % 5 == 0 or end_idx == num_rows:
                elapsed = time.time() - start_time
                progress = (end_idx / num_rows) * 100
                print(f"  进度: {progress:.1f}% ({end_idx}/{num_rows} 行) - 已用时: {elapsed:.1f}秒")
    
    elapsed = time.time() - start_time
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    print(f"  {filename} 写入完成: {data.shape} - 大小: {file_size:.2f} MB - 用时: {elapsed:.1f}秒")
    return elapsed

def generate_and_write_ret(num_timepoints, num_stocks, seed_offset=0):
    """生成并写入收益率数据"""
    np.random.seed(42 + seed_offset)
    print(f"[线程1] 开始生成收益率数据...")
    gen_start = time.time()
    returns = np.random.normal(0.0, 0.02, size=(num_timepoints, num_stocks))
    gen_time = time.time() - gen_start
    print(f"[线程1] 数据生成完成，用时: {gen_time:.1f}秒")
    
    write_time = write_array_to_csv_fast(returns, 'input_ret.csv', chunk_size=50000)
    stats = {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns),
        'gen_time': gen_time,
        'write_time': write_time
    }
    del returns
    print(f"[线程1] 收益率数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_and_write_cap(num_timepoints, num_stocks, seed_offset=1):
    """生成并写入市值数据"""
    np.random.seed(42 + seed_offset)
    print(f"[线程2] 开始生成市值数据...")
    gen_start = time.time()
    log_cap = np.random.normal(9.0, 1.5, size=(num_timepoints, num_stocks))
    cap = np.exp(log_cap)
    gen_time = time.time() - gen_start
    print(f"[线程2] 数据生成完成，用时: {gen_time:.1f}秒")
    
    write_time = write_array_to_csv_fast(cap, 'input_cap.csv', chunk_size=50000)
    stats = {
        'mean': np.mean(cap),
        'std': np.std(cap),
        'min': np.min(cap),
        'max': np.max(cap),
        'gen_time': gen_time,
        'write_time': write_time
    }
    del cap, log_cap
    print(f"[线程2] 市值数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_and_write_valid(num_timepoints, num_stocks, seed_offset=2):
    """生成并写入有效性数据"""
    np.random.seed(42 + seed_offset)
    print(f"[线程3] 开始生成有效性数据...")
    gen_start = time.time()
    valid_prob = 0.9
    valid = np.random.binomial(1, valid_prob, size=(num_timepoints, num_stocks)).astype(float)
    gen_time = time.time() - gen_start
    print(f"[线程3] 数据生成完成，用时: {gen_time:.1f}秒")
    
    write_time = write_array_to_csv_fast(valid, 'input_valid.csv', chunk_size=50000)
    stats = {
        'mean': np.mean(valid),
        'sum': np.sum(valid),
        'total': valid.size,
        'gen_time': gen_time,
        'write_time': write_time
    }
    del valid
    print(f"[线程3] 有效性数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_5001_test_data():
    """生成5001测试案例的输入数据（多线程并行版本）"""
    
    total_start = time.time()
    
    num_stocks = 1000
    num_timepoints = 3000
    
    print(f"正在生成 {num_timepoints} 个时间点 x {num_stocks} 只股票的数据...")
    print("使用多线程并行生成...")
    print("=" * 60)
    
    # 使用线程池并行生成和写入三个文件
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交三个任务
        future_ret = executor.submit(generate_and_write_ret, num_timepoints, num_stocks, 0)
        future_cap = executor.submit(generate_and_write_cap, num_timepoints, num_stocks, 1)
        future_valid = executor.submit(generate_and_write_valid, num_timepoints, num_stocks, 2)
        
        # 等待所有任务完成并收集结果
        results['ret'] = future_ret.result()
        results['cap'] = future_cap.result()
        results['valid'] = future_valid.result()
    
    ret_stats = results['ret']
    cap_stats = results['cap']
    valid_stats = results['valid']
    
    # 生成统计信息
    print("\n" + "=" * 60)
    print("数据统计信息:")
    print(f"收益率数据: {num_timepoints} x {num_stocks}")
    print(f"  均值: {ret_stats['mean']:.6f}")
    print(f"  标准差: {ret_stats['std']:.6f}")
    print(f"  最小值: {ret_stats['min']:.6f}")
    print(f"  最大值: {ret_stats['max']:.6f}")
    print(f"  生成用时: {ret_stats['gen_time']:.1f}秒, 写入用时: {ret_stats['write_time']:.1f}秒")
    
    print(f"\n市值数据: {num_timepoints} x {num_stocks}")
    print(f"  均值: {cap_stats['mean']:.2e}")
    print(f"  标准差: {cap_stats['std']:.2e}")
    print(f"  最小值: {cap_stats['min']:.2e}")
    print(f"  最大值: {cap_stats['max']:.2e}")
    print(f"  生成用时: {cap_stats['gen_time']:.1f}秒, 写入用时: {cap_stats['write_time']:.1f}秒")
    
    print(f"\n有效性数据: {num_timepoints} x {num_stocks}")
    print(f"  均值: {valid_stats['mean']:.6f}")
    print(f"  有效值总数: {valid_stats['sum']:.0f}")
    print(f"  无效值总数: {valid_stats['total'] - valid_stats['sum']:.0f}")
    print(f"  生成用时: {valid_stats['gen_time']:.1f}秒, 写入用时: {valid_stats['write_time']:.1f}秒")
    
    # 检查文件大小
    ret_file_size = os.path.getsize('input_ret.csv') / (1024 * 1024)  # MB
    cap_file_size = os.path.getsize('input_cap.csv') / (1024 * 1024)  # MB
    valid_file_size = os.path.getsize('input_valid.csv') / (1024 * 1024)  # MB
    
    print("\n" + "=" * 60)
    print("文件大小:")
    print(f"input_ret.csv: {ret_file_size:.2f} MB")
    print(f"input_cap.csv: {cap_file_size:.2f} MB")
    print(f"input_valid.csv: {valid_file_size:.2f} MB")
    
    total_time = time.time() - total_start
    print(f"\n总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print("5001测试案例数据生成完成！")

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    generate_5001_test_data()


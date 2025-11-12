#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成5002_cne5_1_beta1 测试案例的输入数据（更贴近市场、波动更剧烈）

特性：
- 收益率具备市场因子 + 个股噪声结构，带有状态切换的波动聚集（高/低波动 regime）
- 使用重尾分布（Student-t）与罕见跳跃，增强极端波动场景
- 市值为对数正态分布并随收益演化（动量与噪声驱动），截面分布长尾
- 有效性标记包含“停牌段落”式的时间聚集失效
"""

import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

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

def _student_t(df, size, rng):
    # 生成标准Student-t并标准化为单位方差（便于用sigma缩放）
    x = rng.standard_t(df, size=size)
    # t分布方差 = df/(df-2)（df>2），据此做单位方差化
    scale = np.sqrt(df / (df - 2.0))
    return x / scale

def generate_and_write_ret(num_timepoints, num_stocks, seed_seq=None, ready_event: Event | None = None):
    """生成并写入收益率数据（百分比单位：例如1.0代表1%）"""
    if seed_seq is None:
        seed_seq = np.random.SeedSequence()
    rng = np.random.default_rng(seed_seq)
    print(f"[线程1] 开始生成收益率数据...")
    gen_start = time.time()

    # 市场波动状态（Markov两状态：低/高波动）
    p_stay_low, p_stay_high = 0.97, 0.95
    sigma_low, sigma_high = 0.005, 0.025  # 对应 0.5% 与 2.5% 日波动（以绝对值返回率计）
    df_t = 5  # 重尾

    # 生成状态序列
    regime = np.zeros(num_timepoints, dtype=np.int32)
    for t in range(1, num_timepoints):
        if regime[t-1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    # 市场因子收益（绝对数，如0.01=1%）
    market_sigma_t = np.where(regime == 0, sigma_low, sigma_high)
    market_factor = _student_t(df_t, size=num_timepoints, rng=rng) * market_sigma_t

    # 行业/风格小因子（弱相关噪声）
    num_small_factors = 3
    small_factors = _student_t(df_t, size=(num_small_factors, num_timepoints), rng=rng) * (market_sigma_t * 0.3)
    loadings = rng.normal(0.0, 0.5, size=(num_stocks, num_small_factors))

    # 个股beta与特异性波动
    beta = rng.normal(1.0, 0.3, size=num_stocks)
    idio_vol = np.exp(rng.normal(np.log(0.01), 0.5, size=num_stocks))  # 中位数~1%

    # 生成个股收益
    returns_abs = np.empty((num_timepoints, num_stocks), dtype=np.float64)
    for i in range(num_stocks):
        idio = _student_t(df_t, size=num_timepoints, rng=rng) * idio_vol[i]
        small = (loadings[i, :][..., None] * small_factors).sum(axis=0)
        base = beta[i] * market_factor + small + idio
        # 罕见跳跃
        jumps = (rng.random(num_timepoints) < 0.01).astype(np.float64) * rng.normal(0.0, 0.08, size=num_timepoints)
        x = base + jumps
        # 限幅（交易所涨跌停近似，留给后续winsor再处理）
        x = np.clip(x, -0.25, 0.25)
        returns_abs[:, i] = x

    # 转为百分比（与C++侧除以100对应）
    returns_pct = returns_abs * 100.0

    gen_time = time.time() - gen_start
    print(f"[线程1] 数据生成完成，用时: {gen_time:.1f}秒")

    write_time = 0.0
    try:
        write_time = write_array_to_csv_fast(returns_pct, 'input_ret.csv', chunk_size=50000)
    finally:
        if ready_event is not None:
            ready_event.set()
    stats = {
        'mean': float(np.mean(returns_abs)),
        'std': float(np.std(returns_abs)),
        'min': float(np.min(returns_abs)),
        'max': float(np.max(returns_abs)),
        'gen_time': gen_time,
        'write_time': write_time
    }
    del returns_abs, returns_pct
    print(f"[线程1] 收益率数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_and_write_cap(num_timepoints, num_stocks, seed_seq=None, ready_event: Event | None = None):
    """生成并写入市值数据（随收益演化，截面长尾）"""
    if seed_seq is None:
        seed_seq = np.random.SeedSequence()
    rng = np.random.default_rng(seed_seq)
    print(f"[线程2] 开始生成市值数据...")
    gen_start = time.time()

    # 初始截面：对数正态（单位：任意）
    init_log_cap = rng.normal(11.0, 1.2, size=(1, num_stocks))
    cap = np.exp(init_log_cap)

    # 读取刚生成的收益（百分比），用于演化
    if ready_event is not None:
        ready_event.wait()
    ret = np.loadtxt('input_ret.csv', delimiter=',')  # (T,N)，百分比
    ret_abs = ret / 100.0

    # 时间演化：
    # logCap_t = logCap_{t-1} + mu - 0.5*sigma^2 + alpha * ret_t + noise
    mu, sigma_e, alpha = 0.0002, 0.02, 0.2
    log_cap_series = np.empty_like(ret_abs)
    log_cap_series[0:1, :] = np.log(cap)
    for t in range(1, ret_abs.shape[0]):
        noise = rng.normal(0.0, sigma_e, size=(num_stocks,))
        log_cap_series[t, :] = (
            log_cap_series[t-1, :] + mu - 0.5 * sigma_e * sigma_e + alpha * ret_abs[t, :] + noise
        )

    cap_path = np.exp(log_cap_series)
    gen_time = time.time() - gen_start
    print(f"[线程2] 数据生成完成，用时: {gen_time:.1f}秒")

    write_time = write_array_to_csv_fast(cap_path, 'input_cap.csv', chunk_size=50000)
    stats = {
        'mean': float(np.mean(cap_path)),
        'std': float(np.std(cap_path)),
        'min': float(np.min(cap_path)),
        'max': float(np.max(cap_path)),
        'gen_time': gen_time,
        'write_time': write_time
    }
    del ret, ret_abs, cap_path, log_cap_series
    print(f"[线程2] 市值数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_and_write_valid(num_timepoints, num_stocks, seed_seq=None):
    """生成并写入有效性数据（含停牌段落式聚集失效）"""
    if seed_seq is None:
        seed_seq = np.random.SeedSequence()
    rng = np.random.default_rng(seed_seq)
    print(f"[线程3] 开始生成有效性数据...")
    gen_start = time.time()
    base_prob = 0.93
    valid = rng.binomial(1, base_prob, size=(num_timepoints, num_stocks)).astype(float)

    # 叠加停牌段落：为每只股票随机生成若干个失效段
    max_segments = 3
    for j in range(num_stocks):
        seg_count = rng.integers(0, max_segments + 1)
        for _ in range(seg_count):
            start = int(rng.integers(0, max(1, num_timepoints - 1)))
            length = int(np.clip(rng.normal(10, 8), 3, num_timepoints//6))
            end = min(num_timepoints, start + length)
            valid[start:end, j] = 0.0

    gen_time = time.time() - gen_start
    print(f"[线程3] 数据生成完成，用时: {gen_time:.1f}秒")

    write_time = write_array_to_csv_fast(valid, 'input_valid.csv', chunk_size=50000)
    stats = {
        'mean': float(np.mean(valid)),
        'sum': float(np.sum(valid)),
        'total': int(valid.size),
        'gen_time': gen_time,
        'write_time': write_time
    }
    del valid
    print(f"[线程3] 有效性数据完成，总用时: {gen_time + write_time:.1f}秒")
    return stats

def generate_5002_test_data():
    """生成5002测试案例的输入数据（多线程并行版本）"""
    
    total_start = time.time()
    
    # 可按需调整规模（注意生成时间与文件大小）
    num_stocks = 4
    num_timepoints = 8
    
    print(f"正在生成 {num_timepoints} 个时间点 x {num_stocks} 只股票的数据...")
    print("使用多线程并行生成...")
    print("=" * 60)
    
    # 使用线程池并行生成和写入三个文件
    results = {}
    ret_ready_event = Event()
    base_entropy = (
        int(time.time_ns()),
        os.getpid(),
        int.from_bytes(os.urandom(8), 'little')
    )
    # 基于时间、进程ID与操作系统熵混合生成主种子，保证每次运行均不同
    entropy = [value & 0xFFFFFFFF for value in base_entropy]
    master_seed = np.random.SeedSequence(entropy)
    ret_seed, cap_seed, valid_seed = master_seed.spawn(3)
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 提交三个任务
        future_ret = executor.submit(generate_and_write_ret, num_timepoints, num_stocks, ret_seed, ret_ready_event)
        future_cap = executor.submit(generate_and_write_cap, num_timepoints, num_stocks, cap_seed, ret_ready_event)
        future_valid = executor.submit(generate_and_write_valid, num_timepoints, num_stocks, valid_seed)
        
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
    print("5002测试案例数据生成完成！")

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    generate_5002_test_data()


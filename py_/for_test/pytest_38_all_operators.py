import numpy as np
import pandas as pd
from copy import deepcopy
import os
import time
from datetime import datetime

# 从m_vpc_mut_ty_log38.py中提取的所有算子函数，完全不变

def get_current_timestamp():
    """获取当前时间戳，格式与C++的getCurrentTimestamp()一致"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def get_mean(df):
    return pd.Series(np.nanmean(df,axis=0),index = df.columns)

def get_std(df):
    return pd.Series(np.nanstd(df,axis=0),index = df.columns)

def get_kurt(df):
    return df.kurt(axis=0)

def get_skew(df):
    return df.skew(axis=0)

#变异系数倒数
def get_ms(df):
    avg = pd.Series(np.nanmean(df,axis=0),index = df.columns)
    std = pd.Series(np.nanstd(df,axis=0),index = df.columns)
    cv = pd.Series(std.values/avg.values, index = avg.index)
    std[pd.Series(cv.values<0.00001,index = cv.index)] = np.nan
    ms = pd.Series(avg.values/std.values, index = avg.index)
    return ms

#半段自相关系数
def get_scm(df):
    def get_corresponding_corr(x_df, y_df):
        x_df.dropna(how='all', inplace=True)
        y_df.dropna(how='all', inplace=True)
        common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
        x_df = x_df.reindex(common_idx)
        y_df = y_df.reindex(common_idx)
        common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
        x_df = x_df[common_columns]
        y_df = y_df[common_columns]

        subdf1_array = x_df.values
        subdf2_array = y_df.values
        subcorr = np.nanmean(
            (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (
                        subdf2_array - np.nanmean(subdf2_array, axis=0)),
            axis=0) / (np.nanstd(subdf1_array, axis=0) * np.nanstd(subdf2_array, axis=0))
        subcorr = pd.Series(subcorr, index=x_df.columns)
        return subcorr
    half_length = df.shape[0]//2;
    idx_fh = df.index[:half_length];
    idx_lh = df.index[half_length:];
    df_fh = df.loc[idx_fh,:].reset_index(drop=True);
    df_lh = df.loc[idx_lh,:].reset_index(drop=True);
    self_corr = get_corresponding_corr(df_fh,df_lh)
    scm = self_corr
    return scm

#时间Beta系数
def get_tb(df):
    # 由于解释变量x一样，所以用协方差替代
    def get_corresponding_cov(x_df, y_df):
        x_df.dropna(how='all', inplace=True)
        y_df.dropna(how='all', inplace=True)
        common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
        x_df = x_df.reindex(common_idx)
        y_df = y_df.reindex(common_idx)
        common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
        x_df = x_df[common_columns]
        y_df = y_df[common_columns]

        subdf1_array = x_df.values
        subdf2_array = y_df.values
        subcov = np.nanmean(
            (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (subdf2_array - np.nanmean(subdf2_array, axis=0)),
            axis=0)
        subcov = pd.Series(subcov, index=x_df.columns)
        return subcov
    time_idx = deepcopy(df)
    time_idx[:] = np.tile(np.array(range(time_idx.shape[0])),(time_idx.shape[1],1)).T
    time_cov = get_corresponding_cov(df,time_idx)
    return time_cov

def get_min(df):
    return pd.Series(np.nanmin(df, axis=0),index = df.columns)

def get_max(df):
    return pd.Series(np.nanmax(df, axis=0),index = df.columns)

#差分均值
def get_dm(df):
    df_delta = df.values - df.shift(1).values
    return pd.Series(np.nanmean(df_delta,axis=0),index = df.columns)

#百分位排序半段自相关系数
def get_srcm(df):
    def get_corresponding_corr(x_df, y_df):
        x_df.dropna(how='all', inplace=True)
        y_df.dropna(how='all', inplace=True)
        common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
        x_df = x_df.reindex(common_idx)
        y_df = y_df.reindex(common_idx)
        common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
        x_df = x_df[common_columns]
        y_df = y_df[common_columns]

        subdf1_array = x_df.values
        subdf2_array = y_df.values
        subcorr = np.nanmean(
            (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (
                        subdf2_array - np.nanmean(subdf2_array, axis=0)),
            axis=0) / (np.nanstd(subdf1_array, axis=0) * np.nanstd(subdf2_array, axis=0))
        subcorr = pd.Series(subcorr, index=x_df.columns)
        return subcorr
    df = df.rank(axis=1, pct=True)
    half_length = df.shape[0]//2;
    idx_fh = df.index[:half_length];
    idx_lh = df.index[half_length:];
    df_fh = df.loc[idx_fh,:].reset_index(drop=True);
    df_lh = df.loc[idx_lh,:].reset_index(drop=True);
    self_corr = get_corresponding_corr(df_fh,df_lh)
    return self_corr

def get_ols(df, fit_intercept=True):
    """
    最小二乘算法 (OLS - Ordinary Least Squares)
    从m_vpc_mut_ty_log829.py中提取的算法实现
    输入: 
        df: DataFrame，其中每行是一个样本，每列是一个特征
        fit_intercept: bool，是否包含截距项，默认为True
    输出: 每列的回归系数
    """
    # 检查是否有对应的y数据文件
    import os
    csv_file_path = '../../src/factor_case/testcase/0014_ols/x_input.csv'
    y_file_path = '../../src/factor_case/testcase/0014_ols/y_input.csv'
    
    if os.path.exists(y_file_path):
        # 读取y数据
        y_df = pd.read_csv(y_file_path, header=None)
        y = y_df.iloc[:, 0].values  # 取第一列作为目标变量
        
        # 使用df作为特征矩阵X，（去掉最后一列的NaN）
        X = df.iloc[:,:].values  # 掉最后一列的NaN
        
        # 根据fit_intercept参数决定是否添加截距项
        if fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_with_intercept = np.hstack([ones, X])
        else:
            X_with_intercept = X
        
        # 处理缺失值
        valid_mask = ~(np.isnan(X_with_intercept).any(axis=1) | np.isnan(y))
        X_valid = X_with_intercept[valid_mask]
        y_valid = y[valid_mask]
        
        if X_valid.shape[0] == 0 or X_valid.shape[0] < X_valid.shape[1]:
            # 如果有效样本数不足，返回NaN
            return pd.Series([np.nan] * df.shape[1], index=df.columns)
        
        try:
            # 最小二乘算法核心：b = (X^T * X)^(-1) * X^T * y
            # 这里使用numpy的线性代数求解
            XTX = X_valid.T.dot(X_valid)
            
            # 检查矩阵是否可逆
            if np.linalg.det(XTX) == 0:
                return pd.Series([np.nan] * df.shape[1], index=df.columns)
            
            XTX_inv = np.linalg.inv(XTX)
            b = XTX_inv.dot(X_valid.T).dot(y_valid)
            
            # 返回回归系数，根据是否包含截距项调整结果组织方式
            result_values = [np.nan] * df.shape[1]
            
            if fit_intercept:
                # 包含截距项：截距项对应第0个系数，特征系数对应第1到n个系数
                # 将特征系数放入结果中
                for i in range(1, min(len(b), df.shape[1])):
                    result_values[i-1] = b[i]  # 特征系数
                # 将截距项放在最后一列
                if len(b) > 0 and df.shape[1] > 0:
                    result_values[-1] = b[0]  # 截距项
            else:
                # 不包含截距项：所有系数都是特征系数
                for i in range(min(len(b), df.shape[1])):
                    result_values[i] = b[i]  # 特征系数
            
            return pd.Series(result_values, index=df.columns)
            
        except (np.linalg.LinAlgError, ValueError):
            # 如果线性代数计算失败，返回NaN
            return pd.Series([np.nan] * df.shape[1], index=df.columns)
    else:
        # 如果没有y数据文件，假设最后一列是目标变量
        if df.shape[1] < 2:
            return pd.Series([np.nan] * df.shape[1], index=df.columns)
        
        # 分离特征矩阵X和目标变量y
        X = df.iloc[:, :-1].values  # 除了最后一列的所有列作为特征
        y = df.iloc[:, -1].values   # 最后一列作为目标变量
        
        # 根据fit_intercept参数决定是否添加截距项
        if fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_with_intercept = np.hstack([ones, X])
        else:
            X_with_intercept = X
        
        # 处理缺失值
        valid_mask = ~(np.isnan(X_with_intercept).any(axis=1) | np.isnan(y))
        X_valid = X_with_intercept[valid_mask]
        y_valid = y[valid_mask]
        
        if X_valid.shape[0] == 0 or X_valid.shape[0] < X_valid.shape[1]:
            # 如果有效样本数不足，返回NaN
            return pd.Series([np.nan] * df.shape[1], index=df.columns)
        
        try:
            # 最小二乘算法核心：b = (X^T * X)^(-1) * X^T * y
            # 这里使用numpy的线性代数求解
            XTX = X_valid.T.dot(X_valid)
            
            # 检查矩阵是否可逆
            if np.linalg.det(XTX) == 0:
                return pd.Series([np.nan] * df.shape[1], index=df.columns)
            
            XTX_inv = np.linalg.inv(XTX)
            b = XTX_inv.dot(X_valid.T).dot(y_valid)
            
            # 返回回归系数，根据是否包含截距项调整结果组织方式
            result_values = [np.nan] * df.shape[1]
            
            if fit_intercept:
                # 包含截距项：截距项对应第0个系数，特征系数对应第1到n个系数
                result_values[-1] = b[0]  # 截距项放在最后一列
                for i in range(1, min(len(b), df.shape[1])):
                    result_values[i-1] = b[i]  # 特征系数
            else:
                # 不包含截距项：所有系数都是特征系数
                for i in range(min(len(b), df.shape[1])):
                    result_values[i] = b[i]  # 特征系数
            
            return pd.Series(result_values, index=df.columns)
            
        except (np.linalg.LinAlgError, ValueError):
            # 如果线性代数计算失败，返回NaN
            return pd.Series([np.nan] * df.shape[1], index=df.columns)

#百分位排序时间Beta系数
def get_trb(df):
    def get_corresponding_cov(x_df, y_df):
        x_df.dropna(how='all', inplace=True)
        y_df.dropna(how='all', inplace=True)
        common_idx = sorted(list(set(x_df.index).intersection(set(y_df.index))))
        x_df = x_df.reindex(common_idx)
        y_df = y_df.reindex(common_idx)
        common_columns = sorted(list(set(x_df.columns).intersection(set(y_df.columns))))
        x_df = x_df[common_columns]
        y_df = y_df[common_columns]

        subdf1_array = x_df.values
        subdf2_array = y_df.values
        subcov = np.nanmean(
            (subdf1_array - np.nanmean(subdf1_array, axis=0)) * (subdf2_array - np.nanmean(subdf2_array, axis=0)),
            axis=0)
        subcov = pd.Series(subcov, index=x_df.columns)
        return subcov
    
    print("=== get_trb() 函数执行过程 ===")
    print("原始数据矩阵:")
    print(df)
    print()
    
    df = df.rank(axis=1, pct=True)
    print("行排序后的矩阵 (百分位排名):")
    print(df)
    print()
    
    time_idx = deepcopy(df)
    time_idx[:] = np.tile(np.array(range(time_idx.shape[0])),(time_idx.shape[1],1)).T
    print("时间索引矩阵:")
    print(time_idx)
    print()
    
    time_cov = get_corresponding_cov(df,time_idx)
    print("时间协方差结果:")
    print(time_cov)
    print("=== get_trb() 函数执行完成 ===")
    print()
    
    return time_cov

#差分均值百分位排序
def get_rdm(df):
    df = df.rank(axis=1, pct=True)
    df_delta = df.values - df.shift(1).values
    return pd.Series(np.nanmean(df_delta,axis=0),index = df.columns)

def test_all_operators():
    """
    测试38.py中的所有算子函数，使用对应的测试数据
    包含详细的耗时统计
    """
    print("=== 38.py 所有算子函数测试 ===")
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 定义所有算子及其对应的测试数据
    operators = {
        'mean': get_mean,
        'std': get_std,
        'kurt': get_kurt,
        'skew': get_skew,
        'ms': get_ms,
        'scm': get_scm,
        'tb': get_tb,
        'min': get_min,
        'max': get_max,
        'dm': get_dm,
        'srcm': get_srcm,
        'trb': get_trb,
        'rdm': get_rdm,
        'ols': get_ols
    }
    
    # 对应的测试数据文件，每个算子使用对应的C++测试案例输入数据
    test_data_files = {
        'mean': '../../src/factor_case/testcase/0004_mean/input.csv',
        'std': '../../src/factor_case/testcase/0006_std/input.csv',
        'kurt': '../../src/factor_case/testcase/0002_kurt/input.csv',
        'skew': '../../src/factor_case/testcase/0001_skew/input.csv',
        'ms': '../../src/factor_case/testcase/0008_ms/input.csv',
        'scm': '../../src/factor_case/testcase/0007_scm/input.csv',
        'tb': '../../src/factor_case/testcase/0009_tb/input.csv',
        'min': '../../src/factor_case/testcase/0005_min/input.csv',
        'max': '../../src/factor_case/testcase/0003_max/input.csv',
        'dm': '../../src/factor_case/testcase/0010_dm/input.csv',
        'srcm': '../../src/factor_case/testcase/0011_srcm/input.csv',
        'trb': '../../src/factor_case/testcase/0012_trb/input.csv',
        'rdm': '../../src/factor_case/testcase/0013_rdm/input.csv',
        'ols': '../../src/factor_case/testcase/0014_ols/x_input.csv'
    }
    
    results = {}
    timing_results = {}  # 存储每个算子的耗时
    
    for op_name, op_func in operators.items():
        if op_name in test_data_files:
            csv_file_path = test_data_files[op_name]
            
            print(f"\n=== 测试 {op_name.upper()} 算子 ===")
            
            # 记录单个算子开始时间
            op_start_time = time.time()
            
            # 读取CSV文件
            read_start_time = time.time()
            df = pd.read_csv(csv_file_path, header=None)
            read_time = time.time() - read_start_time
            
            print(f"输入数据维度: {df.shape[0]} x {df.shape[1]}")
            print(f"数据读取耗时: {read_time:.4f} 秒")
            
            # 执行算子计算
            try:
                compute_start_time = time.time()
                result = op_func(df)
                compute_time = time.time() - compute_start_time
                
                print(f"各列{op_name}结果:")
                for i in range(len(result)):
                    value = result.iloc[i] if not pd.isna(result.iloc[i]) else "nan"
                    print(f"col_{i}: {value}")
                
                # 保存结果
                save_start_time = time.time()
                results[op_name] = result
                
                # 写出到对应的输出文件，格式与C++完全一致
                output_dir = os.path.dirname(csv_file_path)
                output_path = os.path.join(output_dir, "py_output.csv")
                with open(output_path, "w", encoding="utf-8") as f:
                    # 第一行写入时间戳，格式与C++一致
                    f.write(f"# Generated at: {get_current_timestamp()}\n")
                    f.write("factor,value\n")
                    for i in range(len(result)):
                        value = result.iloc[i]
                        if pd.isna(value):
                            f.write(f"col_{i}_{op_name},nan\n")
                        else:
                            f.write(f"col_{i}_{op_name},{value:.6f}\n")
                save_time = time.time() - save_start_time
                
                # 记录总耗时
                total_op_time = time.time() - op_start_time
                timing_results[op_name] = {
                    'total': total_op_time,
                    'read': read_time,
                    'compute': compute_time,
                    'save': save_time
                }
                
                print(f"计算耗时: {compute_time:.4f} 秒")
                print(f"保存耗时: {save_time:.4f} 秒")
                print(f"总耗时: {total_op_time:.4f} 秒")
                print(f"结果已写入: {output_path}")
                
            except Exception as e:
                total_op_time = time.time() - op_start_time
                timing_results[op_name] = {
                    'total': total_op_time,
                    'read': read_time,
                    'compute': float('inf'),
                    'save': 0
                }
                print(f"计算 {op_name} 时出错: {e}")
                print(f"失败前耗时: {total_op_time:.4f} 秒")
                results[op_name] = None
    
    # 记录总体结束时间
    total_end_time = time.time()
    total_test_time = total_end_time - total_start_time
    
    # 输出总结
    print(f"\n=== 测试总结 ===")
    print(f"总体测试耗时: {total_test_time:.4f} 秒")
    print()
    
    # 按耗时排序显示结果
    sorted_timing = sorted(timing_results.items(), key=lambda x: x[1]['total'])
    
    print("=== 各算子耗时统计 ===")
    print(f"{'算子名称':<8} {'总耗时(秒)':<12} {'读取(秒)':<10} {'计算(秒)':<10} {'保存(秒)':<10} {'状态':<6}")
    print("-" * 70)
    
    for op_name, timing in sorted_timing:
        status = "成功" if results[op_name] is not None else "失败"
        compute_time_str = f"{timing['compute']:.4f}" if timing['compute'] != float('inf') else "失败"
        print(f"{op_name:<8} {timing['total']:<12.4f} {timing['read']:<10.4f} {compute_time_str:<10} {timing['save']:<10.4f} {status:<6}")
    
    print()
    print("=== 详细结果统计 ===")
    for op_name, result in results.items():
        if result is not None:
            print(f"{op_name}: 成功 - {len(result)} 个结果")
        else:
            print(f"{op_name}: 失败")

def test_single_operator(operator_name):
    """
    测试单个算子函数
    包含详细的耗时统计
    """
    print(f"=== 单独测试 {operator_name.upper()} 算子 ===")
    
    # 记录总体开始时间
    total_start_time = time.time()
    
    # 定义所有算子及其对应的测试数据
    operators = {
        'mean': get_mean,
        'std': get_std,
        'kurt': get_kurt,
        'skew': get_skew,
        'ms': get_ms,
        'scm': get_scm,
        'tb': get_tb,
        'min': get_min,
        'max': get_max,
        'dm': get_dm,
        'srcm': get_srcm,
        'trb': get_trb,
        'rdm': get_rdm,
        'ols': get_ols
    }
    
    # 对应的测试数据文件
    test_data_files = {
        'mean': '../../src/factor_case/testcase/0004_mean/input.csv',
        'std': '../../src/factor_case/testcase/0006_std/input.csv',
        'kurt': '../../src/factor_case/testcase/0002_kurt/input.csv',
        'skew': '../../src/factor_case/testcase/0001_skew/input.csv',
        'ms': '../../src/factor_case/testcase/0008_ms/input.csv',
        'scm': '../../src/factor_case/testcase/0007_scm/input.csv',
        'tb': '../../src/factor_case/testcase/0009_tb/input.csv',
        'min': '../../src/factor_case/testcase/0005_min/input.csv',
        'max': '../../src/factor_case/testcase/0003_max/input.csv',
        'dm': '../../src/factor_case/testcase/0010_dm/input.csv',
        'srcm': '../../src/factor_case/testcase/0011_srcm/input.csv',
        'trb': '../../src/factor_case/testcase/0012_trb/input.csv',
        'rdm': '../../src/factor_case/testcase/0013_rdm/input.csv',
        'ols': '../../src/factor_case/testcase/0014_ols/x_input.csv'
    }
    
    if operator_name not in operators:
        print(f"错误: 未知的算子名称 '{operator_name}'")
        print(f"可用的算子: {', '.join(operators.keys())}")
        return
    
    if operator_name not in test_data_files:
        print(f"错误: 算子 '{operator_name}' 没有对应的测试数据文件")
        return
    
    csv_file_path = test_data_files[operator_name]
    op_func = operators[operator_name]
    
    # 记录数据读取时间
    read_start_time = time.time()
    df = pd.read_csv(csv_file_path, header=None)
    read_time = time.time() - read_start_time
    
    print(f"输入数据维度: {df.shape[0]} x {df.shape[1]}")
    print(f"使用测试数据: {csv_file_path}")
    print(f"数据读取耗时: {read_time:.4f} 秒")
    
    # 执行算子计算
    try:
        compute_start_time = time.time()
        result = op_func(df)
        compute_time = time.time() - compute_start_time
        
        print(f"\n各列{operator_name}结果:")
        for i in range(len(result)):
            value = result.iloc[i] if not pd.isna(result.iloc[i]) else "nan"
            print(f"col_{i}: {value}")
        
        # 保存结果
        save_start_time = time.time()
        output_dir = os.path.dirname(csv_file_path)
        output_path = os.path.join(output_dir, "py_output.csv")
        with open(output_path, "w", encoding="utf-8") as f:
            # 第一行写入时间戳，格式与C++一致
            f.write(f"# Generated at: {get_current_timestamp()}\n")
            f.write("factor,value\n")
            for i in range(len(result)):
                value = result.iloc[i]
                if pd.isna(value):
                    f.write(f"col_{i}_{operator_name},nan\n")
                else:
                    f.write(f"col_{i}_{operator_name},{value:.6f}\n")
        save_time = time.time() - save_start_time
        
        # 计算总耗时
        total_time = time.time() - total_start_time
        
        print(f"\n=== 耗时统计 ===")
        print(f"数据读取耗时: {read_time:.4f} 秒")
        print(f"计算耗时: {compute_time:.4f} 秒")
        print(f"保存耗时: {save_time:.4f} 秒")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"\n结果已写入: {output_path}")
        
    except Exception as e:
        total_time = time.time() - total_start_time
        print(f"计算 {operator_name} 时出错: {e}")
        print(f"失败前总耗时: {total_time:.4f} 秒")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果提供了参数，执行单个算子测试
        operator_name = sys.argv[1].lower()
        test_single_operator(operator_name)
    else:
        # 如果没有参数，执行所有算子测试
        print("用法:")
        print("  python pytest_38_all_operators.py           # 测试所有算子")
        print("  python pytest_38_all_operators.py trb       # 单独测试trb算子")
        print("  python pytest_38_all_operators.py mean      # 单独测试mean算子")
        print()
        print("可用的算子: mean, std, kurt, skew, ms, scm, tb, min, max, dm, srcm, trb, rdm, ols")
        print()
        
        # 默认执行所有算子测试
        test_all_operators()

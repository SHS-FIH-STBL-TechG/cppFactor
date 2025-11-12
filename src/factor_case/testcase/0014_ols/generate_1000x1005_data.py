#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成1000x1005的测试数据用于0014_ols测试
"""

import numpy as np
import pandas as pd
import os

def generate_1000x1005_test_data():
    """生成1000x1005的测试数据"""
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 生成1000x1005的输入矩阵X
    print("正在生成1000x1005的X矩阵...")
    X = np.random.normal(0, 1, (1005, 1000))
    
    # 生成对应的y向量（1000个样本）
    # 使用线性组合 + 噪声
    true_coefficients = np.random.normal(0, 0.5, 1000)  # 1005个特征对应1005个系数
    y = X @ true_coefficients + np.random.normal(0, 0.1, 1005)  # 线性关系 + 噪声
    
    # 保存X矩阵到CSV文件
    print("正在保存X矩阵...")
    X_df = pd.DataFrame(X)
    X_df.to_csv('x_input.csv', index=False, header=False)
    print(f"X矩阵已保存: {X_df.shape}")
    
    # 保存y向量到CSV文件
    print("正在保存y向量...")
    y_df = pd.DataFrame(y)
    y_df.to_csv('y_input.csv', index=False, header=False)
    print(f"y向量已保存: {y_df.shape}")
    
    # 生成一些统计信息
    print("\n数据统计信息:")
    print(f"X矩阵形状: {X.shape}")
    print(f"X矩阵均值: {np.mean(X):.6f}")
    print(f"X矩阵标准差: {np.std(X):.6f}")
    print(f"X矩阵最小值: {np.min(X):.6f}")
    print(f"X矩阵最大值: {np.max(X):.6f}")
    
    print(f"\ny向量形状: {y.shape}")
    print(f"y向量均值: {np.mean(y):.6f}")
    print(f"y向量标准差: {np.std(y):.6f}")
    print(f"y向量最小值: {np.min(y):.6f}")
    print(f"y向量最大值: {np.max(y):.6f}")
    
    # 检查文件大小
    x_file_size = os.path.getsize('x_input.csv') / (1024 * 1024)  # MB
    y_file_size = os.path.getsize('y_input.csv') / (1024 * 1024)  # MB
    
    print(f"\n文件大小:")
    print(f"x_input.csv: {x_file_size:.2f} MB")
    print(f"y_input.csv: {y_file_size:.2f} MB")
    
    print("\n1000x1005数据生成完成！")

if __name__ == "__main__":
    generate_1000x1005_test_data()

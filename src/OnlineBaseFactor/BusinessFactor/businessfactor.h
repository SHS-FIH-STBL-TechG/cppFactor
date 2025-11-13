#ifndef BUSINESS_FACTOR_H
#define BUSINESS_FACTOR_H

#include "../BaseFactor/BaseFactor.h"
#include "../../DataProcess/DataProcess.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ===========================================
// 业务层算子 (Business Layer Operators)
// ===========================================

// 基于数学基本算法做了贴近业务层修改的算子集合
// 这些算子包含了特定的业务逻辑、数据结构处理或业务场景的优化
class BusinessFactor {
public:
    // ---- 分段处理算子 (Split Processing Operators) ----
    
    // 半段自相关系数算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列半段皮尔逊相关系数向量)
    // 业务特色：将数据分为前后两半，然后计算相关系数
    // 包含：数据分割、索引对齐、NaN处理等业务逻辑
    static void col_split_corr(const MatrixXd& mat, VectorXd& result);
    
    // 时间序列Beta系数算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列时间序列Beta系数向量)
    // 业务特色：计算时间Beta
    // 包含：时间索引构建、对齐索引
    static void col_time_beta(const MatrixXd& mat, VectorXd& result, bool unbiased = true);

    // 半段时间Beta系数算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列半段时间Beta系数向量)
    // 业务特色：将数据分为前后两半，然后计算时间Beta
    // 包含：数据分割、时间索引构建、协方差计算等业务逻辑
    static void col_split_time_beta(const MatrixXd& mat, VectorXd& result);
    
    // ---- 排序预处理算子 (Rank Preprocessing Operators) ----
    
    // 半段排序自相关系数算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列半段排序皮尔逊相关系数向量)
    // 业务特色：先排序，再分段，然后计算相关系数
    // 包含：排序、数据分割、索引对齐等业务逻辑
    static void col_split_rankcorr(const MatrixXd& mat, VectorXd& result);
    
    // 半段排序时间Beta系数算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列半段排序时间Beta系数向量)
    // 业务特色：先排序，再分段，然后计算时间Beta
    // 包含：排序、数据分割、时间索引构建等业务逻辑
    static void col_rank_time_beta(const MatrixXd& mat, VectorXd& result, bool unbiased = true);
    
    // 半段排序差分均值算子
    // 输入：mat(二维矩阵，可能包含NaN值)
    // 输出：result(每列半段排序差分均值向量)
    // 业务特色：先排序，再计算差分均值
    // 包含：排序、差分计算等业务逻辑
    static void col_split_rank_diff_mean(const MatrixXd& mat, VectorXd& result);
    
    // 分段协方差算子
    // 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
    // 输出：result(每列分段协方差向量，将矩阵分为上下两段计算协方差)
    // 业务特色：数据分段处理，计算前后两段的协方差
    // 包含：数据分割、协方差计算等业务逻辑
    static void col_split_cov(const MatrixXd& mat, VectorXd& result, bool unbiased = true);
    
    // 转差分序列
    static void to_diff_sequence(const VectorXd& sequence, VectorXd& result);

    // 转差分收益率序列
    static void to_diff_return_sequence(const VectorXd& sequence, VectorXd& result);
};

#endif // BUSINESS_FACTOR_H

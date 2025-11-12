#include "businessfactor.h"
#include <iostream>
#include <algorithm>
#include <set>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

// ===========================================
// 业务层算子实现 (Business Layer Operators Implementation)
// ===========================================

// 半段自相关系数算子 - 从38.py的get_scm提取
void BusinessFactor::col_split_corr(const MatrixXd& mat, VectorXd& result) {
    // 业务逻辑：数据分割
    int half_length = mat.rows() / 2;
    int cols = mat.cols();
    result.resize(cols);
    
    if (half_length == 0) {
        for (int i = 0; i < cols; ++i) {
            result[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    
    // 拆分为两半矩阵
    MatrixXd first_half = mat.topRows(half_length);
    MatrixXd second_half = mat.bottomRows(mat.rows() - half_length);
    
    // 业务逻辑：索引对齐处理
    // 获取有效行索引
    std::vector<int> valid_rows_first;
    std::vector<int> valid_rows_second;
    
    valid_rows_first = EigenExtra::getValidRowIndices(first_half);
    valid_rows_second = EigenExtra::getValidRowIndices(second_half);
    
    // 业务逻辑：求交集
    std::vector<int> intersection;
    std::set<int> set_second(valid_rows_second.begin(), valid_rows_second.end());
    for (int idx : valid_rows_first) {
        if (set_second.find(idx) != set_second.end()) {
            intersection.push_back(idx);
        }
    }
    
    if (intersection.empty()) {
        for (int i = 0; i < cols; ++i) {
            result[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    
    // 计算每列的相关系数
    #pragma omp parallel for
    for (int col = 0; col < cols; ++col) {
        // 构建交集行的向量
        VectorXd vec1(intersection.size());
        VectorXd vec2(intersection.size());
        
        for (int i = 0; i < static_cast<int>(intersection.size()); ++i) {
            int original_idx = intersection[i];
            vec1(i) = first_half(original_idx, col);
            vec2(i) = second_half(original_idx, col);
        }
        
        // 调用数学层的基础函数
        BaseFactor::pearson_correlation(vec1, vec2, result[col]);
    }
}

// 时间序列beta（解释变量为time index） - 线性回归Beta系数
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：result(每列的时间序列Beta系数向量，表示时间趋势的线性回归斜率)
void BusinessFactor::col_time_beta(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    int rows = mat.rows(), cols = mat.cols();

    result.resize(cols);
    // 业务逻辑：索引对齐处理
    // 获取有效行索引
    std::vector<int> valid_rows_first;
    valid_rows_first = EigenExtra::getValidRowIndices(mat);

    // 计算每列的时间Beta系数
    #pragma omp parallel for
    for (int col = 0; col < cols; ++col) {
        // 构建有效行的向量
        VectorXd vec_data(valid_rows_first.size());
        VectorXd vec_time(valid_rows_first.size());
        
        for (int i = 0; i < static_cast<int>(valid_rows_first.size()); ++i) {
            int original_idx = valid_rows_first[i];
            vec_data(i) = mat(original_idx, col);
            vec_time(i) = original_idx;  // 时间索引
        }
        
        // 计算时间Beta系数（协方差/时间方差）
        BaseFactor::covariance(vec_data, vec_time, result[col], unbiased);
    }
}

// 半段时间Beta系数算子 - 从38.py的get_tb提取
void BusinessFactor::col_split_time_beta(const MatrixXd& mat, VectorXd& result) {
    // 业务逻辑：数据分割
    int half_length = mat.rows() / 2;
    int cols = mat.cols();
    result.resize(cols);
    
    if (half_length == 0) {
        for (int i = 0; i < cols; ++i) {
            result[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    
    // 拆分为两半矩阵
    MatrixXd first_half = mat.topRows(half_length);
    MatrixXd second_half = mat.bottomRows(mat.rows() - half_length);
    
    // 业务逻辑：时间索引构建
    MatrixXd time_idx_first(half_length, cols);
    MatrixXd time_idx_second(mat.rows() - half_length, cols);
    
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < half_length; ++r) {
            time_idx_first(r, c) = r;
        }
        for (int r = 0; r < static_cast<int>(mat.rows() - half_length); ++r) {
            time_idx_second(r, c) = r;
        }
    }
    
    // 业务逻辑：索引对齐处理（简化版本）
    // 计算每列的协方差（时间Beta系数）
    #pragma omp parallel for
    for (int col = 0; col < cols; ++col) {
        VectorXd vec1 = first_half.col(col);
        VectorXd vec2 = time_idx_first.col(col);
        
        // 调用数学层的基础函数
        BaseFactor::pearson_correlation(vec1, vec2, result[col]);
    }
}

// 半段排序自相关系数算子 - 从38.py的get_srcm提取
void BusinessFactor::col_split_rankcorr(const MatrixXd& mat, VectorXd& result) {
    // 业务逻辑：排序预处理
    MatrixXd ranked_mat;
    BaseFactor::row_rankpct(mat, ranked_mat);
    
    // 调用半段自相关系数算子
    col_split_corr(ranked_mat, result);
}

// 半段排序时间Beta系数算子 - 从38.py的get_trb提取
void BusinessFactor::col_rank_time_beta(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    // 业务逻辑：排序预处理
    MatrixXd ranked_mat;
    BaseFactor::row_rankpct(mat, ranked_mat);

    // 调用时间Beta系数算子
    col_time_beta(ranked_mat, result, unbiased);
}

// 半段排序差分均值算子 - 从38.py的get_rdm提取
void BusinessFactor::col_split_rank_diff_mean(const MatrixXd& mat, VectorXd& result) {
    // 业务逻辑：排序预处理
    MatrixXd ranked_mat;
    BaseFactor::row_rankpct(mat, ranked_mat);
    
    // 调用数学层的差分均值函数
    BaseFactor::col_rankdm(ranked_mat, result);
}

// 分段协方差算子 - 协方差计算
// 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(每列分段协方差向量，将矩阵分为上下两段计算协方差)
void BusinessFactor::col_split_cov(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    // 业务逻辑：数据分割
    int hlen = mat.rows() / 2, cols = mat.cols();
    result.resize(cols);
    
    if (hlen == 0) {
        for (int i = 0; i < cols; ++i) {
            result[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return;
    }
    
    MatrixXd x = mat.topRows(hlen);
    MatrixXd y = mat.bottomRows(mat.rows() - hlen);
    
    // 业务逻辑：协方差计算
    #pragma omp parallel for
    for(int c = 0; c < cols; ++c) {
        VectorXd v1 = x.col(c), v2 = y.col(c);
        double mean1, mean2;
        BaseFactor::nanmean(v1, mean1);
        BaseFactor::nanmean(v2, mean2);
        double s12 = 0; 
        int cnt = 0;
        for(int i = 0; i < v1.size(); ++i) {
            if(!std::isnan(v1[i]) && !std::isnan(v2[i])) { 
                s12 += (v1[i] - mean1) * (v2[i] - mean2); 
                ++cnt;
            }
        }
        if (cnt == 0) {
            result[c] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // 无偏估计：除以n-1；有偏估计：除以n
            double divisor = unbiased ? (cnt - 1) : cnt;
            result[c] = s12 / divisor;
        }
    }
}

// 转差分序列
void BusinessFactor::to_diff_sequence(const VectorXd& sequence, VectorXd& result){
    const int len = sequence.size();
    result.resize(len);
    if(len == 0){
        return;
    }
    result[0] = std::numeric_limits<double>::quiet_NaN();
    for(int i = 1; i < len; ++i){
        const double prev = sequence[i - 1];
        const double curr = sequence[i];
        if(std::isnan(prev) || std::isnan(curr)){
            result[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            result[i] = curr - prev;
        }
    }
}
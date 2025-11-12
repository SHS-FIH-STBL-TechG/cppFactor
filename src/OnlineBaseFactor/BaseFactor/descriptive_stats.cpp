#include "basefactor.h"
#include <cmath>
#include <limits>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ===========================================
// 描述性统计方法 (Descriptive Statistics)
// ===========================================

// nan安全均值 - 一阶矩估计
// 输入：v(一维向量，可能包含NaN值) 
// 输出：result(算术均值，当所有值都为NaN时设为NaN)
void BaseFactor::nanmean(const VectorXd& v, double& result) {
    double sum = 0.0; 
    int count = 0;
    for(const auto& val : v) {
        if (!std::isnan(val)) { 
            sum += val; 
            ++count; 
        }
    }
    result = count == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / count;
}

// nan安全标准差 - 二阶矩估计（样本标准差/总体标准差）
// 输入：v(一维向量，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(标准差，当有效值少于2个时设为NaN)
void BaseFactor::nanstd(const VectorXd& v, double& result, bool unbiased) {
    double mean;
    nanmean(v, mean);
    double accumulator = 0; 
    int count = 0;
    for(const auto& val : v) {
        if (!std::isnan(val)) { 
            accumulator += (val - mean) * (val - mean); 
            ++count; 
        }
    }
    if (count <= 1) {
        result = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    // 无偏估计：除以n-1；有偏估计：除以n
    double divisor = unbiased ? (count - 1) : count;
    result = std::sqrt(accumulator / divisor);
}

// nan安全偏度 - 三阶矩估计（无偏/有偏估计）
// 输入：v(一维向量，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(偏度，当有效值少于3个或标准差为0时设为NaN)
void BaseFactor::nanskew(const VectorXd& v, double& result, bool unbiased) {
    double mean;
    nanmean(v, mean);
    double stdDev;
    nanstd(v, stdDev, unbiased);  // 使用相同的无偏/有偏设置
    if (stdDev == 0 || std::isnan(stdDev)) {
        result = std::numeric_limits<double>::quiet_NaN();  // 如果标准差为0或NaN，直接设为NaN
        return;
    }
    
    double skew = 0; 
    int count = 0;
    for (const auto& val : v) {
        if (!std::isnan(val)) { 
            skew += std::pow((val - mean) / stdDev, 3); 
            ++count;
        }
    }
    
    if (count > 2) {
        if (unbiased) {
            // 无偏偏度估计 = (n/((n-1)(n-2))) × Σ((xi-μ)/σ)³
            double unbiasedFactor = (double)count / ((count - 1) * (count - 2));
            skew *= unbiasedFactor;
        }
        // 有偏估计：直接使用原始值，不进行无偏修正
        result = skew;
    } else {
        result = std::numeric_limits<double>::quiet_NaN();  // 需要至少3个数据点
    }
}

// nan安全峰度 - 四阶矩估计（无偏/有偏估计）
// 输入：v(一维向量，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(Fisher峰度，当有效值少于4个或标准差为0时设为NaN)
void BaseFactor::nankurt(const VectorXd& v, double& result, bool unbiased) {
    double mean;
    nanmean(v, mean);
    double stdDev;
    nanstd(v, stdDev, unbiased);  // 使用相同的无偏/有偏设置
    if (stdDev == 0 || std::isnan(stdDev)) {
        result = std::numeric_limits<double>::quiet_NaN();  // 如果标准差为0或NaN，直接设为NaN
        return;
    }
    
    double kurt = 0; 
    int count = 0;
    for(const auto& val : v) {
        if(!std::isnan(val)) {
            kurt += std::pow((val - mean) / stdDev, 4); 
            ++count;
        }
    }
    
    if (count > 3) {
        if (unbiased) {
            // 无偏峰度估计公式
            // 使用 Fisher 峰度（减3）的无偏估计
            double numSamples = count;
            double unbiasedFactor = (numSamples * (numSamples + 1)) / ((numSamples - 1) * (numSamples - 2) * (numSamples - 3));
            double biasCorrection = 3 * (numSamples - 1) * (numSamples - 1) / ((numSamples - 2) * (numSamples - 3));
            kurt = (unbiasedFactor * kurt) - biasCorrection;
        } else {
            // 有偏峰度估计：直接使用原始值减去3（Fisher峰度）
            kurt = kurt - 3.0;
        }
        result = kurt;
    } else {
        result = std::numeric_limits<double>::quiet_NaN();  // 需要至少4个数据点
    }
}

// nan安全最小值 - 极值统计
// 输入：v(一维向量，可能包含NaN值)
// 输出：result(最小值，当所有值都为NaN时设为NaN)
void BaseFactor::nanmin(const VectorXd& v, double& result) {
    double minValue = std::numeric_limits<double>::max(); 
    bool hasValidValue = false;
    for(const auto& val : v) {
        if(!std::isnan(val)) {
            minValue = std::min(minValue, val); 
            hasValidValue = true;
        }
    }
    result = hasValidValue ? minValue : std::numeric_limits<double>::quiet_NaN();
}

// nan安全最大值 - 极值统计
// 输入：v(一维向量，可能包含NaN值)
// 输出：result(最大值，当所有值都为NaN时设为NaN)
void BaseFactor::nanmax(const VectorXd& v, double& result) {
    double maxValue = -std::numeric_limits<double>::max(); 
    bool hasValidValue = false;
    for(const auto& val : v) {
        if(!std::isnan(val)) {
            maxValue = std::max(maxValue, val); 
            hasValidValue = true;
        }
    }
    result = hasValidValue ? maxValue : std::numeric_limits<double>::quiet_NaN();
}

// ===========================================
// 矩阵列统计方法 (Matrix Column Statistics)
// ===========================================

// 按列计算nan安全均值
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：result(每列的均值向量，长度等于矩阵列数)
void BaseFactor::col_nanmean(const MatrixXd& mat, VectorXd& result) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nanmean(mat.col(colIdx), result[colIdx]);
    }
}

// 按列计算nan安全标准差
// 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(每列的标准差向量，长度等于矩阵列数)
void BaseFactor::col_nanstd(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nanstd(mat.col(colIdx), result[colIdx], unbiased);
    }
}

// 按列计算nan安全偏度
// 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(每列的偏度向量，长度等于矩阵列数)
void BaseFactor::col_nanskew(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nanskew(mat.col(colIdx), result[colIdx], unbiased);
    }
}

// 按列计算nan安全峰度
// 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(每列的峰度向量，长度等于矩阵列数)
void BaseFactor::col_nankurt(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nankurt(mat.col(colIdx), result[colIdx], unbiased);
    }
}

// 按列计算nan安全最小值
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：result(每列的最小值向量，长度等于矩阵列数)
void BaseFactor::col_nanmin(const MatrixXd& mat, VectorXd& result) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nanmin(mat.col(colIdx), result[colIdx]);
    }
}

// 按列计算nan安全最大值
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：result(每列的最大值向量，长度等于矩阵列数)
void BaseFactor::col_nanmax(const MatrixXd& mat, VectorXd& result) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        nanmax(mat.col(colIdx), result[colIdx]);
    }
}

// 均值/标准差(变异系数倒数) - 变异系数分析
// 输入：mat(二维矩阵，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(每列的变异系数倒数向量，当变异系数接近0时设为NaN)
void BaseFactor::col_mean_over_std(const MatrixXd& mat, VectorXd& result, bool unbiased) {
    VectorXd mean, stdDev;
    col_nanmean(mat, mean);
    col_nanstd(mat, stdDev, unbiased);
    VectorXd coefficientOfVariation = stdDev.cwiseQuotient(mean.array().abs().matrix());
    for(int i = 0; i < coefficientOfVariation.size(); ++i) {
        if (std::abs(coefficientOfVariation[i]) < 0.00001) {
            stdDev[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
    result = mean.cwiseQuotient(stdDev);
}

// 差分均值 - 时间序列差分分析
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：result(每列的相邻元素差值均值向量，长度等于矩阵列数)
void BaseFactor::col_meandiff(const MatrixXd& mat, VectorXd& result) {
    result.resize(mat.cols());
    #pragma omp parallel for
    for(int colIdx = 0; colIdx < mat.cols(); ++colIdx) {
        VectorXd column = mat.col(colIdx);
        int numElements = column.size(); 
        double sum = 0; 
        int count = 0;
        for(int i = 1; i < numElements; ++i) {
            if (!std::isnan(column[i]) && !std::isnan(column[i-1])) {
                sum += column[i] - column[i-1]; 
                ++count;
            }
        }
        result[colIdx] = count == 0 ? std::numeric_limits<double>::quiet_NaN() : (sum / count);
    }
}

// ===========================================
// 权重计算方法 (Weight Calculation Methods)
// ===========================================

// 计算指数衰减权重系数（参考 pandas ewm(halflife)）
// 输入：length(权重序列长度), halflife(半衰期)
// 输出：result(权重系数向量，result[0]对应最旧数据，result[length-1]对应最新数据，权重从旧到新递增)
// 说明：
//  - halflife: 半衰期，表示权重衰减到一半所需的时间步数
//  - 权重公式：w[i] = alpha * (1-alpha)^(length-1-i)，其中 alpha = 1 - exp(-ln(2)/halflife)
//  - 权重已归一化，使得 sum(result) = 1.0
void BaseFactor::ewm_weights(size_t length, double halflife, VectorXd& result) {
    if (length == 0) {
        result.resize(0);
        return;
    }
    if (halflife <= 0.0) {
        result = VectorXd::Ones(length) / static_cast<double>(length);
        return;
    }

    result.resize(length);

    const double log2 = std::log(2.0);
    const double alpha = 1.0 - std::exp(-log2 / halflife);
    const double one_minus_alpha = 1.0 - alpha;

    for (size_t i = 0; i < length; ++i) {
        // i=0 最旧，i=length-1 最新（权重最大）
        result[i] = std::pow(one_minus_alpha, length - 1 - i);
    }

    // 让 result[length-1] = 1.0
    result /= result[length - 1];
}


// 计算权重和
void BaseFactor::ewm_weights_sum(const VectorXd& weights, double& result) {
    
}

// ===========================================
// 带权方差无偏估计系数 (Weight Calculation Methods)
// ===========================================

// 输入：weights(长度为窗口大小的权重序列，任意非负，可未归一)
// 返回：coef = (Σw)^2 / ((Σw)^2 - Σw^2)
// 说明：
//  - 当 weights 已归一化且 Σw=1 时，coef = 1 / (1 - Σw^2)
//  - 当分母<=0（如窗口过小、权重极端集中）或有效样本不足时，返回 1.0（不放大修正）
void BaseFactor::weighted_variance_unbiased_coef(const VectorXd& weights, double& result) {
    if (weights.size() <= 1) {
        result = 1.0;
        return;
    }
    double sumW = 0.0;
    double sumWSquared = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
        const double weight = weights[i];
        if (weight < 0.0 || std::isnan(weight)) {
            continue; // 跳过无效或负权重
        }
        sumW += weight;
        sumWSquared += weight * weight;
    }

    const double sumWSquaredTotal = sumW * sumW;
    const double denominator = sumWSquaredTotal - sumWSquared;
    if (denominator > 0.0) {
        result = sumWSquaredTotal / denominator;
    } else {
        result = 1.0;
    }
}

// 计算带权偏度的无偏修正系数
// 参数:
//   weights = 权重序列（已按EWM或任意分布给出）
// 返回:
//   Fisher–Pearson无偏修正系数（若样本太少则返回1.0）
double BaseFactor::computeWeightedSkewBesselCorrection(const Eigen::VectorXd& weights) noexcept {
    if (weights.size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // 假设权重已经归一化：sum(w) = 1
    const double S2 = weights.squaredNorm();
    const double S3 = (weights.array() * weights.array() * weights.array()).sum();

    // 分母防止极端数值
    const double denom = 1.0 - 3.0 * S2 + 2.0 * S3;
    if (denom <= 0.0) {
        return 1.0;  // 或 NaN
    }

    const double numerator = std::pow(1.0 - S2, 1.5);
    return numerator / denom;
}

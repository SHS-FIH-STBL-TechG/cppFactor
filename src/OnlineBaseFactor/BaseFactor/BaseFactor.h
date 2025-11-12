#pragma once
#include "../../Tool/Tool.h"
#include "../../Eigen_extra/Eigen_extra.h"

using namespace EigenExtra;
using std::vector;
using std::string;
using std::map;

// 因子所需接口与静态函数
class BaseFactor {
public:
    virtual ~BaseFactor() = default;

    // ===========================================
    // 数学方法分类声明 (Mathematical Methods Classification)
    // ===========================================


    // ---- 描述性统计方法 (Descriptive Statistics) ----
    // 基础统计量（nan-safe）
    // 输入：vec(一维向量，可能包含NaN值) 输出：result(统计量，无效时设为NaN)
    static void nanmean(const VectorXd& vec, double& result);        // 均值（一阶矩）
    static void nanstd(const VectorXd& vec, double& result, bool unbiased = true);         // 标准差（二阶矩，默认无偏）
    static void nanskew(const VectorXd& vec, double& result, bool unbiased = true);        // 偏度（三阶矩，默认无偏）
    static void nankurt(const VectorXd& vec, double& result, bool unbiased = true);        // 峰度（四阶矩，默认无偏）
    static void nanmin(const VectorXd& vec, double& result);         // 最小值（极值统计）
    static void nanmax(const VectorXd& vec, double& result);         // 最大值（极值统计）

    // 矩阵列统计
    // 输入：mat(二维矩阵，可能包含NaN值) 输出：result(每列的统计量向量)
    static void col_nanmean(const MatrixXd& mat, VectorXd& result);        // 按列计算均值
    static void col_nanstd(const MatrixXd& mat, VectorXd& result, bool unbiased = true);         // 按列计算标准差（默认无偏）
    static void col_nanskew(const MatrixXd& mat, VectorXd& result, bool unbiased = true);        // 按列计算偏度（默认无偏）
    static void col_nankurt(const MatrixXd& mat, VectorXd& result, bool unbiased = true);        // 按列计算峰度（默认无偏）
    static void col_nanmin(const MatrixXd& mat, VectorXd& result);         // 按列计算最小值
    static void col_nanmax(const MatrixXd& mat, VectorXd& result);         // 按列计算最大值
    static void col_mean_over_std(const MatrixXd& mat, VectorXd& result, bool unbiased = true);  // 变异系数倒数(均值/标准差，默认无偏)
    static void col_meandiff(const MatrixXd& mat, VectorXd& result);       // 差分均值(相邻元素差值均值)

    // ---- 相关性分析方法 (Correlation Analysis) ----
    // 输入：vecX,vecY(两个一维向量) 输出：result(皮尔逊相关系数，无效时设为NaN)
    static void pearson_correlation(const VectorXd& vecX, const VectorXd& vecY, double& result);  // 皮尔逊相关系数
    static void covariance(const VectorXd& vecX, const VectorXd& vecY, double& result, bool unbiased = true);  // 协方差（默认无偏）


    // ---- 非参数统计方法 (Non-parametric Statistics) ----
    // 输入：mat(一维向量) 输出：result(百分位排名向量)
    static void rankpct(const VectorXd& mat, VectorXd& result);        // 百分位排名
    // 输入：mat(二维矩阵) 输出：result(每行百分位排名矩阵)
    static void row_rankpct(const MatrixXd& mat, MatrixXd& result);        // 行百分位排名
    // 输入：mat(二维矩阵) 输出：result(每列分段斯皮尔曼相关系数向量)
    static void col_rankdm(const MatrixXd& mat, VectorXd& result);         // 排序差分均值

    // ---- 回归与最小二乘 (Regression & OLS) ----
    // 最小二乘法求解：给定设计矩阵matX和目标向量vecY，计算系数beta
    // 说明：
    //  - 会自动丢弃含NaN的样本行（matX或vecY任一位置为NaN）
    //  - fit_intercept=true时自动在matX左侧拼接一列1用于截距项
    //  - method参数：0=QR分解，1=SVD分解，2=Ridge正则化
    //  - 输出：系数向量beta，通过引用参数返回结果
    static void ols_solve(const MatrixXd& matX, const VectorXd& vecY, VectorXd& beta, bool fit_intercept = true, int method = 0);

    // ---- 权重计算方法 (Weight Calculation Methods) ----
    // 计算指数衰减权重系数（参考 pandas ewm(halflife)）
    // 输入：length(权重序列长度), halflife(半衰期)
    // 输出：result(权重系数向量，result[0]对应最旧数据，result[length-1]对应最新数据，权重从旧到新递增)
    // 说明：
    //  - halflife: 半衰期，表示权重衰减到一半所需的时间步数
    //  - 权重公式：w[i] = alpha * (1-alpha)^(length-1-i)，其中 alpha = 1 - exp(-ln(2)/halflife)
    //  - 权重已归一化，使得 sum(result) = 1.0
    static void ewm_weights(size_t length, double halflife, VectorXd& result);

    // 计算权重和
    static void ewm_weights_sum(const VectorXd& weights, double& result);

    // 带权方差无偏估计系数（输入为权重序列）
    // 输入：weights(长度为窗口大小的权重序列，任意非负，可未归一)
    // 返回：coef = (Σw)^2 / ((Σw)^2 - Σw^2)，当分母<=0或样本不足时返回1.0
    static void weighted_variance_unbiased_coef(const VectorXd& weights, double& result);

    // 计算带权偏度的无偏修正系数
    static double computeWeightedSkewBesselCorrection(const VectorXd& weights) noexcept;
};

#include "basefactor.h"
#include <cmath>
#include <iostream>
#include "../../Eigen_extra/Eigen_extra.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
// ===========================================
// 相关性分析方法 (Correlation Analysis)
// ===========================================

// 单元化皮尔逊相关系数函数 - 输入两条序列，计算相关系数
// 输入：x,y(两个等长的一维向量，可能包含NaN值)
// 输出：result(皮尔逊相关系数，范围[-1,1]，当无有效数据对或方差为0时设为NaN)
void BaseFactor::pearson_correlation(const VectorXd& x, const VectorXd& y, double& result) {
    if (x.size() != y.size()) {
        cout << "pearson_correlation: x.size() != y.size()" << endl;
        result = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    double meanX, meanY;
    nanmean(x, meanX);
    nanmean(y, meanY);
    double sumXY = 0, sumX2 = 0, sumY2 = 0;
    int count = 0;
    
    for (int i = 0; i < x.size(); ++i) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            sumXY += (x[i] - meanX) * (y[i] - meanY);
            sumX2 += (x[i] - meanX) * (x[i] - meanX);
            sumY2 += (y[i] - meanY) * (y[i] - meanY);
            ++count;
        }
    }
    
    // 皮尔逊相关系数公式: r = Σ(xi-x̄)(yi-ȳ) / √[Σ(xi-x̄)² × Σ(yi-ȳ)²]
    result = (!count || sumX2 == 0 || sumY2 == 0) ? std::numeric_limits<double>::quiet_NaN() : sumXY / std::sqrt(sumX2 * sumY2);
}

// 协方差计算函数 - 输入两条序列，计算协方差值
// 输入：x,y(两个等长的一维向量，可能包含NaN值), unbiased(是否使用无偏估计，默认true)
// 输出：result(协方差值，当无有效数据对时设为NaN)
void BaseFactor::covariance(const VectorXd& x, const VectorXd& y, double& result, bool unbiased) {
    if (x.size() != y.size()) {
        cout << "covariance: x.size() != y.size()" << endl;
        result = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    double meanX, meanY;
    nanmean(x, meanX);
    nanmean(y, meanY);
    double sumXY = 0;
    int count = 0;
    
    for (int i = 0; i < x.size(); ++i) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            sumXY += (x[i] - meanX) * (y[i] - meanY);
            ++count;
        }
    }
    
    if (count == 0) {
        result = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // 无偏估计：除以n-1；有偏估计：除以n
    double divisor = unbiased ? (count - 1) : count;
    result = sumXY / divisor;
}

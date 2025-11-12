#pragma once
#include "../Eigen_extra/Eigen_extra.h"

using namespace EigenExtra;

class DataProcess {
public:
    // Winsor化：将数值裁剪到 [lower, upper]，保留 NaN
    static Ma winsor(const Ma& input, double lower, double upper);
    static Ve winsor(const Ve& input, double lower, double upper);

    // Winsor化（就地裁剪）：将数值裁剪到 [lower, upper]，保留 NaN（尽量减少拷贝与临时）
    static void winsorInplace(Ma& inout, double lower, double upper);
    static void winsorInplace(Ve& inout, double lower, double upper);

    // 分位数（线性插值，忽略 NaN），等价 pandas quantile(..., method='linear')
    static bool quantileLinear(const Ve& values, double quantile, double& outQuantile);

    // 按 [qLow, qHigh] 分位对向量缩尾（NaN 原样保留），返回新向量
    static Ve clipByQuantile(const Ve& input, double qLow, double qHigh);
};


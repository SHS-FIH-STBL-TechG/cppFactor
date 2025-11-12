#include "DataProcess.h"
#include <algorithm>
#include <vector>
#include <cmath>


Ma DataProcess::winsor(const Ma& input, double lower, double upper) {
    Ma output = input;
    // 使用 Eigen 的 array 操作，向量化处理，性能更好
    auto array = output.array();
    auto validMask = array.isFinite(); // 同时检查 NaN 和 Inf
    array = (validMask).select(
        array.cwiseMax(lower).cwiseMin(upper),
        array  // NaN/Inf 保持原样
    );
    return output;
}

Ve DataProcess::winsor(const Ve& input, double lower, double upper) {
    Ve output = input;
    // 使用 Eigen 的 array 操作，向量化处理，性能更好
    auto array = output.array();
    auto validMask = array.isFinite(); // 同时检查 NaN 和 Inf
    array = (validMask).select(
        array.cwiseMax(lower).cwiseMin(upper),
        array  // NaN/Inf 保持原样
    );
    return output;
}

void DataProcess::winsorInplace(Ma& inout, double lower, double upper) {
    // 快速路径：全部为有限值时，避免生成掩码
    if (inout.allFinite()) {
        inout.array() = inout.array().cwiseMax(lower).cwiseMin(upper);
        return;
    }
    // 含 NaN/Inf 时，仅对有效值裁剪
    auto array = inout.array();
    auto validMask = array.isFinite();
    array = (validMask).select(
        array.cwiseMax(lower).cwiseMin(upper),
        array
    );
}

void DataProcess::winsorInplace(Ve& inout, double lower, double upper) {
    if (inout.allFinite()) {
        inout.array() = inout.array().cwiseMax(lower).cwiseMin(upper);
        return;
    }
    auto array = inout.array();
    auto validMask = array.isFinite();
    array = (validMask).select(
        array.cwiseMax(lower).cwiseMin(upper),
        array
    );
}

// 分位数（线性插值，忽略 NaN）
bool DataProcess::quantileLinear(const Ve& values, double quantile, double& outQuantile) {
    // 使用 std::copy_if 可能比手动循环稍快，但差异不大
    std::vector<double> sortedValues;
    sortedValues.reserve(static_cast<size_t>(values.size()));
    std::copy_if(values.data(), values.data() + values.size(), 
                 std::back_inserter(sortedValues),
                 [](double value) { return !std::isnan(value); });
    
    if (sortedValues.empty()) {
        return false;
    }
    std::sort(sortedValues.begin(), sortedValues.end());
    const size_t numValues = sortedValues.size();
    if (quantile <= 0.0) {
        outQuantile = sortedValues.front();
        return true;
    }
    if (quantile >= 1.0) {
        outQuantile = sortedValues.back();
        return true;
    }

    const double position = (numValues - 1) * quantile;
    const size_t index = static_cast<size_t>(std::floor(position));
    const double fraction = position - static_cast<double>(index);
    if (index + 1 >= numValues) {
        outQuantile = sortedValues.back();
        return true;
    }
    outQuantile = ((1.0 - fraction) * sortedValues[index]) + (fraction * sortedValues[index + 1]);
    return true;
}

// 向量缩尾：按指定分位范围裁剪（NaN 原样保留）
// 性能优化：只排序一次，同时计算两个分位数
Ve DataProcess::clipByQuantile(const Ve& input, double qLow, double qHigh) {
    Ve output = input;
    
    // 提取有效值并排序（只排序一次）
    std::vector<double> sortedValues;
    sortedValues.reserve(static_cast<size_t>(input.size()));
    std::copy_if(input.data(), input.data() + input.size(), 
                 std::back_inserter(sortedValues),
                 [](double value) { return !std::isnan(value); });
    
    if (sortedValues.empty()) {
        return output;
    }
    
    std::sort(sortedValues.begin(), sortedValues.end());
    const size_t numValues = sortedValues.size();
    
    // 计算下界分位数
    double lowerBound;
    if (qLow <= 0.0) {
        lowerBound = sortedValues.front();
    } else if (qLow >= 1.0) {
        lowerBound = sortedValues.back();
    } else {
        const double position = (numValues - 1) * qLow;
        const size_t index = static_cast<size_t>(std::floor(position));
        const double fraction = position - static_cast<double>(index);
        if (index + 1 >= numValues) {
            lowerBound = sortedValues.back();
        } else {
            lowerBound = ((1.0 - fraction) * sortedValues[index]) + (fraction * sortedValues[index + 1]);
        }
    }
    
    // 计算上界分位数
    double upperBound;
    if (qHigh <= 0.0) {
        upperBound = sortedValues.front();
    } else if (qHigh >= 1.0) {
        upperBound = sortedValues.back();
    } else {
        const double position = (numValues - 1) * qHigh;
        const size_t index = static_cast<size_t>(std::floor(position));
        const double fraction = position - static_cast<double>(index);
        if (index + 1 >= numValues) {
            upperBound = sortedValues.back();
        } else {
            upperBound = ((1.0 - fraction) * sortedValues[index]) + (fraction * sortedValues[index + 1]);
        }
    }
    
    // 确保边界顺序
    if (lowerBound > upperBound) {
        std::swap(lowerBound, upperBound);
    }
    
    // 使用 Eigen 的 array 操作进行裁剪（向量化，性能更好）
    auto array = output.array();
    auto validMask = array.isFinite();
    array = (validMask).select(
        array.cwiseMax(lowerBound).cwiseMin(upperBound),
        array  // NaN/Inf 保持原样
    );
    
    return output;
}
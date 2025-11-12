#include "OnlineEWMMethod.h"
#include "OnlineUtils.h"
#include "BaseFactor/BaseFactor.h"
#include <numeric>
#include <cmath>
#include <cstddef>
#include <cstdlib>

// OnlineEWMSum 带权加和（EWM加权和）
void OnlineEWMSum::constructor(const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMSum", window.m_onlineDataCache);
    CHECK_NULLPTR("OnlineEWMSum", window.m_onlineWeightCache);
    
    m_window = window;
    
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    const size_t dataWindowSize = m_window.m_onlineDataCache->getWindowSize();
    m_windowSize = dataWindowSize;
    
    // 使用compute函数计算初始带权加和
    const auto& dataValues = m_window.m_onlineDataCache->getValues();
    const auto& weightValues = m_window.m_onlineWeightCache->getNormedValues();
    
    // 权重序列的前windowSize个对应当前窗口的数据（从旧到新）
    // 数据索引0对应权重索引0+windowsize（最旧的数据）
    for (size_t i = 0; i < dataValues.size(); ++i) {
        if(std::isnan(dataValues[i])){continue;}
        m_value += weightValues[i + m_windowSize] * dataValues[i];
    }
    
    m_version = 0;
}

void OnlineEWMSum::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }
    
    // 更新下游组件（缓存层）
    m_window.m_onlineDataCache->update(inValues, version);
    m_window.m_onlineWeightCache->update(version);
    
    // 增量更新：加上新值的加权和，减去旧值的加权和
    const auto& outValues = m_window.m_onlineDataCache->getOutValues();
    const auto& unnormalizedWeightValues = m_window.m_onlineWeightCache->getUnnormalizedValues();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    computeEWMSum(outValues, unnormalizedWeightValues, normedWeightValues, inValues, m_windowSize, m_value);
    
    m_version = version;  // 更新版本号
}

inline void OnlineEWMSum::computeEWMSum(const std::deque<double>& outValues, const Ve& unnormalizedWeightValues, const Ve& normedWeightValues, const Ve& inValues, size_t windowSize, double& Value) noexcept {
    const size_t stepSize = inValues.size();
    Value *= unnormalizedWeightValues[(2*windowSize) - stepSize - 1];
    // 减去旧值的加权和
    for (size_t i = 0; i < outValues.size(); ++i) {
        if(std::isnan(outValues[i])){continue;}
        Value -= normedWeightValues[windowSize - stepSize + i] * outValues[i];
    }
    // 加上新值的加权和（使用权重序列的后windowSize个，对应新窗口位置）
    for (size_t i = 0; i < stepSize; ++i) {
        if(std::isnan(inValues[i])){continue;}
        Value += normedWeightValues[(2*windowSize) -stepSize + i] * inValues[i];
    }
}

// OnlineEWMSumProduct 带权乘积和（EWM加权乘积和）
void OnlineEWMSumProduct::constructor(const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMSumProduct", window.m_onlineDataCacheX);
    CHECK_NULLPTR("OnlineEWMSumProduct", window.m_onlineDataCacheY);
    CHECK_NULLPTR("OnlineEWMSumProduct", window.m_onlineWeightCache);
    
    m_window = window;
    
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    const size_t windowSizeX = m_window.m_onlineDataCacheX->getWindowSize();
    const size_t windowSizeY = m_window.m_onlineDataCacheY->getWindowSize();
    if (windowSizeX != windowSizeY) {
        std::cout << "OnlineEWMSumProduct: X和Y的窗口大小不一致\n";
        exit(1);
    }
    m_windowSize = windowSizeX;
    
    // 使用计算初始带权乘积和
    const auto& dataValuesX = m_window.m_onlineDataCacheX->getValues();
    const auto& dataValuesY = m_window.m_onlineDataCacheY->getValues();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    
    // 权重序列的前windowSize个对应当前窗口的数据（从旧到新）
    // 数据索引0对应权重索引0（最旧的数据）
    for (size_t i = 0; i < dataValuesX.size(); ++i) {
        if(std::isnan(dataValuesX[i]) || std::isnan(dataValuesY[i])){continue;}
        m_value += normedWeightValues[i+m_windowSize] * dataValuesX[i] * dataValuesY[i];
    }
    
    m_version = 0;
}

void OnlineEWMSumProduct::update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }
    
    if (inValuesX.size() != inValuesY.size()) {
        std::cout << "OnlineEWMSumProduct::update: X和Y的向量大小不一致\n";
        exit(1);
    }
    
    // 更新下游组件（缓存层）
    m_window.m_onlineDataCacheX->update(inValuesX, version);
    m_window.m_onlineDataCacheY->update(inValuesY, version);
    
    // 增量更新：加上新值的加权乘积和，减去旧值的加权乘积和
    const auto& outValuesX = m_window.m_onlineDataCacheX->getOutValues();
    const auto& outValuesY = m_window.m_onlineDataCacheY->getOutValues();
    const auto& unnormalizedWeightValues = m_window.m_onlineWeightCache->getUnnormalizedValues();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();

    computeEWMSumProduct(outValuesX, outValuesY, unnormalizedWeightValues, normedWeightValues, inValuesX, inValuesY, m_windowSize, m_value);
    
    m_version = version;  // 更新版本号
}

inline void OnlineEWMSumProduct::computeEWMSumProduct(const std::deque<double>& outValuesX, const std::deque<double>& outValuesY, 
                                                const Ve& unnormalizedWeightValues, const Ve& normedWeightValues,
                                                const Ve& inValuesX, const Ve& inValuesY, size_t windowSize, double& Value) noexcept {
    const size_t stepSize = inValuesX.size();
    Value *= unnormalizedWeightValues[(2*windowSize) - stepSize - 1];
    for (size_t i = 0; i < outValuesX.size(); ++i) {
        if(std::isnan(outValuesX[i]) || std::isnan(outValuesY[i])){continue;}
        Value -= normedWeightValues[windowSize - stepSize + i] * outValuesX[i] * outValuesY[i];
    }
    // 加上新值的加权乘积和（使用权重序列的后windowSize个，对应新窗口位置）
    for (size_t i = 0; i < stepSize; ++i) {
        if(std::isnan(inValuesX[i]) || std::isnan(inValuesY[i])){continue;}
        Value += normedWeightValues[(2*windowSize) - stepSize + i] * inValuesX[i] * inValuesY[i];
    }
}

// OnlineEWMSumProduct 带权乘积和（EWM加权乘积和）
void OnlineEWMSumProduct3::constructor(const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMSumProduct3", window.m_onlineDataCacheX);
    CHECK_NULLPTR("OnlineEWMSumProduct3", window.m_onlineDataCacheY);
    CHECK_NULLPTR("OnlineEWMSumProduct3", window.m_onlineDataCacheZ);
    CHECK_NULLPTR("OnlineEWMSumProduct3 ", window.m_onlineWeightCache);
    
    m_window = window;
    
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    const size_t windowSizeX = m_window.m_onlineDataCacheX->getWindowSize();
    const size_t windowSizeY = m_window.m_onlineDataCacheY->getWindowSize();
    const size_t windowSizeZ = m_window.m_onlineDataCacheZ->getWindowSize();
    if (windowSizeX != windowSizeY || windowSizeX != windowSizeZ) {
        std::cout << "OnlineEWMSumProduct3: X和Y和Z的窗口大小不一致\n";
        exit(1);
    }
    m_windowSize = windowSizeX;
    
    // 使用计算初始带权乘积和
    const auto& dataValuesX = m_window.m_onlineDataCacheX->getValues();
    const auto& dataValuesY = m_window.m_onlineDataCacheY->getValues();
    const auto& dataValuesZ = m_window.m_onlineDataCacheZ->getValues();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    
    // 权重序列的前windowSize个对应当前窗口的数据（从旧到新）
    // 数据索引0对应权重索引0（最旧的数据）
    for (size_t i = 0; i < dataValuesX.size(); ++i) {
        if(std::isnan(dataValuesX[i]) || std::isnan(dataValuesY[i]) || std::isnan(dataValuesZ[i])){continue;}
        m_value += normedWeightValues[i+m_windowSize] * dataValuesX[i] * dataValuesY[i] * dataValuesZ[i];
    }
    
    m_version = 0;
}

void OnlineEWMSumProduct3::update(const Ve& inValuesX, const Ve& inValuesY, const Ve& inValuesZ, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }
    
    if (inValuesX.size() != inValuesY.size() || inValuesX.size() != inValuesZ.size()) {
        std::cout << "OnlineEWMSumProduct::update: X和Y的向量大小不一致\n";
        exit(1);
    }
    
    // 更新下游组件（缓存层）
    m_window.m_onlineDataCacheX->update(inValuesX, version);
    m_window.m_onlineDataCacheY->update(inValuesY, version);
    m_window.m_onlineDataCacheZ->update(inValuesZ, version);
    // 增量更新：加上新值的加权乘积和，减去旧值的加权乘积和
    const auto& outValuesX = m_window.m_onlineDataCacheX->getOutValues();
    const auto& outValuesY = m_window.m_onlineDataCacheY->getOutValues();
    const auto& outValuesZ = m_window.m_onlineDataCacheZ->getOutValues();
    const auto& unnormalizedWeightValues = m_window.m_onlineWeightCache->getUnnormalizedValues();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();

    computeEWMSumProduct(outValuesX, outValuesY, outValuesZ, unnormalizedWeightValues, normedWeightValues, inValuesX, inValuesY, inValuesZ, m_windowSize, m_value);
    
    m_version = version;  // 更新版本号
}

inline void OnlineEWMSumProduct3::computeEWMSumProduct(const std::deque<double>& outValuesX, const std::deque<double>& outValuesY, 
                                                const std::deque<double>& outValuesZ, const Ve& unnormalizedWeightValues, const Ve& normedWeightValues,
                                                const Ve& inValuesX, const Ve& inValuesY, const Ve& inValuesZ, size_t windowSize, double& Value) noexcept {
    const size_t stepSize = inValuesX.size();
    Value *= unnormalizedWeightValues[(2*windowSize) - stepSize - 1];
    for (size_t i = 0; i < outValuesX.size(); ++i) {
        if(std::isnan(outValuesX[i]) || std::isnan(outValuesY[i]) || std::isnan(outValuesZ[i])){continue;}
        Value -= normedWeightValues[windowSize - stepSize + i] * outValuesX[i] * outValuesY[i] * outValuesZ[i];
    }
    // 加上新值的加权乘积和（使用权重序列的后windowSize个，对应新窗口位置）
    for (size_t i = 0; i < stepSize; ++i) {
        if(std::isnan(inValuesX[i]) || std::isnan(inValuesY[i]) || std::isnan(inValuesZ[i])){continue;}
        Value += normedWeightValues[(2*windowSize) - stepSize + i] * inValuesX[i] * inValuesY[i] * inValuesZ[i];
    }
}

// OnlineEWMMean 带权均值（EWM均值）
void OnlineEWMMean::constructor(const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMMean", window.m_onlineEWMSum);
    CHECK_NULLPTR("OnlineEWMMean", window.m_onlineWeightCache);
    m_window = window;

    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineEWMSum->getWindowSize();
    
    // 计算权重和
    m_weightSum = 1;

    m_value = computeEWMMean(m_window.m_onlineEWMSum->getValue(), m_weightSum);
    m_version = 0;
}

void OnlineEWMMean::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }

    // 更新下游组件（方法层）
    m_window.m_onlineEWMSum->update(inValues, version);

    // 使用compute函数更新自己的值（使用构造时确定的窗口大小）
    const double sum = m_window.m_onlineEWMSum->getValue();
    m_value = computeEWMMean(sum, m_weightSum);
    m_version = version;  // 更新版本号
}

inline double OnlineEWMMean::computeEWMMean(double sum, double weightSum) noexcept {
    if (weightSum <= 0.0 || std::isnan(sum)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / weightSum;
}

// OnlineEWMVar 带权方差（EWM方差，无偏）
void OnlineEWMVar::constructor(const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMVar", window.m_onlineEWMMean);
    CHECK_NULLPTR("OnlineEWMVar", window.m_onlineEWMSumProduct);
    CHECK_NULLPTR("OnlineEWMVar", window.m_onlineWeightCache);

    m_window = window;
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineEWMMean->getWindowSize();


    // 初始方差计算
    const double weightedMean = m_window.m_onlineEWMMean->getValue();
    const double weightedSumSquares = m_window.m_onlineEWMSumProduct->getValue();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    // 计算无偏系数
    BaseFactor::weighted_variance_unbiased_coef(normedWeightValues.tail(m_windowSize), m_varianceBesselCorrection);
    // 计算权重和
    m_weightSum = 1;
    m_value = computeEWMVar(weightedMean, weightedSumSquares, m_weightSum, m_varianceBesselCorrection);
    m_version = 0;
}

void OnlineEWMVar::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }

    // 更新下游组件（方法层）
    m_window.m_onlineEWMMean->update(inValues, version);
    m_window.m_onlineEWMSumProduct->update(inValues, inValues, version);
    const double weightedMean = m_window.m_onlineEWMMean->getValue();
    const double weightedSumSquares = m_window.m_onlineEWMSumProduct->getValue();
    m_value = computeEWMVar(weightedMean, weightedSumSquares, m_weightSum, m_varianceBesselCorrection);
    //debug
    // std::cout << "weightedMean = " << weightedMean << std::endl;
    // std::cout << "weightedSumSquares = " << weightedSumSquares << std::endl;
    // std::cout << "m_weightSum = " << m_weightSum << std::endl;
    // std::cout << "m_varianceBesselCorrection = " << m_varianceBesselCorrection << std::endl;
    // std::cout << "m_value = " << m_value << std::endl;
    // std::cout << "================================================" << std::endl;
    // m_version = version;  // 更新版本号
}

inline double OnlineEWMVar::computeEWMVar(double mean, double sumSquares, double weightSum, double varianceBesselCorrection) noexcept {
    if (weightSum <= 0.0 || std::isnan(mean) || std::isnan(sumSquares)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double var = (sumSquares / weightSum) - mean * mean;
    if (std::isnan(var)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return var * varianceBesselCorrection;
}

// OnlineEWMCov 带权协方差（EWM协方差，无偏）
void OnlineEWMCov::constructor(const Window& window) {
    m_window = window;

    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMCov", window.m_onlineEWMMeanX);
    CHECK_NULLPTR("OnlineEWMCov", window.m_onlineEWMMeanY);
    CHECK_NULLPTR("OnlineEWMCov", window.m_onlineEWMSumProductXY);
    CHECK_NULLPTR("OnlineEWMCov", window.m_onlineWeightCache);

    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineEWMSumProductXY->getWindowSize();

    // 初始协方差计算
    const double weightedMeanX = m_window.m_onlineEWMMeanX->getValue();
    const double weightedMeanY = m_window.m_onlineEWMMeanY->getValue();
    const double weightedSumXY = m_window.m_onlineEWMSumProductXY->getValue();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    // 计算无偏系数
    BaseFactor::weighted_variance_unbiased_coef(normedWeightValues.tail(m_windowSize), m_varianceBesselCorrection);
    
    // 计算权重和
    m_weightSum = 1;
    m_value = computeEWMCov(weightedSumXY, weightedMeanX, weightedMeanY, m_weightSum, m_varianceBesselCorrection);
    m_version = 0;

    //debug
    // std::cout << "weightedSumXY = " << weightedSumXY << std::endl;
    // std::cout << "weightedMeanX = " << weightedMeanX << std::endl;
    // std::cout << "weightedMeanY = " << weightedMeanY << std::endl;
    // std::cout << "m_weightSum = " << m_weightSum << std::endl;
    // std::cout << "m_varianceBesselCorrection = " << m_varianceBesselCorrection << std::endl;
    // std::cout << "m_value = " << m_value << std::endl;
    // std::cout << "================================================" << std::endl;
}

void OnlineEWMCov::update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }

    // 更新下游组件（方法层）
    m_window.m_onlineEWMMeanX->update(inValuesX, version);
    m_window.m_onlineEWMMeanY->update(inValuesY, version);
    m_window.m_onlineEWMSumProductXY->update(inValuesX, inValuesY, version);
    
    const double weightedMeanX = m_window.m_onlineEWMMeanX->getValue();
    const double weightedMeanY = m_window.m_onlineEWMMeanY->getValue();
    const double weightedSumXY = m_window.m_onlineEWMSumProductXY->getValue();

    m_value = computeEWMCov(weightedSumXY, weightedMeanX, weightedMeanY, m_weightSum, m_varianceBesselCorrection);
    m_version = version;  // 更新版本号
}

inline double OnlineEWMCov::computeEWMCov(double sumXY, double meanX, double meanY, double weightSum, double varianceBesselCorrection) noexcept {
    if (weightSum <= 0.0 || std::isnan(sumXY) || std::isnan(meanX) || std::isnan(meanY)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return ((sumXY / weightSum) - (meanX * meanY)) * varianceBesselCorrection;
}

// OnlineEWMSkew 带权偏度
void OnlineEWMSkew::constructor(const Window& window) {
    m_window = window;

    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineEWMSkew", window.m_onlineEWMMean);
    CHECK_NULLPTR("OnlineEWMSkew", window.m_onlineEWMSumProduct);
    CHECK_NULLPTR("OnlineEWMSkew", window.m_onlineEWMSumProduct3);
    CHECK_NULLPTR("OnlineEWMSkew", window.m_onlineWeightCache);

    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineEWMSumProduct3->getWindowSize();

    // 初始偏度计算
    const double weightedMean = m_window.m_onlineEWMMean->getValue();
    const double weightedSumProduct = m_window.m_onlineEWMSumProduct->getValue();
    const double weightedSumProduct3 = m_window.m_onlineEWMSumProduct3->getValue();
    const auto& normedWeightValues = m_window.m_onlineWeightCache->getNormedValues();
    // 计算无偏系数
    m_weightedSkewBesselCorrection = BaseFactor::computeWeightedSkewBesselCorrection(normedWeightValues.tail(m_windowSize));
    // EWM 权重和固定为 1
    m_weightSum = 1.0;
    //计算值
    m_value = computeEWMSkew(weightedMean, weightedSumProduct, weightedSumProduct3, m_weightSum, m_weightedSkewBesselCorrection);
    m_version = 0;
}

void OnlineEWMSkew::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if (m_version == version) {
        return;
    }

    // 更新下游组件（方法层）
    m_window.m_onlineEWMMean->update(inValues, version);
    m_window.m_onlineEWMSumProduct->update(inValues, inValues, version);
    m_window.m_onlineEWMSumProduct3->update(inValues, inValues, inValues, version);

    const double weightedMean = m_window.m_onlineEWMMean->getValue();
    const double weightedSumProduct = m_window.m_onlineEWMSumProduct->getValue();
    const double weightedSumProduct3 = m_window.m_onlineEWMSumProduct3->getValue();
    m_value = computeEWMSkew(weightedMean, weightedSumProduct, weightedSumProduct3, m_weightSum, m_weightedSkewBesselCorrection);
    m_version = version;  // 更新版本号
}

inline double OnlineEWMSkew::computeEWMSkew(double mean, double sumProduct, double sumProduct3, double weightSum, double weightedSkewBesselCorrection) noexcept {
    if (std::isnan(mean) || std::isnan(sumProduct) || std::isnan(sumProduct3) || weightSum <= 0.0){
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    const double m2 = sumProduct - (mean * mean);
    const double m3 = sumProduct3 - (3 * mean * sumProduct) + (2 * mean * mean * mean);

    if (std::isnan(m2) || std::abs(m2) < 1e-14){
        return std::numeric_limits<double>::quiet_NaN();
    }

    return (m3 / std::sqrt(m2 * m2 * m2)) * weightedSkewBesselCorrection;
}
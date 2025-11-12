#include "OnlineMethod.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <algorithm>

// OnlineSum 求和
void OnlineSum::constructor(const Ve& initialValue, const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineSum", window.m_onlineDataCache);
    
    m_window = window;
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineDataCache->getWindowSize();
    // 使用compute函数计算sum
    const auto& samples = m_window.m_onlineDataCache->getValues();
    for(auto sample : samples){
        m_value += sample;
    }
    m_version = 0;
}

void OnlineSum::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    // 更新下游组件（缓存层）
    m_window.m_onlineDataCache->update(inValues, version);
    
    // 增量更新：加上新值的总和，减去旧值的总和
    const auto& outValues = m_window.m_onlineDataCache->getOutValues();
    m_value += computeSum(outValues, inValues);
    m_version = version;  // 更新版本号
}

inline double OnlineSum::computeSum(const std::deque<double>& outValues, const Ve& inValues) noexcept {
    double delta = 0.0;
    for(size_t i = 0; i < inValues.size(); ++i){
        delta += inValues[i];
        delta -= outValues[i];
    }
    return delta;
}

// OnlineSumProduct 乘积和
void OnlineSumProduct::constructor(const Ve& initialValueX, const Ve& initialValueY, const Window& window) {
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineSumProduct", window.m_onlineDataCacheX);
    CHECK_NULLPTR("OnlineSumProduct", window.m_onlineDataCacheY);
    
    m_window = window;
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    const size_t windowSizeX = m_window.m_onlineDataCacheX->getWindowSize();
    const size_t windowSizeY = m_window.m_onlineDataCacheY->getWindowSize();
    if (windowSizeX != windowSizeY) {
        std::cout << "OnlineSumProduct: X和Y的窗口大小不一致\n";
        exit(1);
    }
    m_windowSize = windowSizeX;
    
    // 使用compute函数计算乘积和
    const auto& samplesX = m_window.m_onlineDataCacheX->getValues();
    const auto& samplesY = m_window.m_onlineDataCacheY->getValues();
    for(size_t i = 0; i < samplesX.size(); ++i){
        m_value += samplesX[i] * samplesY[i];
    }
}

void OnlineSumProduct::update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    if(inValuesX.size() != inValuesY.size()){
        std::cout << "OnlineSumProduct::update: X和Y的向量大小不一致\n";
        exit(1);
    }
    
    // 更新下游组件（缓存层）
    m_window.m_onlineDataCacheX->update(inValuesX, version);
    m_window.m_onlineDataCacheY->update(inValuesY, version);
    
    // 增量更新：加上新值的乘积和，减去旧值的乘积和
    const auto& oldValuesX = m_window.m_onlineDataCacheX->getOutValues();
    const auto& oldValuesY = m_window.m_onlineDataCacheY->getOutValues();
    m_value += computeSumProduct(oldValuesX, oldValuesY, inValuesX, inValuesY);
    m_version = version;  // 更新版本号
}

inline double OnlineSumProduct::computeSumProduct(const std::deque<double>& oldValuesX, const std::deque<double>& oldValuesY, 
                                                   const Ve& newValuesX, const Ve& newValuesY) noexcept {
    // 计算增量：新值乘积和 - 旧值乘积和
    double delta = 0.0;
    for(size_t i = 0; i < newValuesX.size(); ++i){
        delta += newValuesX[i] * newValuesY[i];  // 加上新值的乘积
        delta -= oldValuesX[i] * oldValuesY[i];  // 减去旧值的乘积
    }
    return delta;
}

// OnlineMean 均值
void OnlineMean::constructor(const Ve& initialValue, const Window& window){
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineMean", window.m_onlineSum);
    
    m_window = window;
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineSum->getWindowSize();
    // 使用compute函数计算均值
    const double sum = m_window.m_onlineSum->getValue();
    m_value = computeMean(sum, m_windowSize);
}

void OnlineMean::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    // 更新下游组件（方法层）
    m_window.m_onlineSum->update(inValues, version);
    
    // 使用compute函数更新自己的值（使用构造时确定的窗口大小）
    const double sum = m_window.m_onlineSum->getValue();
    m_value = computeMean(sum, m_windowSize);
    m_version = version;  // 更新版本号
}

inline double OnlineMean::computeMean(double sum, size_t windowSize) noexcept {
    return (windowSize > 0) ? (sum / static_cast<double>(windowSize)) : 0.0;
}

// OnlineVar 方差
void OnlineVar::constructor(const Ve& initialValue, const Window& window){
    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineVar", window.m_onlineMean);
    CHECK_NULLPTR("OnlineVar", window.m_onlineSumProduct);
    
    m_window = window;
    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineSumProduct->getWindowSize();

    const double mean = m_window.m_onlineMean->getValue();
    const double sumSquares = m_window.m_onlineSumProduct->getValue();
    m_value = computeVariance(sumSquares, mean, m_windowSize);
}

void OnlineVar::update(const Ve& inValues, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    // 更新窗口中所有下游组件
    m_window.m_onlineMean->update(inValues, version);
    m_window.m_onlineSumProduct->update(inValues, inValues, version);
    
    // 使用构造时确定的窗口大小
    const double mean = m_window.m_onlineMean->getValue();
    const double sumSquares = m_window.m_onlineSumProduct->getValue();
    m_value = computeVariance(sumSquares, mean, m_windowSize);
    m_version = version;  // 更新版本号
}

inline double OnlineVar::computeVariance(double sumSquares, double mean, size_t windowSize) noexcept {
    const double numSamples = static_cast<double>(windowSize);
    return (numSamples > 1.0) 
        ? std::max(0.0, (sumSquares - numSamples * mean * mean) / (numSamples - 1.0))
        : 0.0;
}

// OnlineCov 协方差
void OnlineCov::constructor(const Ve& initialValueX, const Ve& initialValueY, const Window& window){
    m_window = window;

    // 检查必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineCov", window.m_onlineMeanX);
    CHECK_NULLPTR("OnlineCov", window.m_onlineMeanY);
    CHECK_NULLPTR("OnlineCov", window.m_onlineSumProductXY);

    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    m_windowSize = m_window.m_onlineSumProductXY->getWindowSize();

    const double meanX = m_window.m_onlineMeanX->getValue();
    const double meanY = m_window.m_onlineMeanY->getValue();
    const double sumXY = m_window.m_onlineSumProductXY->getValue();
    m_value = computeCovariance(sumXY, meanX, meanY, m_windowSize);
}

void OnlineCov::update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    // 更新窗口中所有下游组件
    m_window.m_onlineMeanX->update(inValuesX, version);
    m_window.m_onlineMeanY->update(inValuesY, version);
    m_window.m_onlineSumProductXY->update(inValuesX, inValuesY, version);
    
    // 使用构造时确定的窗口大小
    const double meanX = m_window.m_onlineMeanX->getValue();
    const double meanY = m_window.m_onlineMeanY->getValue();
    const double sumXY = m_window.m_onlineSumProductXY->getValue();
    m_value = computeCovariance(sumXY, meanX, meanY, m_windowSize);
    m_version = version;  // 更新版本号
}

inline double OnlineCov::computeCovariance(double sumXY, double meanX, double meanY, size_t windowSize) noexcept {
    const double numSamples = static_cast<double>(windowSize);
    return (numSamples > 1.0)
        ? ((sumXY - numSamples * meanX * meanY) / (numSamples - 1.0))
        : std::numeric_limits<double>::quiet_NaN();
}

// OnlineCorrelation 相关系数
void OnlineCorrelation::constructor(const Window& window){
    m_window = window;
    
    // 检查所有必需的组件是否已实例化（框架层只负责链接，不负责创建）
    CHECK_NULLPTR("OnlineCorrelation", m_window.m_onlineVarX);
    CHECK_NULLPTR("OnlineCorrelation", m_window.m_onlineVarY);
    CHECK_NULLPTR("OnlineCorrelation", m_window.m_onlineCovXY);

    // 在构造时确定窗口大小（只确定一次，后续不再更新）
    // 假设X和Y的窗口大小相同，从OnlineVarX获取
    m_windowSize = m_window.m_onlineVarX->getWindowSize();

    const double varX = m_window.m_onlineVarX->getValue();
    const double varY = m_window.m_onlineVarY->getValue();
    const double covXY = m_window.m_onlineCovXY->getValue();

    m_value = computeCorrelation(covXY, varX, varY);
}

void OnlineCorrelation::update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept {
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    
    if(inValuesX.size() != inValuesY.size()){
        std::cout << "OnlineCorrelation::update: X和Y的向量大小不一致\n";
        exit(1);
    }
    
    // 更新下游组件（方法层）
    m_window.m_onlineVarX->update(inValuesX, version);
    m_window.m_onlineVarY->update(inValuesY, version);
    m_window.m_onlineCovXY->update(inValuesX, inValuesY, version);
    
    // 使用构造时确定的窗口大小，更新自己的值
    const double varX = m_window.m_onlineVarX->getValue();
    const double varY = m_window.m_onlineVarY->getValue();
    const double covXY = m_window.m_onlineCovXY->getValue();
    
    m_value = computeCorrelation(covXY, varX, varY);
    m_version = version;  // 更新版本号
}

inline double OnlineCorrelation::computeCorrelation(double covXY, double varX, double varY) noexcept {
    const double stdX = std::sqrt(varX);
    const double stdY = std::sqrt(varY);
    if(stdX == 0.0 || stdY == 0.0 || std::isnan(covXY)){
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::clamp(covXY / (stdX * stdY), -1.0, 1.0);
}
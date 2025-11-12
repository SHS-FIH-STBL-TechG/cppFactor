#include "OnlineDataCache.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

void OnlineDataCache::constructor(const Ve& initialValue){
    if (initialValue.size() <= 1)
    {
        std::cout << "OnlineDataCache: 初始值大小小于等于1，无法构造\n";
        exit(1);
    }
    m_winValues.clear();
    for(auto val : initialValue){
        m_winValues.push_back(val);
    }
    m_windowSize = initialValue.size();
}

void OnlineDataCache::update(const Ve& inValues, size_t version){
    // 版本号检查：如果版本号相等，则已更新过，直接返回
    if(m_version == version){
        return;
    }
    if(inValues.size() > m_windowSize){
        std::cout << "OnlineDataCache: 新数据大小大于窗口大小，无法更新\n";
        exit(1);
    }
    m_outValues.clear();
    for(auto val : inValues){
        m_winValues.push_back(val);
        m_outValues.push_back(m_winValues.front());
        m_winValues.pop_front();
    }
    m_version = version;  // 更新版本号
}

// OnlineWeightCache 实现
void OnlineWeightCache::constructor(const Ve& initialValue){
    if (initialValue.size() <= 1)
    {
        std::cout << "OnlineWeightCache: 初始权重值大小小于等于1，无法构造\n";
        exit(1);
    }
    m_windowSize = initialValue.size();
    m_unnormalizedValues.resize(m_windowSize);
    m_unnormalizedValues = initialValue;
    m_normedValues.resize(m_windowSize);
    m_normedValues = initialValue;
    // 只用“最新半段”来归一化
    const size_t half_len = m_windowSize / 2;
    double norm_sum = 0.0;

    for (size_t i = m_windowSize - half_len; i < m_windowSize; ++i) {
        norm_sum += m_normedValues[i];
    }

    // 用半段的和进行归一化
    if (norm_sum > 0.0) {
        m_normedValues /= norm_sum;
    } else {
        m_normedValues = VectorXd::Ones(m_windowSize) / static_cast<double>(m_windowSize);
    }

    m_version = 0;
}

void OnlineWeightCache::update(size_t version){
    // 权重缓存通常不需要更新，因为权重是固定的
    if (m_version == version){
        return;
    }
    m_version = version;
}
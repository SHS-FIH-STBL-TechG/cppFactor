#pragma once
#include "OnlineBaseFactor.h"
#include "../Eigen_extra/Eigen_extra.h"
#include <deque>
#include <functional>

using namespace EigenExtra;

// 数据缓存层：维护滑动窗口历史数据缓存
class OnlineDataCache : public OnlineBaseFactor{
    public:
    OnlineDataCache(const Ve& initialValue) { constructor(initialValue); }
    void constructor(const Ve& initialValue);
    void update(const Ve& inValues, size_t version);
    [[nodiscard]] const std::deque<double>& getValues() const {return m_winValues;}
    [[nodiscard]] const std::deque<double>& getOutValues() const {return m_outValues;}
    [[nodiscard]] size_t getWindowSize() const {return m_windowSize;}

    private:
    std::deque<double> m_winValues;
    std::deque<double> m_outValues;
    size_t m_windowSize = 0;
    size_t m_version = 0;
};

class OnlineWeightCache : public OnlineBaseFactor{
// 权重系数缓存层：维护窗口长度大小x2的权重系数序列(todo：目前实现的窗口固定模式，待续)
    public:
    // 构造函数1：直接传入初始权重序列
    OnlineWeightCache(const Ve& initialValue) { constructor(initialValue); }
    void constructor(const Ve& initialValue);

    void update(size_t version);//todo：未必需要实现
    [[nodiscard]] const Ve& getUnnormalizedValues() const {return m_unnormalizedValues;}
    [[nodiscard]] const Ve& getNormedValues() const {return m_normedValues;}
    [[nodiscard]] size_t getWindowSize() const {return m_windowSize;}

    private:
    // 未归一化的值
    Ve m_unnormalizedValues;
    // 归一化的值
    Ve m_normedValues;
    size_t m_windowSize = 0;
    size_t m_version = 0;
};
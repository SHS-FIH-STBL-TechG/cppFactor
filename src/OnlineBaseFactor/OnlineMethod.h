#pragma once
#include "OnlineDataCache.h"
#include <memory>
#include <limits>

// 方法层：通过读取数据缓存值进行统计计算

// 在线求和类（方法层）
class OnlineSum : public OnlineBaseFactor {
    public:
    struct Window{
        // 缓存层
        std::shared_ptr<OnlineDataCache> m_onlineDataCache;
    };
    
    OnlineSum(const Ve& initialValue, const Window& window) {
        constructor(initialValue, window);
    }
    // 构造
    void constructor(const Ve& initialValue, const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算队列的 sum
    [[nodiscard]] static inline double computeSum(const std::deque<double>& outValues, const Ve& inValues) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
};

// 在线乘积和类（方法层，两个输入向量相同时为平方和）
class OnlineSumProduct : public OnlineBaseFactor {
    public:
    struct Window{
        // 缓存层X
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheX;
        // 缓存层Y
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheY;
    };
    
    OnlineSumProduct(const Ve& initialValueX, const Ve& initialValueY, const Window& window) {
        constructor(initialValueX, initialValueY, window);
    }
    // 构造
    void constructor(const Ve& initialValueX, const Ve& initialValueY, const Window& window);
    // 递归更新
    void update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算增量：新值乘积和 - 旧值乘积和（用于增量更新）
    [[nodiscard]] static inline double computeSumProduct(const std::deque<double>& oldValuesX, const std::deque<double>& oldValuesY, 
                                                         const Ve& newValuesX, const Ve& newValuesY) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
};

// 在线均值类
class OnlineMean : public OnlineBaseFactor {
    public:
    struct Window{
        // 方法层（去重后）
        std::shared_ptr<OnlineSum> m_onlineSum;
    };
    
    OnlineMean(const Ve& initialValue, const Window& window) {
        constructor(initialValue, window);
    }
    // 构造
    void constructor(const Ve& initialValue, const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算均值
    [[nodiscard]] static inline double computeMean(double sum, size_t windowSize) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;  // 当前窗口大小，支持未来可变窗口大小
};

// 在线方差类
class OnlineVar : public OnlineBaseFactor {
    public:
    struct Window{
        // 方法层（去重后）
        std::shared_ptr<OnlineSumProduct> m_onlineSumProduct;
        // 方法层（去重后）
        std::shared_ptr<OnlineMean> m_onlineMean;
    };
    
    OnlineVar(const Ve& initialValue, const Window& window) {
        constructor(initialValue, window);
    }
    // 构造
    void constructor(const Ve& initialValue, const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }

    private:
    // 计算方差
    [[nodiscard]] static inline double computeVariance(double sumSquares, double mean, size_t windowSize) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;  // 当前窗口大小，支持未来可变窗口大小
};

// 在线样本协方差类
class OnlineCov : public OnlineBaseFactor {
    public:
    struct Window{
        // 方法层（去重后）
        std::shared_ptr<OnlineSumProduct> m_onlineSumProductXY;
        // 方法层（去重后）
        std::shared_ptr<OnlineMean> m_onlineMeanX;
        std::shared_ptr<OnlineMean> m_onlineMeanY;
    };
    
    OnlineCov(const Ve& initialValueX, const Ve& initialValueY, const Window& window) {
        constructor(initialValueX, initialValueY, window);
    }
    // 构造
    void constructor(const Ve& initialValueX, const Ve& initialValueY, const Window& window);
    // 递归更新
    void update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }

    private:
    // 计算协方差
    [[nodiscard]] static inline double computeCovariance(double sumXY, double meanX, double meanY, size_t windowSize) noexcept;

    Window m_window;
    double m_value = std::numeric_limits<double>::quiet_NaN();
    size_t m_version = 0;
    size_t m_windowSize = 0;  // 当前窗口大小，支持未来可变窗口大小
};

// 在线相关系数类
class OnlineCorrelation : public OnlineBaseFactor {
    public:
    struct Window{
        // 方法层（去重后）
        std::shared_ptr<OnlineVar> m_onlineVarX;
        std::shared_ptr<OnlineVar> m_onlineVarY;
        std::shared_ptr<OnlineCov> m_onlineCovXY;
    };
    
    OnlineCorrelation(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValuesX, const Ve& inValuesY, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算相关系数
    [[nodiscard]] static inline double computeCorrelation(double covXY, double varX, double varY) noexcept;

    Window m_window;
    double m_value = std::numeric_limits<double>::quiet_NaN();
    size_t m_version = 0;
    size_t m_windowSize = 0;  // 当前窗口大小，支持未来可变窗口大小（从依赖的OnlineVar中获取）
};

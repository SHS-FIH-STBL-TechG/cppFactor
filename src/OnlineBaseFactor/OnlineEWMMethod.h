#pragma once
#include "OnlineDataCache.h"
#include <memory>

// 方法层：通过读取数据缓存值进行统计计算

// 在线带权加和类（方法层）
class OnlineEWMSum : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 数据缓存层
        std::shared_ptr<OnlineDataCache> m_onlineDataCache;
    };
    OnlineEWMSum(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算带权加和
    static inline void computeEWMSum(const std::deque<double>& outValues, const Ve& unnormalizedWeightValues, const Ve& normedWeightValues, const Ve& inValues, size_t windowSize, double& Value) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
};

// 在线带权乘积和类（方法层，两个输入向量相同时为加权平方和）
class OnlineEWMSumProduct : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 数据缓存层X
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheX;
        // 数据缓存层Y
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheY;
    };
    OnlineEWMSumProduct( const Window& window) {
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
    // 计算带权乘积和增量（不可静态）
    static inline void computeEWMSumProduct(const std::deque<double>& outValuesX, const std::deque<double>& outValuesY, 
                                            const Ve& unnormalizedWeightValues, const Ve& normedWeightValues,
                                            const Ve& inValuesX, const Ve& inValuesY, size_t windowSize, double& Value) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
};

// 在线带权三元乘积和类
class OnlineEWMSumProduct3 : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 数据缓存层X
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheX;
        // 数据缓存层Y
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheY;
        // 数据缓存层Z
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheZ;
    };
    OnlineEWMSumProduct3(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValuesX, const Ve& inValuesY, const Ve& inValuesZ, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算带权乘积和增量（不可静态）
    static inline void computeEWMSumProduct(const std::deque<double>& outValuesX, const std::deque<double>& outValuesY, 
                                            const std::deque<double>& outValuesZ, const Ve& unnormalizedWeightValues, const Ve& normedWeightValues,
                                            const Ve& inValuesX, const Ve& inValuesY, const Ve& inValuesZ, size_t windowSize, double& Value) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
};

// 在线带权均值类（方法层）
class OnlineEWMMean : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 方法层（去重后）
        std::shared_ptr<OnlineEWMSum> m_onlineEWMSum;
    };
    OnlineEWMMean(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }
    
    private:
    // 计算带权均值
    [[nodiscard]] static inline double computeEWMMean(double sum, double weightSum) noexcept;
    
    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
    //
    double m_weightSum = 0.0;
};

// 在线带权方差类（方法层）
class OnlineEWMVar : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 方法层（去重后）
        std::shared_ptr<OnlineEWMMean> m_onlineEWMMean;
        std::shared_ptr<OnlineEWMSumProduct> m_onlineEWMSumProduct;
    };
    OnlineEWMVar(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }

    private:
    // 计算带权方差（使用权重和以及无偏系数修正）
    [[nodiscard]] static inline double computeEWMVar(double mean, double sumSquares, double weightSum, double varianceBesselCorrection) noexcept;

    Window m_window;
    double m_value = 0.0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
    //
    double m_weightSum = 0.0;
    double m_varianceBesselCorrection = 0.0;
};

// 在线带权协方差类（方法层）
class OnlineEWMCov : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 方法层（去重后）
        std::shared_ptr<OnlineEWMSumProduct> m_onlineEWMSumProductXY;
        std::shared_ptr<OnlineEWMMean> m_onlineEWMMeanX;
        std::shared_ptr<OnlineEWMMean> m_onlineEWMMeanY;
    };
    OnlineEWMCov(const Window& window) {
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
    // 计算带权协方差（使用权重和以及无偏系数修正）
    [[nodiscard]] static inline double computeEWMCov(double sumXY, double meanX, double meanY, double weightSum, double varianceBesselCorrection) noexcept;

    Window m_window;
    double m_value = std::numeric_limits<double>::quiet_NaN();
    size_t m_version = 0;
    size_t m_windowSize = 0;
    //
    double m_weightSum = 0.0;
    double m_varianceBesselCorrection = 0.0;
};

// 在线带权偏度类 （方法层）
class OnlineEWMSkew : public OnlineBaseFactor {
    public:
    struct Window{
        // 权重系数缓存层
        std::shared_ptr<OnlineWeightCache> m_onlineWeightCache;
        // 方法层（去重后）
        std::shared_ptr<OnlineEWMMean> m_onlineEWMMean;
        std::shared_ptr<OnlineEWMSumProduct> m_onlineEWMSumProduct;
        std::shared_ptr<OnlineEWMSumProduct3> m_onlineEWMSumProduct3;
    };
    OnlineEWMSkew(const Window& window) {
        constructor(window);
    }
    // 构造
    void constructor(const Window& window);
    // 递归更新
    void update(const Ve& inValues, size_t version) noexcept;
    // 获取值
    [[nodiscard]] double getValue() const noexcept { return m_value; }
    // 获取窗口大小
    [[nodiscard]] size_t getWindowSize() const noexcept { return m_windowSize; }

    private:
    // 计算带权偏度
    [[nodiscard]] static inline double computeEWMSkew(double mean, double sumProduct, double sumProduct3, double weightSum, double weightedSkewBesselCorrection) noexcept;

    Window m_window;
    double m_value = std::numeric_limits<double>::quiet_NaN();
    size_t m_version = 0;
    size_t m_windowSize = 0;
    //
    double m_weightSum = 0.0;
    double m_weightedSkewBesselCorrection = 0.0;
};
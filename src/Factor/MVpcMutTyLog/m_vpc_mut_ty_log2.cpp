#include "m_vpc_mut_ty_log2.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <vector>
#include "OnlineBaseFactor/OnlineMethod.h"

namespace {
constexpr double kRankThreshold = 0.9;
constexpr int kDefaultWindowSize = 240;
constexpr double kValueTol = 1e-12;
constexpr double kZeroTol = 1e-12;

struct RankedReturn {
    double value;
    std::size_t index;
};

inline bool almostEqual(double lhs, double rhs) {
    const double absLhs = std::fabs(lhs);
    const double absRhs = std::fabs(rhs);
    double scale = std::max(absLhs, absRhs);
    scale = std::max(scale, 1.0);
    return std::fabs(lhs - rhs) <= kValueTol * scale;
}

void buildTopRankMask(const std::vector<double>& returns,
                      std::vector<bool>& mask) {
    mask.assign(returns.size(), false);

    std::vector<RankedReturn> ranked;
    ranked.reserve(returns.size());
    for (std::size_t idx = 0; idx < returns.size(); ++idx) {
        double val = returns[idx];
        if (std::isnan(val) || !std::isfinite(val)) {
            continue;
        }
        ranked.push_back(RankedReturn{val, idx});
    }

    const std::size_t validCount = ranked.size();
    if (validCount == 0) {
        return;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedReturn& lhs, const RankedReturn& rhs) {
                  if (almostEqual(lhs.value, rhs.value)) {
                      return lhs.index < rhs.index;
                  }
                  return lhs.value < rhs.value;
              });

    const double thresholdRank = kRankThreshold * static_cast<double>(validCount);
    std::size_t pos = 0;
    while (pos < validCount) {
        const double currentValue = ranked[pos].value;
        const std::size_t groupStart = pos;
        while (pos < validCount && almostEqual(ranked[pos].value, currentValue)) {
            ++pos;
        }
        const std::size_t groupEnd = pos; // exclusive
        const double firstRank = static_cast<double>(groupStart + 1);
        const double lastRank = static_cast<double>(groupEnd);
        const double avgRank = (firstRank + lastRank) * 0.5;
        if ((avgRank / static_cast<double>(validCount)) > kRankThreshold) {
            for (std::size_t k = groupStart; k < groupEnd; ++k) {
                mask[ranked[k].index] = true;
            }
        }
    }
}

double computeCorrelationOnline(const Ve& closeVec,
                                const Ve& volumeVec) {
    if (closeVec.size() != volumeVec.size() || closeVec.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto cacheClose = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(closeVec);
    auto cacheVolume = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(volumeVec);

    auto sumClose = OnlineBaseFactor::createOnlineBaseF<OnlineSum>(
        closeVec, OnlineSum::Window{cacheClose});
    auto sumVolume = OnlineBaseFactor::createOnlineBaseF<OnlineSum>(
        volumeVec, OnlineSum::Window{cacheVolume});

    auto sumProductCloseClose = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(
        closeVec, closeVec, OnlineSumProduct::Window{cacheClose, cacheClose});
    auto sumProductVolumeVolume = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(
        volumeVec, volumeVec, OnlineSumProduct::Window{cacheVolume, cacheVolume});
    auto sumProductCross = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(
        closeVec, volumeVec, OnlineSumProduct::Window{cacheClose, cacheVolume});

    auto meanClose = OnlineBaseFactor::createOnlineBaseF<OnlineMean>(
        closeVec, OnlineMean::Window{sumClose});
    auto meanVolume = OnlineBaseFactor::createOnlineBaseF<OnlineMean>(
        volumeVec, OnlineMean::Window{sumVolume});

    auto varClose = OnlineBaseFactor::createOnlineBaseF<OnlineVar>(
        closeVec, OnlineVar::Window{sumProductCloseClose, meanClose});
    auto varVolume = OnlineBaseFactor::createOnlineBaseF<OnlineVar>(
        volumeVec, OnlineVar::Window{sumProductVolumeVolume, meanVolume});

    auto covCross = OnlineBaseFactor::createOnlineBaseF<OnlineCov>(
        closeVec, volumeVec, OnlineCov::Window{sumProductCross, meanClose, meanVolume});

    auto correlation = OnlineBaseFactor::createOnlineBaseF<OnlineCorrelation>(
        OnlineCorrelation::Window{varClose, varVolume, covCross});

    return correlation->getValue();
}

double computeCorrelation(const std::deque<double>& closes,
                          const std::deque<double>& volumes) {
    if (closes.size() != volumes.size() || closes.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> returns(closes.size(),
                                std::numeric_limits<double>::quiet_NaN());
    for (std::size_t idx = 1; idx < closes.size(); ++idx) {
        const double prevClose = closes[idx - 1];
        const double currClose = closes[idx];
        if (std::isnan(prevClose) || std::isnan(currClose) ||
            !std::isfinite(prevClose) || std::fabs(prevClose) < kZeroTol) {
            returns[idx] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        returns[idx] = (currClose - prevClose) / prevClose;
    }

    std::vector<bool> mask;
    buildTopRankMask(returns, mask);

    const std::size_t flaggedCount =
        static_cast<std::size_t>(std::count(mask.begin(), mask.end(), true));
    if (flaggedCount == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Ve closeVec(flaggedCount);
    Ve volumeVec(flaggedCount);
    std::size_t cursor = 0;
    for (std::size_t idx = 0; idx < mask.size(); ++idx) {
        if (!mask[idx]) {
            continue;
        }
        closeVec(cursor) = closes[idx];
        volumeVec(cursor) = volumes[idx];
        ++cursor;
    }

    return computeCorrelationOnline(closeVec, volumeVec);
}
} // namespace

m_vpc_mut_ty_log2::m_vpc_mut_ty_log2() = default;

m_vpc_mut_ty_log2::~m_vpc_mut_ty_log2() = default;

void m_vpc_mut_ty_log2::Finish()
{
    // 暂无额外资源需要清理
}

int m_vpc_mut_ty_log2::Init(const Ma& initAmt, const Ma& initClose)
{
    const int lagWindow = initAmt.rows();
    const int stocksNum = initAmt.cols();

    if (lagWindow < 2 || initClose.rows() != lagWindow || initClose.cols() != stocksNum) {
        std::cerr << "Init Error: 输入矩阵维度不匹配或窗口长度不足\n";
        return -1;
    }

    const int effectiveWindow = std::min(lagWindow, kDefaultWindowSize);
    if (effectiveWindow < 2) {
        std::cerr << "Init Error: 有效窗口长度不足\n";
        return -1;
    }

    m_windowSize = static_cast<std::size_t>(effectiveWindow);
    m_version = 0;
    m_windows.resize(stocksNum);

    for (int i = 0; i < stocksNum; ++i) {
        auto& window = m_windows[i];

        const Ve initVolume = initAmt.col(i).tail(effectiveWindow).eval();
        const Ve initCloseVec = initClose.col(i).tail(effectiveWindow).eval();

        window.m_onlineDataCacheVolume =
            OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initVolume);
        window.m_onlineDataCacheClose =
            OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initCloseVec);
    }

    m_valueCoef.resize(stocksNum);
    for (int i = 0; i < stocksNum; ++i) {
        const auto& window = m_windows[i];
        m_valueCoef[i] = computeCorrelation(
            window.m_onlineDataCacheClose->getValues(),
            window.m_onlineDataCacheVolume->getValues());
    }

    return 0;
}

void m_vpc_mut_ty_log2::Update(const Ma& newAmt, const Ma& newClose)
{
    if (newAmt.cols() != static_cast<int>(m_windows.size()) ||
        newClose.cols() != static_cast<int>(m_windows.size()) ||
        newAmt.rows() != newClose.rows()) {
        std::cerr << "Update Error: 输入矩阵维度不匹配\n";
        return;
    }

    const int rows = newAmt.rows();
    const int cols = newAmt.cols();

    for (int col = 0; col < cols; ++col) {
        auto& window = m_windows[col];

        for (int row = 0; row < rows; ++row) {
            ++m_version;
            Ve volumeValue(1);
            volumeValue[0] = newAmt(row, col);
            window.m_onlineDataCacheVolume->update(volumeValue, m_version);

            ++m_version;
            Ve closeValue(1);
            closeValue[0] = newClose(row, col);
            window.m_onlineDataCacheClose->update(closeValue, m_version);
        }

        m_valueCoef[col] = computeCorrelation(
            window.m_onlineDataCacheClose->getValues(),
            window.m_onlineDataCacheVolume->getValues());
    }
}


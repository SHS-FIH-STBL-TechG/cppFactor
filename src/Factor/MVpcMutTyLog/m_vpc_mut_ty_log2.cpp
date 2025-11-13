#include "m_vpc_mut_ty_log2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <limits>
#include <vector>

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
    m_windowSize = lagWindow;
    m_windows.resize(stocksNum);
    m_valueCoef.resize(stocksNum);
    for(int i = 0; i < stocksNum; ++i){
        // 构建缓存序列
        m_windows[i].cacheDataAmt = initAmt.col(i);
        m_windows[i].cacheDataClose = initClose.col(i);

        // 构建收益率排名
        Ve retClose(m_windowSize);
        BusinessFactor::to_diff_return_sequence(m_windows[i].cacheDataClose, retClose);
        Ve rankClose(m_windowSize);
        BaseFactor::rankpct(retClose, rankClose);

        // 构建输入，类似flag操作
        Ve flagClose = m_windows[i].cacheDataClose;
        for(int j = 0;j < rankClose.size(); ++j){
            if(rankClose(j) < 0.9 || std::isnan(rankClose(j))){
                flagClose[j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        double res;
        BaseFactor::pearson_correlation(flagClose, m_windows[i].cacheDataAmt, res);
        m_valueCoef[i] = res;
    }
    return 0;
}

void m_vpc_mut_ty_log2::Update(const Ma& newAmt, const Ma& newClose)
{
    size_t stepSize = newAmt.rows();
    for(int i = 0; i < newAmt.cols(); ++i){
        // 构建缓存序列
        m_windows[i].cacheDataAmt.head(m_windowSize - stepSize) = m_windows[i].cacheDataAmt.tail(m_windowSize - stepSize);
        m_windows[i].cacheDataAmt.tail(stepSize) = newAmt.col(i);
        m_windows[i].cacheDataClose.head(m_windowSize - stepSize) = m_windows[i].cacheDataClose.tail(m_windowSize - stepSize);
        m_windows[i].cacheDataClose.tail(stepSize) = newClose.col(i);

        // 构建收益率排名
        Ve retClose(m_windowSize);
        BusinessFactor::to_diff_return_sequence(m_windows[i].cacheDataClose, retClose);
        Ve rankClose(m_windowSize);
        BaseFactor::rankpct(retClose, rankClose);
        // 构建输入，类似flag操作
        Ve flagClose = m_windows[i].cacheDataClose;
        for(int j = 0;j < rankClose.size(); ++j){
            if(rankClose(j) < 0.9 || std::isnan(rankClose(j))){
                flagClose[j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
        double res;
        BaseFactor::pearson_correlation(flagClose, m_windows[i].cacheDataAmt, res);
        m_valueCoef[i] = res;

        // debug
        static int updateNum = 0;
        updateNum ++;
        if(updateNum == 2){
            int debug =1;
        }
    }
}


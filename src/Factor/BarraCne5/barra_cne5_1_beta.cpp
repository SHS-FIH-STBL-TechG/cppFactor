#include "barra_cne5_1_beta.h"
#include "../../Eigen_extra/RingVec.h"
#include "../../Tool/profiler.h"

barra_cne5_1_beta::barra_cne5_1_beta(/* args */)
{
    // 构造函数：初始化成员变量
}

barra_cne5_1_beta::~barra_cne5_1_beta()
{
    // 析构函数
}

void barra_cne5_1_beta::Finish()
{
    // 结束函数：清理资源
}

int barra_cne5_1_beta::Init(const Ma& initRet, const Ma& initCap, const Ve& initValid)
{
    //初始化窗口
    //1.假设获取到了初始数据 假设三团数据是完全对齐的（暂时不做股票名字对齐）
    int lagWindow = initRet.rows();  // 从实际数据获取行数（时间窗口大小）
    int stocksNum = initRet.cols();  // 从实际数据获取列数（股票数量）
    Ma ret = initRet;
    Ma cap = initCap;
    Ve valid = initValid;
    
    // 初始化m_value为正确的股票数量
    m_value.resize(stocksNum);
    // 初始化窗口向量，为每只股票预留空间
    m_windows.resize(stocksNum);
    //......

    //winsor处理（就地裁剪，减少拷贝）
    {
        PROFILE_SCOPE("Init::winsor处理");
        DataProcess::winsorInplace(ret, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
    }
    //构建维护类
    //1.构建简单市场收益率值队列
    Ve marketRet(lagWindow);
    {
        PROFILE_SCOPE("Init::构建市场收益率");
        for(int i = 0; i < lagWindow; i++){
            marketRet[i] = ret.row(i).dot(cap.row(i)) / cap.row(i).sum();
        }
    }
    //2.为每只股票构建beta值窗口
    //2.1构建相关系数窗口
    {
        PROFILE_SCOPE("Init::构建beta值窗口");
        for(int i = 0; i < stocksNum; i++){
            // 构造该股票与市场的相关系数所需组件（容器层 -> 方法层）
            // 容器层
            auto cacheX = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(ret.col(i));
            auto cacheY = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(marketRet);
            
            // 方法层：求和和乘积和
            auto sumX  = OnlineBaseFactor::createOnlineBaseF<OnlineSum>(ret.col(i), OnlineSum::Window{cacheX});
            auto sumY  = OnlineBaseFactor::createOnlineBaseF<OnlineSum>(marketRet, OnlineSum::Window{cacheY});
            auto prodXX = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(ret.col(i), ret.col(i), OnlineSumProduct::Window{cacheX, cacheX});
            auto prodYY = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(marketRet, marketRet, OnlineSumProduct::Window{cacheY, cacheY});
            auto prodXY = OnlineBaseFactor::createOnlineBaseF<OnlineSumProduct>(ret.col(i), marketRet, OnlineSumProduct::Window{cacheX, cacheY});

            // 方法层：均值
            auto meanX = OnlineBaseFactor::createOnlineBaseF<OnlineMean>(ret.col(i), OnlineMean::Window{sumX});
            auto meanY = OnlineBaseFactor::createOnlineBaseF<OnlineMean>(marketRet, OnlineMean::Window{sumY});

            // 方法层：方差
            auto varX = OnlineBaseFactor::createOnlineBaseF<OnlineVar>(ret.col(i), OnlineVar::Window{prodXX, meanX});
            auto varY = OnlineBaseFactor::createOnlineBaseF<OnlineVar>(marketRet, OnlineVar::Window{prodYY, meanY});

            // 方法层：协方差
            auto covXY = OnlineBaseFactor::createOnlineBaseF<OnlineCov>(ret.col(i), marketRet, OnlineCov::Window{prodXY, meanX, meanY});
            
            // 方法层：相关系数
            auto correlation = OnlineBaseFactor::createOnlineBaseF<OnlineCorrelation>(OnlineCorrelation::Window{varX, varY, covXY});

            // 应用层：Beta - 为每只股票保存独立的窗口
            m_windows[i].m_onlineCorrelation = correlation;
            m_windows[i].m_onlineVarX = varX;
            m_windows[i].m_onlineVarY = varY;
            // Beta公式：beta = corr * std(ret) / std(rm) = corr * sqrt(var(ret)) / sqrt(var(rm))
            double corr = correlation->getValue();
            double varX_val = varX->getValue();
            double varY_val = varY->getValue();
            if(std::isnan(corr) || std::isnan(varX_val) || std::isnan(varY_val) || varY_val <= 0.0){
                m_value[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double stdX = std::sqrt(varX_val);
                double stdY = std::sqrt(varY_val);
                double beta = corr * stdX / stdY;
                m_value[i] = beta;
            }
        }
    }
    // 分位数缩尾
    {
        PROFILE_SCOPE("Init::分位数缩尾");
        m_value = DataProcess::clipByQuantile(m_value, WINSOR_QUANTILE_LOW, WINSOR_QUANTILE_HIGH);
    }

    // 有效性检测
    {
        PROFILE_SCOPE("Init::有效性检测");
        m_value.array() *= valid.array();
    }

    // 标准化
    {
        PROFILE_SCOPE("Init::标准化");
        Ve capLastRow = cap.row(cap.rows() - 1).transpose();  // 行向量转置为列向量
        double betaMean = (m_value.array() * capLastRow.array()).sum() / capLastRow.array().sum();
        double betaStd;
        BaseFactor::nanstd(m_value, betaStd, true);
        m_value = (m_value.array() - betaMean) / betaStd;
    }
    return 0;
}

void barra_cne5_1_beta::Update(const Ma& ret, const Ma& cap, const Ve& valid)
{
    // 假设来了一批数据
    Ma newRet = ret;
    Ma newCap = cap;
    Ve newValid = valid;

    // winsor处理（就地裁剪，减少拷贝）
    {
        PROFILE_SCOPE("Update::winsor处理");
        DataProcess::winsorInplace(newRet, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
    }

    //1.构建增量市场收益率值队列
    Ve newMarketRet(newRet.rows());
    {
        PROFILE_SCOPE("Update::构建市场收益率");
        for(int i = 0; i < newRet.rows(); i++){
            newMarketRet[i] = newRet.row(i).dot(newCap.row(i)) / newCap.row(i).sum();
        }
    }

    //2.更新beta值窗口
    m_version++;
    // 为每只股票更新相关系数
    {
        PROFILE_SCOPE("Update::更新beta值窗口");
        for(int i = 0; i < newRet.cols(); i++){
            // OnlineCorrelation::update内部会更新m_onlineVarX和m_onlineVarY
            m_windows[i].m_onlineCorrelation->update(newRet.col(i), newMarketRet, m_version);
            
            //3.计算beta值
            double corr = m_windows[i].m_onlineCorrelation->getValue();
            double varX = m_windows[i].m_onlineVarX->getValue();
            double varY = m_windows[i].m_onlineVarY->getValue();
            
            // 检查除零和NaN情况
            if(std::isnan(corr) || std::isnan(varX) || std::isnan(varY) || varY <= 0.0){
                m_value[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double stdX = std::sqrt(varX);
                double stdY = std::sqrt(varY);
                double beta = corr * stdX / stdY;
                m_value[i] = beta;
            }
        }
    }
    // 分位数缩尾
    {
        PROFILE_SCOPE("Update::分位数缩尾");
        m_value = DataProcess::clipByQuantile(m_value, WINSOR_QUANTILE_LOW, WINSOR_QUANTILE_HIGH);
    }
    
    // 有效性检测
    {
        PROFILE_SCOPE("Update::有效性检测");
        m_value.array() *= newValid.array();
    }

    // 标准化
    {
        PROFILE_SCOPE("Update::标准化");
        Ve capLastRow = newCap.row(newCap.rows() - 1).transpose();  // 行向量转置为列向量
        double betaMean = (m_value.array() * capLastRow.array()).sum() / capLastRow.array().sum();
        double betaStd;
        BaseFactor::nanstd(m_value, betaStd, true);
        m_value = (m_value.array() - betaMean) / betaStd;
    }
}
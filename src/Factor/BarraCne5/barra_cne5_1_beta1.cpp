#include "barra_cne5_1_beta1.h"
#include "../../Eigen_extra/RingVec.h"
#include "../../Tool/profiler.h"

barra_cne5_1_beta1::barra_cne5_1_beta1(/* args */)
{
    // 构造函数：初始化成员变量
}

barra_cne5_1_beta1::~barra_cne5_1_beta1()
{
    // 析构函数
}

void barra_cne5_1_beta1::Finish()
{
    // 结束函数：清理资源
}

int barra_cne5_1_beta1::Init(const Ma& initRet, const Ma& initCap, const Ve& initValid)
{
    //初始化窗口
    //1.假设获取到了初始数据 假设三团数据是完全对齐的（暂时不做股票名字对齐）
    int lagWindow = initRet.rows();  // 从实际数据获取行数（时间窗口大小）
    int stocksNum = initRet.cols();  // 从实际数据获取列数（股票数量）
    Ma ret = initRet / 100;
    Ma cap = initCap;
    Ve valid = initValid;
    
    // 初始化m_value为正确的股票数量
    m_value.resize(stocksNum);
    // 初始化窗口向量，为每只股票预留空间
    m_windows.resize(stocksNum);
    
    auto retValid = ret;
    // 有效性检测（在winsor处理之前，与Python对齐）
    {
        PROFILE_SCOPE("Init::有效性检测");
        // 将无效股票的数据置为0（与Python的 ret * valid.iloc[-1] 对应）
        // Python中 ret * valid.iloc[-1] 后，如果valid的某个位置是0，ret中该股票的所有时间点数据会变成0
        for(int i = 0; i < lagWindow; i++){
            retValid.row(i).array() *= valid.array();
        }
    }
    
    //winsor处理（就地裁剪，减少拷贝）
    {
        PROFILE_SCOPE("Init::winsor处理");
        DataProcess::winsorInplace(ret, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
        DataProcess::winsorInplace(retValid, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
    }
    
    //构建维护类
    //1.构建简单市场收益率值队列
    Ve marketRet(lagWindow);
    {
        PROFILE_SCOPE("Init::构建市场收益率");
        for(int i = 0; i < lagWindow; i++){
            marketRet[i] = retValid.row(i).dot(cap.row(i)) / cap.row(i).sum();
            marketRet[i] = std::round(marketRet[i] * 1e6) / 1e6;
        }
    }
    
    //2.创建EWM权重缓存（halflife=63，共享给所有股票）
    const size_t windowSize = static_cast<size_t>(lagWindow);
    const size_t ewmWeightsSize = windowSize * 2;
    const double halflife = 2.0;
    VectorXd ewmWeightsVec(ewmWeightsSize);
    BaseFactor::ewm_weights(ewmWeightsSize, halflife, ewmWeightsVec);

    // debug
    // for(int i = 0; i < ewmWeightsVec.size(); i++){
    //     std::cout << ewmWeightsVec[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "================================================" << std::endl;

    auto weightCache = OnlineBaseFactor::createOnlineBaseF<OnlineWeightCache>(ewmWeightsVec);
    
    //3.为每只股票构建beta值窗口
    {
        PROFILE_SCOPE("Init::构建beta值窗口");
        for(int i = 0; i < stocksNum; i++){
            // 构造该股票的EWM协方差和方差所需组件
            // 容器层
            auto cacheX = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(ret.col(i));
            auto cacheY = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(marketRet);
            
            // 方法层：EWM求和
            auto sumX = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSum>(OnlineEWMSum::Window{weightCache, cacheX});
            auto sumY = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSum>(OnlineEWMSum::Window{weightCache, cacheY});
            
            // 方法层：EWM均值
            auto meanX = OnlineBaseFactor::createOnlineBaseF<OnlineEWMMean>(OnlineEWMMean::Window{weightCache, sumX});
            auto meanY = OnlineBaseFactor::createOnlineBaseF<OnlineEWMMean>(OnlineEWMMean::Window{weightCache, sumY});
            
            // 方法层：EWM乘积和（用于方差和协方差）
            auto prodXX = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(
                                                                                      OnlineEWMSumProduct::Window{weightCache, cacheX, cacheX});
            auto prodXY = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(
                                                                                      OnlineEWMSumProduct::Window{weightCache, cacheX, cacheY});
            
            // 方法层：EWM方差和协方差
            auto varX = OnlineBaseFactor::createOnlineBaseF<OnlineEWMVar>(
                                                                             OnlineEWMVar::Window{weightCache, meanX, prodXX});
            auto covXY = OnlineBaseFactor::createOnlineBaseF<OnlineEWMCov>(
                                                                             OnlineEWMCov::Window{weightCache, prodXY, meanX, meanY});
            
            // 应用层：Beta - 为每只股票保存独立的窗口
            m_windows[i].m_onlineEWMVarX = varX;
            m_windows[i].m_onlineEWMCovXY = covXY;
            
            // Beta公式：beta = cov / var
            double cov = covXY->getValue();
            double var = varX->getValue();

            if(std::isnan(cov) || std::isnan(var) || var <= 0.0){
                m_value[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                m_value[i] = cov / var;
            }
            // debug
            // std::cout << "cov = " << cov << std::endl;
            // std::cout << "var = " << var << std::endl;
            // std::cout << "beta = " << m_value[i] << std::endl;
            // std::cout << "================================================" << std::endl;
        }
    }
    
    // 分位数缩尾
    {
        PROFILE_SCOPE("Init::分位数缩尾");
        m_value = DataProcess::clipByQuantile(m_value, WINSOR_QUANTILE_LOW, WINSOR_QUANTILE_HIGH);
    }

    // 有效性过滤
    for(int i = 0; i < m_value.size(); i++){
        if(valid[i] == 0){
            m_value[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // 标准化
    {
        PROFILE_SCOPE("Init::标准化");
        Ve capLastRow = cap.row(cap.rows() - 1).transpose();  // 行向量转置为列向量
        // 手搓均值，防止nan
        double betaMean = 0;
        int count = 0;
        int capsum = 0;
        for(int i = 0; i < m_value.size(); i++){
            if(!std::isnan(m_value[i]) && !std::isnan(capLastRow[i])){
                betaMean += m_value[i] * capLastRow[i];
                count++;
            }
            capsum += capLastRow[i];
        }
        betaMean /= capsum;

        double betaStd;
        BaseFactor::nanstd(m_value, betaStd, true);
        m_value = (m_value.array() - betaMean) / betaStd;
    }

    return 0;
}

void barra_cne5_1_beta1::Update(const Ma& ret, const Ma& cap, const Ve& valid)
{
    // 假设来了一批数据
    Ma newRet = ret / 100;
    Ma newCap = cap;
    Ve newValid = valid;

    // 有效性检测（在winsor处理之前，与Python对齐）（暂时略过，存疑）
    {
        PROFILE_SCOPE("Update::有效性检测");
        // 将无效股票的数据置为0（与Python的 ret * valid.iloc[-1] 对应）
        for(int i = 0; i < newRet.rows(); i++){
            newRet.row(i).array() *= newValid.array();
        }
    }

    auto newRetValid = newRet;
    // winsor处理（就地裁剪，减少拷贝）
    {
        PROFILE_SCOPE("Update::winsor处理");
        DataProcess::winsorInplace(newRet, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
        DataProcess::winsorInplace(newRetValid, PRICE_LIMIT_DOWN_PCT, PRICE_LIMIT_UP_PCT);
    }

    //1.构建增量市场收益率值队列
    Ve newMarketRet(newRet.rows());
    {
        PROFILE_SCOPE("Update::构建市场收益率");
        for(int i = 0; i < newRet.rows(); i++){
            newMarketRet[i] = newRetValid.row(i).dot(newCap.row(i)) / newCap.row(i).sum();
        }
    }

    // debug
    std::cout << "update now============================****************** " << newMarketRet << std::endl;

    //2.更新beta值窗口
    m_version++;
    {
        PROFILE_SCOPE("Update::更新beta值窗口");
        for(int i = 0; i < newRet.cols(); i++){
            // 先更新协方差（它会更新共享的meanX和meanY）
            m_windows[i].m_onlineEWMCovXY->update(newRet.col(i), newMarketRet, m_version);
            // 再更新方差（它会更新共享的meanX，但由于版本号检查会跳过）
            m_windows[i].m_onlineEWMVarX->update(newRet.col(i), m_version);
            
            //3.计算beta值
            double cov = m_windows[i].m_onlineEWMCovXY->getValue();
            double var = m_windows[i].m_onlineEWMVarX->getValue();
            
            // 检查除零和NaN情况
            if(std::isnan(cov) || std::isnan(var) || var <= 0.0){
                m_value[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double beta = cov / var;
                m_value[i] = beta;
            }
            //debug
            std::cout << "cov = " << std::fixed << std::setprecision(9) << cov << std::endl;
            std::cout << "var = " << std::fixed << std::setprecision(9) << var << std::endl;
            std::cout << "beta = " << std::fixed << std::setprecision(9) << m_value[i] << std::endl;
            std::cout << "================================================" << std::endl;
        }
    }
    
    for(int i = 0; i < m_value.size(); i++){
        std::cout << std::fixed << std::setprecision(9) << m_value[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "================================================" << std::endl;
    
    // 分位数缩尾
    {
        PROFILE_SCOPE("Update::分位数缩尾");
        m_value = DataProcess::clipByQuantile(m_value, WINSOR_QUANTILE_LOW, WINSOR_QUANTILE_HIGH);
    }

    // 有效性过滤
    for(int i = 0; i < m_value.size(); i++){
        if(newValid[i] == 0){
            m_value[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    // 标准化
    {
        PROFILE_SCOPE("Update::标准化");
        Ve capLastRow = newCap.row(newCap.rows() - 1).transpose();  // 行向量转置为列向量
        // 手搓均值，防止nan
        double betaMean = 0;
        int count = 0;
        int capsum = 0;
        for(int i = 0; i < m_value.size(); i++){
            if(!std::isnan(m_value[i]) && !std::isnan(capLastRow[i])){
                betaMean += m_value[i] * capLastRow[i];
                count++;
            }
            capsum += capLastRow[i];
        }
        betaMean /= capsum;
        
        double betaStd;
        BaseFactor::nanstd(m_value, betaStd, true);
        m_value = (m_value.array() - betaMean) / betaStd;
    }
}

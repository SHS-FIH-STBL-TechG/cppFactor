#include "m_vpc_mut_ty_log1.h"

m_vpc_mut_ty_log1::m_vpc_mut_ty_log1() = default;

m_vpc_mut_ty_log1::~m_vpc_mut_ty_log1() = default;

void m_vpc_mut_ty_log1::Finish()
{
    // 暂无额外资源需要清理
}

int m_vpc_mut_ty_log1::Init(const Ma& initAmt, const Ma& initClose)
{
    // 1. 获取窗口行列
    int lagWindow = initAmt.rows();   // 时间窗口大小
    int stocksNum = initAmt.cols();   // 股票数量

    if (lagWindow < std_period) {
        std::cerr << "Init Error: lagWindow 必须大于 std_period" << std::endl;
        return -1;
    }

    // 2. 初始化成员变量
    m_windowSize = lagWindow;
    m_version = 0;

    // 初始化窗口容器
    m_windows.resize(stocksNum);

    // 3. 创建通用权重缓存（共享）
    const size_t windowSize = static_cast<size_t>(lagWindow);
    Ve ewmVarWeightsVec(std_period*2);
    ewmVarWeightsVec.setConstant(1.0);

    Ve ewmWeightsVec((windowSize - rollingtime)*2);
    ewmWeightsVec.setConstant(1.0);

    auto weightCacheVar = OnlineBaseFactor::createOnlineBaseF<OnlineWeightCache>(ewmVarWeightsVec);
    auto weightCache = OnlineBaseFactor::createOnlineBaseF<OnlineWeightCache>(ewmWeightsVec);


    // 4. 为每只股票创建在线计算组件
    for (int i = 0; i < stocksNum; ++i)
    {
        auto& w = m_windows[i];

        // 初始化数据缓存（成交额、收盘价）
        w.m_onlineDataCacheAmt   = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initAmt.col(i));
        w.m_onlineDataCacheClose = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initClose.col(i));

        // 初始化加信号（add_signal）队列
        w.m_addSignal.clear();
        for (int k = 0; k < lagWindow; ++k)
            w.m_addSignal.emplace_back(0.0);

        // 初始化 UPDO 队列
        w.m_UPDO.clear();
        for (int k = 0; k < lagWindow; ++k)
            w.m_UPDO.emplace_back(0.0);

        // 差分序列
        Ve diffAmt;
        BusinessFactor::to_diff_sequence(initAmt.col(i), diffAmt);

        // 初始化在线方差和偏度
        // 初始化窗口方差
        Ve initValuesAmt = initAmt.col(i).tail(std_period).eval();
        auto cacheValues = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initValuesAmt);
        auto sumValues = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSum>(OnlineEWMSum::Window{weightCacheVar, cacheValues});
        auto meanValues = OnlineBaseFactor::createOnlineBaseF<OnlineEWMMean>(OnlineEWMMean::Window{weightCacheVar, sumValues});
        auto prodValues = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(OnlineEWMSumProduct::Window{weightCacheVar, cacheValues, cacheValues});
        w.m_onlineEWMVar = OnlineBaseFactor::createOnlineBaseF<OnlineEWMVar>(
            OnlineEWMVar::Window{weightCacheVar, meanValues, prodValues}
        );

        // 初始化偏度计算器（Skew）
        Ve initValuesClose(m_windowSize - rollingtime);
        initValuesClose.setConstant(0);
        auto cacheValuesClose = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initValuesClose);
        auto sumValuesClose = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSum>(OnlineEWMSum::Window{weightCache, cacheValuesClose});
        auto meanValuesClose = OnlineBaseFactor::createOnlineBaseF<OnlineEWMMean>(OnlineEWMMean::Window{weightCache, sumValuesClose});
        auto prodValuesClose = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(OnlineEWMSumProduct::Window{weightCache, cacheValuesClose, cacheValuesClose});
        auto prod3ValuesClose = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct3>(OnlineEWMSumProduct3::Window{weightCache, cacheValuesClose, cacheValuesClose, cacheValuesClose});
        w.m_onlineEWMSkew = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSkew>(
            OnlineEWMSkew::Window{weightCache, meanValuesClose, prodValuesClose, prod3ValuesClose}
        );
    }

    // 5. 初始化 skew 值缓存
    m_valueSkew.resize(stocksNum);
    for (auto& v : m_valueSkew) {
        v = 0.0;
    }

    std::cout << "Init 完成：窗口大小=" << lagWindow
              << " 股票数=" << stocksNum << std::endl;

    return 0;
}

void m_vpc_mut_ty_log1::Update(const Ma& newAmt, const Ma& newClose)
{
    Ma amt = newAmt;
    Ma close = newClose;

    for(int i = 0; i < amt.cols(); i++){
        Ve inSkewValues = Ve::Constant(amt.rows(), 0.0);
        for(int j = 0; j < amt.rows(); j++){
            m_version ++;
            //构造增量
            Ve inValueAmt = amt.col(i).segment(j, 1);
            //成交额差分
            double minuteTurnoverDiff = inValueAmt[0] - m_windows[i].m_onlineDataCacheAmt->getValues()[m_windowSize-1];
            //update容器
            m_windows[i].m_onlineDataCacheAmt->update(inValueAmt, m_version);

            //update方法层
            m_windows[i].m_onlineEWMVar->update(inValueAmt, m_version);
            //add_signal
            double addSignal = minuteTurnoverDiff / std::sqrt(m_windows[i].m_onlineEWMVar->getValue());
            if(addSignal > 2.0){
                m_windows[i].m_addSignal.pop_front();
                m_windows[i].m_addSignal.push_back(1.0);
                inSkewValues[j] = m_windows[i].m_addSignal[m_windowSize - rollingtime - 1];
            }else{
                m_windows[i].m_addSignal.pop_front();
                m_windows[i].m_addSignal.push_back(0.0);
                inSkewValues[j] = m_windows[i].m_addSignal[m_windowSize - rollingtime - 1];
            }
        }
        //UPDO
        //update容器
        const auto& cacheValues = m_windows[i].m_onlineDataCacheClose->getValues();
        for(size_t j = 0; j < close.rows(); j++){
            int index = m_windowSize - rollingtime;
            m_windows[i].m_UPDO[index] = abs(cacheValues[index] - close.col(i)[j]);
            m_windows[i].m_onlineDataCacheClose->update(close.col(i).segment(j, 1), m_version);
            inSkewValues[j] *= m_windows[i].m_UPDO[index];
            
            m_windows[i].m_UPDO.emplace_back(std::numeric_limits<double>::quiet_NaN());
            m_windows[i].m_UPDO.pop_front();
            m_version ++;
        }
        //更新skew
        m_windows[i].m_onlineEWMSkew->update(inSkewValues, m_version);
        m_valueSkew[i] = -1*m_windows[i].m_onlineEWMSkew->getValue();
    }
    // debug
    // static int update_num = 0;
    // update_num ++;
    // if(update_num == 9){
    //     std::cout << "MinuteTurnover: " << std::endl;
    //     for(size_t j = 0; j < amt.rows(); j++){
    //         std::cout << amt.col(0)[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "MinuteClose: " << std::endl;
    //     for(size_t j = 0; j < close.rows(); j++){
    //         std::cout << close.col(0)[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "UPDO: " << std::endl;
    //     for(size_t j = 0; j < m_windows[0].m_UPDO.size(); j++){
    //         std::cout << m_windows[0].m_UPDO[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "add_signal: " << std::endl;
    //     for(size_t j = 0; j < m_windows[0].m_addSignal.size(); j++){
    //         std::cout << m_windows[0].m_addSignal[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}
 
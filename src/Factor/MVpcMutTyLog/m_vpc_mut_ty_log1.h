//创建一个应用，继承Application
#include "../../OnlineBaseFactor/OnlineEWMMethod.h"
#include "../../OnlineBaseFactor/BusinessFactor/businessfactor.h"
#include "../../DataProcess/DataProcess.h"
#include "../../include/TradeConstants.h"
#include "../../Eigen_extra/RingVec.h"
#include "../../OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include "OnlineBaseFactor/OnlineDataCache.h"
#include <cstddef>
#include <deque>
#include <memory>
class m_vpc_mut_ty_log1
{
public:
    m_vpc_mut_ty_log1(/* args */);
    ~m_vpc_mut_ty_log1();

    //初始化的函数
    virtual int Init(const Ma& initAmt, const Ma& initClose) ;
    //
    virtual void Update(const Ma& newAmt, const Ma& newClose) ;

    //结束时执行的函数
    virtual void Finish() ;

    //获取因子值结果
    const Ve& getValue() const { return m_valueSkew; }

    struct Window{
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheAmt;
        std::shared_ptr<OnlineDataCache> m_onlineDataCacheClose;

        std::shared_ptr<OnlineEWMVar> m_onlineEWMVar;
        std::shared_ptr<OnlineEWMSkew> m_onlineEWMSkew;
        
        std::deque<double> m_addSignal;
        std::deque<double> m_UPDO;
    };
protected:
    //接收数据的函数

private: 
    Ve m_valueSkew;
    size_t m_version = 0;
    size_t m_windowSize = 0;
    std::vector<Window> m_windows;  // 为每只股票维护独立的窗口
    //魔法配置
    int std_period = 5;
    int rollingtime = 2;
};


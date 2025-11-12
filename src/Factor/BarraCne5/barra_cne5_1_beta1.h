//创建一个应用，继承Application
#include "../../OnlineBaseFactor/OnlineDataCache.h"
#include "../../OnlineBaseFactor/OnlineEWMMethod.h"
#include "../../DataProcess/DataProcess.h"
#include "../../include/TradeConstants.h"
#include "../../Eigen_extra/RingVec.h"
#include "../../OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include <vector>
class barra_cne5_1_beta1
{
public:
    barra_cne5_1_beta1(/* args */);
    ~barra_cne5_1_beta1();

    //初始化的函数
    virtual int Init(const Ma& ret, const Ma& cap, const Ve& valid) ;
    //
    virtual void Update(const Ma& ret, const Ma& cap, const Ve& valid) ;

    //结束时执行的函数
    virtual void Finish() ;

    //获取beta值结果
    const Ve& getValue() const { return m_value; }

    struct Window{
        std::shared_ptr<OnlineEWMVar> m_onlineEWMVarX;
        std::shared_ptr<OnlineEWMCov> m_onlineEWMCovXY;
    };
protected:
    //接收数据的函数
    void RecvSHStockData();

private: 
    Ve m_value;
    size_t m_version = 0;
    std::vector<Window> m_windows;  // 为每只股票维护独立的窗口
};


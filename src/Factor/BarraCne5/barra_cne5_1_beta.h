//创建一个应用，继承Application
#include "../../OnlineBaseFactor/OnlineDataCache.h"
#include "../../OnlineBaseFactor/OnlineMethod.h"
#include "../../DataProcess/DataProcess.h"
#include "../../include/TradeConstants.h"
#include "../../Eigen_extra/RingVec.h"
#include "../../OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include <vector>
class barra_cne5_1_beta
{
public:
    barra_cne5_1_beta(/* args */);
    ~barra_cne5_1_beta();

    //初始化的函数
    virtual int Init(const Ma& ret, const Ma& cap, const Ve& valid) ;
    //
    virtual void Update(const Ma& ret, const Ma& cap, const Ve& valid) ;
    //结束时执行的函数
    virtual void Finish() ;

    //获取beta值结果
    const Ve& getValue() const { return m_value; }

    struct Window{
        std::shared_ptr<OnlineVar> m_onlineVarX;
        std::shared_ptr<OnlineVar> m_onlineVarY;
        std::shared_ptr<OnlineCorrelation> m_onlineCorrelation;
    };
protected:
    //接收数据的函数
    void RecvSHStockData();

private: 
    Ve m_value;
    size_t m_version = 0;
    std::vector<Window> m_windows;  // 为每只股票维护独立的窗口
};



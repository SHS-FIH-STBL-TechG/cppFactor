//创建一个应用，继承Application
#include "../../Eigen_extra/Eigen_extra.h"
#include "OnlineBaseFactor/OnlineDataCache.h"
#include "OnlineBaseFactor/OnlineMethod.h"
#include "OnlineBaseFactor/BusinessFactor/businessfactor.h"
#include "OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include <cstddef>
#include <vector>
#include <memory>
class m_vpc_mut_ty_log2
{
public:
    m_vpc_mut_ty_log2(/* args */);
    ~m_vpc_mut_ty_log2();

    //初始化的函数
    virtual int Init(const Ma& initAmt, const Ma& initClose) ;
    //
    virtual void Update(const Ma& newAmt, const Ma& newClose) ;

    //结束时执行的函数
    virtual void Finish() ;

    //获取因子值结果
    const Ve& getValue() const { return m_valueCoef; }

    struct Window{
        Ve cacheDataAmt;
        Ve cacheDataClose;
    };
protected:
    //接收数据的函数

private: 
    Ve m_valueCoef;
    size_t m_stockCount = 0;
    size_t m_version = 0;
    size_t m_windowSize = 0;
    std::vector<Window> m_windows;  // 为每只股票维护独立的窗口
};


#pragma once
#include "../include/TradeConstants.h"
#include "../OnlineBaseFactor/OnlineBaseFactor.h"

//创建一个应用，继承Application

using namespace Tool;
using namespace Eigen;

class AppDemo
{
public:
    AppDemo(/* args */);
    ~AppDemo();

    //重写，初始化的函数（订阅表等初始化）
    virtual int Init();
    //重写，更新函数
    virtual void Update();
    //重写，结束时执行的函数
    virtual void Finish();

protected:

private:

};

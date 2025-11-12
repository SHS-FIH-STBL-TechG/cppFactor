#pragma once
#include <functional>
#include "AppDemo.h"
#include "CreatedDataAction.h"
#include "AppTools.h"

AppDemo::AppDemo(/* args */)
{
}

AppDemo::~AppDemo()
{
}

int AppDemo::Init()
{
    bool ret;
    //订阅表数据，把自己定义的函数作为回调函数传进去
    ret = SubscribDataAction("SnapshotSHStock", std::bind(&AppDemo::RecvSHStockData, this, std::placeholders::_1), _idInfo);
    if (ret == false)
    {
        //如果订阅失败，可能是表没找到，也可能是表里没有idInfo的读权限
        //初始化失败，返回-1
        return -1;
    }

    //ont数据的表
    ret = SubscribDataAction("OrdAndExeInfo", std::bind(&AppDemo::RecvOntData, this, std::placeholders::_1), _idInfo);
    if (ret == false)
    {
        return -1;
    }

    _setKLine = SetterPositionOfDataAction(35, _idInfo);
    if (_setKLine == NULL)
    {
        return -1;
    }

    return 0;
}

void AppDemo::RecvSHStockData(GetterDtAct * getter)
{
    //获得某个字段的数据
    bool ret;
    uint32_t pxLast;
    uint64_t qtyBid01;
    //通过biref描述字符串，获得某个字段
    ret = getter->GetLastDataByKey("LastPrice", &pxLast);
    //返回值是false，可能是没有找到这个字段
    if (ret == false)
    {
        printf("get field lastPrice failed\n");
    }

    //通过gkid获得某个字段的值
    getter->GetLastDataByKey(1003, &qtyBid01);

    //...

    return;
}

void AppDemo::RecvOntData(GetterDtAct * getter)
{
    //收到ont数据，触发了这个回调函数

    //...

    //经过计算，算出了一些特征值
    uint32_t highPric = 2360;
    uint32_t lowPric = 2310;
    uint64_t totQty = 5000;

    //传给这个DataAction
    _setKLine->SetLastDataByKey("highPrc", highPric);
    _setKLine->SetLastDataByKey("lowPx", lowPric);
    //。。
   
    //如果新的一条数据所有字段都传入了，就可以触发DataAction里的回调函数
    _setKLine->NewDataGenerated();

    return;
}
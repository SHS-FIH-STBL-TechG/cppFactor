#include "Application.h"

//创建一个应用，继承Application

class AppDemo : public Application
{
public:
    AppDemo(/* args */);
    ~AppDemo();

    //重写，初始化的函数
    virtual int Init() override;
    //重写，结束时执行的函数
    virtual void Finish() override;

protected:
    //接收数据的函数
    void RecvSHStockData(GetterDtAct * getter);
    void RecvOntData(GetterDtAct * getter);

private:
    SetterDtAct *_setKLine;

};



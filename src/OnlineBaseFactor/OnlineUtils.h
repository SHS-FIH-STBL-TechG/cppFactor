#pragma once
#include <iostream>
#include <cstdlib>

// 辅助宏：简化构造函数中的空指针检测
#define CHECK_NULLPTR(className, memberPtr) \
    if(!memberPtr){ \
        std::cout << className << ": " << #memberPtr << " 未实例化\n"; \
        exit(1); \
    }


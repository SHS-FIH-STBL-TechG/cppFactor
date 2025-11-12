#ifndef TOOL_H
#define TOOL_H

// 工具类统一头文件
// 包含所有工具类的头文件

#include "timestamp.h"
#include "config_reader.h"
#include "database.h"
#include "profiler.h"
// 工具类命名空间
namespace Tool {
    // 所有工具类都在此命名空间下
    // 使用方式：
    // Tool::Timestamp::getCurrentTimestamp()
    // Tool::ConfigReader::loadConfig()
    // Tool::Database::相关方法
    // Tool::Profiler::getInstance() - 性能分析
}

#endif // TOOL_H

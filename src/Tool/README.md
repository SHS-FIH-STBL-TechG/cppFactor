# Tool 工具类模块

## 概述
本模块整合了项目中的通用工具类，包括时间戳、数据库操作和配置读取功能，统一放在 `Tool` 命名空间下，保持代码结构清晰明了。

## 文件结构

```
src/tool/
├── tool.h              # 统一头文件，包含所有工具类
├── timestamp.h         # 时间戳工具类
├── config_reader.h     # 配置文件读取工具类
├── database.h          # 数据库接口和实现
├── database.cpp        # 数据库实现
├── database_example.cpp # 数据库使用示例
├── test_tool.cpp       # 工具类测试程序
└── README.md           # 本说明文档
```

## 使用方法

### 1. 时间戳工具 (Timestamp)

```cpp
#include "../tool/tool.h"

// 获取完整时间戳 (YYYY-MM-DD HH:MM:SS.mmm)
std::string timestamp = Tool::Timestamp::getCurrentTimestamp();

// 获取简单时间戳 (HH:MM:SS.mmm)
std::string simple = Tool::Timestamp::getSimpleTimestamp();

// 获取日期戳 (YYYY-MM-DD)
std::string date = Tool::Timestamp::getDateStamp();
```

### 2. 配置读取工具 (ConfigReader)

```cpp
#include "../tool/tool.h"

std::map<std::string, std::string> config;
bool success = Tool::ConfigReader::loadConfig("config.ini", config);

// 获取配置值
std::string value = Tool::ConfigReader::getValue(config, "key", "default_value");
```

### 3. 数据库工具 (Database)

```cpp
#include "../tool/tool.h"

// 创建内存数据库
Tool::MemoryDatabase database;

// 从CSV文件加载数据
database.loadFromCSV("data_name", "path/to/file.csv");

// 获取矩阵数据
Eigen::MatrixXd data = database.getMatrix("data_name");
```

## 命名空间

所有工具类都在 `Tool` 命名空间下，避免命名冲突：

- `Tool::Timestamp` - 时间戳相关功能
- `Tool::ConfigReader` - 配置读取功能  
- `Tool::Database` - 数据库接口
- `Tool::MemoryDatabase` - 内存数据库实现

## 集成说明

### 在测试用例中使用

```cpp
#include "../tool/tool.h"

class MyFactor : public BaseFactor {
public:
    VectorXd calc_single(Tool::Database& database) override {
        // 使用时间戳
        std::cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 开始计算" << std::endl;
        
        // 使用数据库
        MatrixXd data = database.getMatrix("test_data");
        
        // 计算逻辑...
        
        std::cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 计算完成" << std::endl;
        return result;
    }
};
```

### 在BaseFactor中使用

BaseFactor类已经更新为使用 `Tool::Database` 接口：

```cpp
virtual VectorXd calc_single(Tool::Database& database) = 0;
```

## 优势

1. **结构清晰**: 所有工具类统一管理，便于维护
2. **命名空间隔离**: 避免命名冲突
3. **统一接口**: 通过 `tool.h` 一次性引入所有工具
4. **易于扩展**: 新增工具类只需添加到tool文件夹
5. **向后兼容**: 保持原有功能不变，只是重新组织

## 注意事项

1. 所有测试用例需要重新编译以使用新的结构
2. 确保包含路径正确指向 `../tool/tool.h`
3. 使用 `Tool::` 前缀访问所有工具类
4. 时间戳功能需要C++11或更高版本支持

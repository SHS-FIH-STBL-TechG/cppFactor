# Miner - 金融因子挖掘系统

## 项目概述

Miner是一个专门用于金融因子挖掘和计算的C++系统，支持多种统计方法和时间序列分析。系统采用模块化设计，提供高效的数值计算和数据处理能力。

## 核心特性

### 🔢 **数学统计方法**
- **描述性统计**: 均值、标准差、偏度、峰度、最值

### ⚡ **性能优化**
- **NaN安全**: 所有函数都处理缺失值，确保计算稳定性
- **数值稳定**: 采用适当的数值方法避免溢出
- **内存优化**: 高效的矩阵运算和内存管理

### 📊 **数据处理**
- **CSV支持**: 自动读取和解析CSV格式数据
- **多维矩阵**: 支持多股票时间序列数据处理
- **配置驱动**: 通过配置文件灵活控制计算参数
- **结果输出**: 标准化的CSV格式结果输出

## 项目结构

```
Miner/
├── src/                    # 源代码
│   ├── database/          # 数据库模块
│   │   ├── database.h     # 数据库接口定义
│   │   ├── database.cpp   # 数据库实现
│   │   └── database_example.cpp  # 数据库使用示例
│   ├── factorbase/        # 因子计算基础库
│   │   ├── factor_base.h  # 因子基类定义
│   │   ├── descriptive_stats.cpp      # 描述性统计
│   │   ├── correlation_analysis.cpp   # 相关性分析（未使用）
│   │   ├── nonparametric_stats.cpp    # 非参数统计（未使用）
│   │   ├── signal_processing.cpp      # 信号处理（未使用）
│   │   └── config_reader.h           # 配置读取器
│   └── factor_/           # 具体因子实现
│       ├── 0001_skew.cpp  # 偏度因子
│       ├── 0002_kurt.cpp  # 峰度因子
│       ├── 0003_max.cpp   # 最大值因子
│       ├── 0004_mean.cpp  # 均值因子
│       ├── 0005_min.cpp   # 最小值因子
│       └── 0006_std.cpp   # 标准差因子
├── testcase/              # 测试案例
│   ├── 0001_skew/        # 偏度因子测试
│   ├── 0002_kurt/        # 峰度因子测试
│   ├── 0003_max/         # 最大值因子测试
│   ├── 0004_mean/        # 均值因子测试
│   ├── 0005_min/         # 最小值因子测试
│   ├── 0006_std/         # 标准差因子测试
│   └── genTestData.py    # 测试数据生成器
├── bin/                   # 编译输出
│   ├── 0001_skew.exe     # 偏度因子可执行文件
│   ├── 0002_kurt.exe     # 峰度因子可执行文件
│   ├── 0003_max.exe      # 最大值因子可执行文件
│   ├── 0004_mean.exe     # 均值因子可执行文件
│   ├── 0005_min.exe      # 最小值因子可执行文件
│   ├── 0006_std.exe      # 标准差因子可执行文件
│   ├── database_example.exe  # 数据库示例
│   └── config.ini        # 配置文件
├── lib/                   # 第三方库
│   └── Eigen/            # Eigen线性代数库
├── docs/                  # 文档
│   ├── README.md         # 项目说明
│   ├── PROJECT_STRUCTURE.md  # 项目结构详解
│   ├── CONFIG_USAGE.md   # 配置使用说明
│   ├── FACTOR_CONFIG_UPDATE.md  # 因子配置更新
│   ├── MATHEMATICAL_CLASSIFICATION.md  # 数学分类
│   ├── UPDATE_CONFIG_STRICT.md  # 配置更新严格模式
│   └── FACTOR_MATHEMATICAL_FORMULAS.xlsx  # 因子数学公式
├── build/                 # 构建文件
├── py_/                   # Python脚本
├── script/                # 脚本文件
└── CMakeLists.txt         # CMake构建配置
```

## 快速开始

### 环境要求

- **C++17** 或更高版本
- **CMake 3.20** 或更高版本
- **Eigen3** 线性代数库（项目自带）
- **OpenMP**（可选，当前未使用）

### 编译构建

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake ..

# 编译
cmake --build . --config Release

# 或者使用ninja（如果可用）
cmake -G Ninja ..
ninja
```

### 运行示例

```bash
# 进入bin目录
cd bin

# 运行偏度因子计算
./0001_skew.exe

# 运行峰度因子计算
./0002_kurt.exe

# 运行均值因子计算
./0004_mean.exe
```

## 配置说明

系统使用 `config.ini` 文件进行配置，支持以下配置项：

```ini
[0001_skew]
input_csv = ../testcase/0001_skew/input.csv
output_csv = ../testcase/0001_skew/output.csv
precision = 6

[0002_kurt]
input_csv = ../testcase/0002_kurt/input.csv
output_csv = ../testcase/0002_kurt/output.csv
precision = 6
```

### 配置参数说明

- **input_csv**: 输入CSV文件路径
- **output_csv**: 输出CSV文件路径
- **precision**: 输出精度（小数位数）

## 数据格式

### 输入数据格式

支持单列和多列CSV格式：

**单列数据**（时间序列）：
```csv
2
7
3
4
6
```

**多列数据**（多股票）：
```csv
1.2,2.1,0.8
-0.5,1.3,-0.9
2.1,0.7,1.5
```

### 输出数据格式

```csv
factor,value
skewness,0.235514
```

## 因子说明

### 已实现的因子

| 因子ID | 因子名称 | 数学公式 | 描述 |
|--------|----------|----------|------|
| 0001 | 偏度 | Skewness = (n/((n-1)(n-2))) × Σ((xi - μ)/σ)³ | 衡量数据分布的对称性 |
| 0002 | 峰度 | Kurtosis = (1/n) × Σ((xi - μ)/σ)⁴ - 3 | 衡量数据分布的尖锐程度 |
| 0003 | 最大值 | Max = max(xi) | 时间序列的最大值 |
| 0004 | 均值 | μ = (1/n) × Σ(xi) | 时间序列的平均值 |
| 0005 | 最小值 | Min = min(xi) | 时间序列的最小值 |
| 0006 | 标准差 | σ = √[(1/(n-1)) × Σ(xi - μ)²] | 时间序列的波动性 |

### 因子计算逻辑

- **单列数据**: 正常计算该列（第一只股票）的特征
- **多列数据**: 显示警告信息，但只计算第一列（第一只股票）的特征
- **NaN处理**: 自动忽略NaN值，确保计算稳定性

## 测试案例

### 运行测试

```bash
# 生成测试数据
cd testcase
python genTestData.py

# 运行所有因子测试
cd ../bin
./0001_skew.exe
./0002_kurt.exe
./0003_max.exe
./0004_mean.exe
./0005_min.exe
./0006_std.exe
```

### 测试数据

- **输入**: 单列时间序列数据（5个数据点）
- **输出**: 对应的统计特征值
- **验证**: 与理论计算结果对比

## 性能特性

### 计算性能

- **高效计算**: 基于Eigen库的优化矩阵运算
- **内存效率**: 优化的矩阵操作，减少内存分配
- **数值稳定**: 采用数值稳定的算法实现

### 扩展性

- **模块化设计**: 易于添加新的因子计算方法
- **配置驱动**: 通过配置文件灵活控制计算参数
- **标准化接口**: 统一的输入输出格式

## 开发指南

### 添加新因子

1. 在 `src/factor_/` 目录下创建新的因子文件
2. 实现因子计算逻辑
3. 在 `CMakeLists.txt` 中添加编译目标
4. 创建对应的测试案例
5. 更新配置文件

### 代码规范

- 使用C++17标准
- 遵循Google C++代码风格
- 添加详细的注释说明
- 实现NaN安全处理

## 数学公式文档

详细的数学公式和实现说明请参考：
- `docs/FACTOR_MATHEMATICAL_FORMULAS.xlsx` - 完整的数学公式表

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

*最后更新: 2024年*
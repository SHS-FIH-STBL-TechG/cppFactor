cmake_minimum_required(VERSION 3.20)
project(FactorCalculation)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# 编译选项配置
# =============================================================================

# MSVC 运行时（UCRT），支持 /MD(d) 或 /MT(d)
option(USE_STATIC_MSVC_RUNTIME "Link with static MSVC runtime (/MT)" OFF)
if (MSVC)
    if (USE_STATIC_MSVC_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

# OpenMP（可选）
option(ENABLE_OPENMP "Enable OpenMP parallel processing" OFF)
find_package(OpenMP)
if(OpenMP_CXX_FOUND AND ENABLE_OPENMP)
    message(STATUS "OpenMP found, enabling parallel processing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP disabled for debugging")
endif()

# Eigen 并行计算开关
option(ENABLE_EIGEN_PARALLEL "Enable Eigen parallel computation" ON)
if(ENABLE_EIGEN_PARALLEL)
    add_definitions(-DEIGEN_DONT_PARALLELIZE=0)
    message(STATUS "Eigen parallel computation enabled")
else()
    add_definitions(-DEIGEN_DONT_PARALLELIZE=1)
    message(STATUS "Eigen parallel computation disabled")
endif()

# BLAS/LAPACK 后端选择
set(BLAS_BACKEND "NONE" CACHE STRING "BLAS/LAPACK backend (NONE, MKL, OPENBLAS)")
set_property(CACHE BLAS_BACKEND PROPERTY STRINGS "NONE" "MKL" "OPENBLAS")

if(BLAS_BACKEND STREQUAL "MKL")
    # 查找Intel MKL
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(MKL QUIET mkl)
    endif()
    
    if(NOT MKL_FOUND)
        # 尝试标准CMake查找
        find_library(MKL_CORE_LIB mkl_core PATHS
            /opt/intel/mkl/lib/intel64
            /opt/intel/oneapi/mkl/latest/lib/intel64
            "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64"
            "C:/Program Files/Intel/oneAPI/mkl/latest/lib/intel64"
            ENV MKLROOT
            PATH_SUFFIXES lib lib/intel64
        )
    endif()
    
    if(MKL_FOUND OR MKL_CORE_LIB)
        # 设置MKL宏定义
        add_definitions(-DEIGEN_USE_MKL_ALL)
        add_definitions(-DEIGEN_USE_LAPACKE_STRICT)
        
        # 链接MKL库
        if(MKL_LIBRARIES)
            link_libraries(${MKL_LIBRARIES})
        elseif(MKL_CORE_LIB)
            link_libraries(${MKL_CORE_LIB})
        endif()
        
        message(STATUS "Using Intel MKL backend")
    else()
        message(WARNING "MKL not found, falling back to default Eigen")
        set(BLAS_BACKEND "NONE")
    endif()
    
elseif(BLAS_BACKEND STREQUAL "OPENBLAS")
    # 查找OpenBLAS
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenBLAS QUIET openblas)
    endif()
    
    if(NOT OpenBLAS_FOUND)
        find_library(OpenBLAS_LIB openblas PATHS
            /usr/lib
            /usr/local/lib
            /opt/local/lib
            "C:/msys64/ucrt64/lib"
            "C:/msys64/mingw64/lib"
        )
    endif()
    
    if(OpenBLAS_FOUND OR OpenBLAS_LIB)
        # 设置OpenBLAS宏定义
        add_definitions(-DEIGEN_USE_OPENBLAS)
        
        # 链接OpenBLAS库
        if(OpenBLAS_LIBRARIES)
            link_libraries(${OpenBLAS_LIBRARIES})
        elseif(OpenBLAS_LIB)
            link_libraries(${OpenBLAS_LIB})
        endif()
        
        message(STATUS "Using OpenBLAS backend")
    else()
        message(WARNING "OpenBLAS not found, falling back to default Eigen")
        set(BLAS_BACKEND "NONE")
    endif()
    
else()
    message(STATUS "Using default Eigen backend (no external BLAS)")
endif()

# =============================================================================
# 编译优化设置
# =============================================================================

# 优化参数和编码设置
if (MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /utf-8 /Zi /DEBUG)  # Debug: 无优化，完整调试信息
    else()
        add_compile_options(/O2 /utf-8)  # Release: 优化
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g -ggdb3)      # Debug: 无优化，完整调试信息
    else()
        # Release: 最高优化（保持精度，不展开循环）
        # -O3: 最高优化级别
        # -march=native: 针对当前CPU架构优化
        # -mtune=native: 针对当前CPU调优
        # -finline-functions: 函数内联（提升性能，不影响精度）
        add_compile_options(-O3 -march=native -mtune=native -finline-functions)
    endif()
endif()

# 启用链接时优化（仅在Release模式下）
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    
    # 对于 GCC，确保链接器也使用 LTO
    if(NOT MSVC)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto=auto")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -flto=auto")
    endif()
endif()

# =============================================================================
# 并行编译设置
# =============================================================================

# 自动检测CPU核心数
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N})
    message(STATUS "设置并行编译线程数: ${N}")
else()
    set(CMAKE_BUILD_PARALLEL_LEVEL 4)
    message(STATUS "无法检测CPU核心数，设置默认并行编译线程数: 4")
endif()

# 设置Make并行编译
if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles" OR CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    set(CMAKE_MAKE_PROGRAM "make -j${CMAKE_BUILD_PARALLEL_LEVEL}")
endif()

# =============================================================================
# 路径和包含目录设置
# =============================================================================

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Eigen 头文件路径（项目内自带）
set(PROJECT_ROOT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR})
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/Eigen)
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
elseif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# 通用包含目录
# 将 src 设为包含根，保证诸如 "tool/tool.h"、"Eigen_extra/Eigen_extra.h" 可直接被找到
set(COMMON_INCLUDE_DIRS ${PROJECT_ROOT_INCLUDE} ${EIGEN3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

# =============================================================================
# 源文件定义
# =============================================================================

set(DATABASE_SOURCES
    src/Tool/database.cpp
)

set(BASE_SOURCES
    src/OnlineBaseFactor/BaseFactor/descriptive_stats.cpp
    src/OnlineBaseFactor/BaseFactor/correlation_analysis.cpp
    src/OnlineBaseFactor/BaseFactor/nonparametric_stats.cpp
    src/OnlineBaseFactor/BaseFactor/signal_processing.cpp
    src/OnlineBaseFactor/BaseFactor/linear_regression.cpp
)

set(BUSINESS_SOURCES
    src/OnlineBaseFactor/BusinessFactor/businessfactor.cpp
)

# =============================================================================
# 库目标定义
# =============================================================================

# 数据库库
add_library(database STATIC ${DATABASE_SOURCES})
target_include_directories(database PRIVATE ${COMMON_INCLUDE_DIRS})

# 基础因子库
add_library(base STATIC ${BASE_SOURCES})
target_include_directories(base PRIVATE ${COMMON_INCLUDE_DIRS})

# 业务因子库
add_library(business STATIC ${BUSINESS_SOURCES})
target_link_libraries(business base)
target_include_directories(business PRIVATE ${COMMON_INCLUDE_DIRS})

# =============================================================================
# 可执行文件定义
# =============================================================================


# 因子挖掘可执行文件列表
set(FACTOR_EXECUTABLES
    0001_skew:src/factor_case/0001_skew.cpp:database,base
    0002_kurt:src/factor_case/0002_kurt.cpp:database,base
    0003_max:src/factor_case/0003_max.cpp:database,base
    0004_mean:src/factor_case/0004_mean.cpp:database,base
    0005_min:src/factor_case/0005_min.cpp:database,base
    0006_std:src/factor_case/0006_std.cpp:database,base
    0007_scm:src/factor_case/0007_scm.cpp:database,base,business
    0008_ms:src/factor_case/0008_ms.cpp:database,base
    0009_tb:src/factor_case/0009_tb.cpp:database,base,business
    0010_dm:src/factor_case/0010_dm.cpp:database,base
    0011_srcm:src/factor_case/0011_srcm.cpp:database,base,business
    0012_trb:src/factor_case/0012_trb.cpp:database,base,business
    0013_rdm:src/factor_case/0013_rdm.cpp:database,base
    0014_ols:src/factor_case/0014_ols.cpp:database,base
    0015_ewm_sum:src/factor_case/0015_ewm_sum.cpp:database,base
    0016_ewm_sum_product:src/factor_case/0016_ewm_sum_product.cpp:database,base
    0017_ewm_mean:src/factor_case/0017_ewm_mean.cpp:database,base
    0018_ewm_var:src/factor_case/0018_ewm_var.cpp:database,base
    0019_ewm_cov:src/factor_case/0019_ewm_cov.cpp:database,base
    0020_ewm_sum_product3:src/factor_case/0020_ewm_sum_product3.cpp:database,base
    0021_ewm_skew:src/factor_case/0021_ewm_skew.cpp:database,base
    5001_cne5_1_beta:src/factor_case/5001_barra_cne5_1_beta.cpp:database,base
    5002_cne5_1_beta1:src/factor_case/5002_barra_cne5_1_beta1.cpp:database,base
    6001_m_vpc_mut_ty_log1:src/factor_case/6001_m_vpc_mut_ty_log1.cpp:database,base,business
    6002_m_vpc_mut_ty_log2:src/factor_case/6002_m_vpc_mut_ty_log2.cpp:database,base,business
    online_example:src/factor_case/online_example.cpp:database,base,business
)

set(ONLINE_EWM_EXECUTABLES
    0015_ewm_sum
    0016_ewm_sum_product
    0017_ewm_mean
    0018_ewm_var
    0019_ewm_cov
    0020_ewm_sum_product3
    0021_ewm_skew
)

# 批量创建因子挖掘可执行文件
foreach(FACTOR_INFO ${FACTOR_EXECUTABLES})
    string(REPLACE ":" ";" FACTOR_LIST ${FACTOR_INFO})
    list(GET FACTOR_LIST 0 EXEC_NAME)
    list(GET FACTOR_LIST 1 SOURCE_FILE)
    list(GET FACTOR_LIST 2 LINK_LIBS_STR)
    
    # 将链接库字符串转换为列表
    string(REPLACE "," ";" LINK_LIBS ${LINK_LIBS_STR})
    
    # 为5001添加额外的源文件
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE})
        message(WARNING "Source file not found for target ${EXEC_NAME}: ${SOURCE_FILE}, skipping")
        continue()
    endif()

    if(EXEC_NAME STREQUAL "5001_cne5_1_beta")
        add_executable(${EXEC_NAME} 
            ${SOURCE_FILE} 
            src/Factor/BarraCne5/barra_cne5_1_beta.cpp
            src/DataProcess/DataProcess.cpp
            src/OnlineBaseFactor/OnlineMethod.cpp
            src/OnlineBaseFactor/OnlineDataCache.cpp
            src/OnlineBaseFactor/OnlineBaseFactor.cpp
        )
    elseif(EXEC_NAME STREQUAL "5002_cne5_1_beta1")
        add_executable(${EXEC_NAME} 
            ${SOURCE_FILE} 
            src/Factor/BarraCne5/barra_cne5_1_beta1.cpp
            src/DataProcess/DataProcess.cpp
            src/OnlineBaseFactor/OnlineEWMMethod.cpp
            src/OnlineBaseFactor/OnlineDataCache.cpp
            src/OnlineBaseFactor/OnlineBaseFactor.cpp
            src/OnlineBaseFactor/BaseFactor/descriptive_stats.cpp
        )
    elseif(EXEC_NAME STREQUAL "6001_m_vpc_mut_ty_log1")
        add_executable(${EXEC_NAME}
            ${SOURCE_FILE}
            src/Factor/MVpcMutTyLog/m_vpc_mut_ty_log1.cpp
            src/DataProcess/DataProcess.cpp
            src/OnlineBaseFactor/OnlineEWMMethod.cpp
            src/OnlineBaseFactor/OnlineDataCache.cpp
            src/OnlineBaseFactor/OnlineBaseFactor.cpp
            src/OnlineBaseFactor/OnlineMethod.cpp
        )
    elseif(EXEC_NAME STREQUAL "6002_m_vpc_mut_ty_log2")
        add_executable(${EXEC_NAME}
            ${SOURCE_FILE}
            src/Factor/MVpcMutTyLog/m_vpc_mut_ty_log2.cpp
            src/DataProcess/DataProcess.cpp
            src/OnlineBaseFactor/OnlineEWMMethod.cpp
            src/OnlineBaseFactor/OnlineDataCache.cpp
            src/OnlineBaseFactor/OnlineBaseFactor.cpp
            src/OnlineBaseFactor/OnlineMethod.cpp
        )
    elseif(EXEC_NAME IN_LIST ONLINE_EWM_EXECUTABLES)
        add_executable(${EXEC_NAME}
            ${SOURCE_FILE}
            src/OnlineBaseFactor/OnlineEWMMethod.cpp
            src/OnlineBaseFactor/OnlineDataCache.cpp
            src/OnlineBaseFactor/OnlineBaseFactor.cpp
        )
    else()
        add_executable(${EXEC_NAME} ${SOURCE_FILE})
    endif()
    
    target_link_libraries(${EXEC_NAME} ${LINK_LIBS})
    target_include_directories(${EXEC_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
endforeach()

# =============================================================================
# 构建信息输出
# =============================================================================

message(STATUS "==========================================")
message(STATUS "Build Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  Eigen parallel: ${ENABLE_EIGEN_PARALLEL}")
message(STATUS "  BLAS backend: ${BLAS_BACKEND}")
if(OpenMP_CXX_FOUND AND ENABLE_OPENMP)
    message(STATUS "  OpenMP flags: ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "  Parallel build threads: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "==========================================")
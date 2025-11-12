#include "database.h"
// #include "m_vpc_mut_ty_log38.h"  // 暂时注释，等待因子实现
#include <iostream>
#include <memory>
#include <fstream>
#include <chrono>

using namespace Tool;

/**
 * Database使用示例
 * 展示如何使用不同类型的数据库进行因子计算
 */

void example_memory_database() {
    std::cout << "=== 内存数据库示例 ===" << std::endl;
    
    // 创建内存数据库
    auto db = DatabaseFactory::createMemoryDatabase();
    
    // 创建测试数据
    int rows = 1000, cols = 50;
    Eigen::MatrixXd amt_data = Eigen::MatrixXd::Random(rows, cols) * 1000.0;
    Eigen::MatrixXd limit_status = Eigen::MatrixXd::Zero(rows, cols);
    
    // 添加一些停牌数据
    for (int i = 0; i < rows; i += 100) {
        for (int j = 0; j < cols; j += 10) {
            limit_status(i, j) = 1.0;
        }
    }
    
    // 将数据添加到数据库
    db->addData(DataNames::AMT_MINUTE, amt_data);
    db->addData(DataNames::LIMIT_STATUS_MINUTE, limit_status);
    
    std::cout << "数据已添加到内存数据库" << std::endl;
    std::cout << "amt_minute 数据形状: (" << amt_data.rows() << ", " << amt_data.cols() << ")" << std::endl;
    std::cout << "limit_status 数据形状: (" << limit_status.rows() << ", " << limit_status.cols() << ")" << std::endl;
    
    // 测试数据访问
    try {
        auto retrieved_amt = db->getMatrix(DataNames::AMT_MINUTE);
        std::cout << "成功获取 amt_minute 数据" << std::endl;
        
        auto retrieved_limit = db->getMatrix(DataNames::LIMIT_STATUS_MINUTE);
        std::cout << "成功获取 limit_status 数据" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "数据访问错误: " << e.what() << std::endl;
    }
}

void example_cached_database() {
    std::cout << "\n=== 缓存数据库示例 ===" << std::endl;
    
    // 创建底层内存数据库
    auto memory_db = DatabaseFactory::createMemoryDatabase();
    
    // 添加测试数据
    int rows = 500, cols = 30;
    Eigen::MatrixXd test_data = Eigen::MatrixXd::Random(rows, cols);
    memory_db->addData("test_data", test_data);
    
    // 注意：缓存数据库功能已移除，直接使用内存数据库
    // auto cached_db = DatabaseFactory::createCachedDatabase(std::move(memory_db), 10);
    
    std::cout << "创建了内存数据库" << std::endl;
    
    // 多次访问同一数据，测试性能
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto data = memory_db->getMatrix("test_data");
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "第 " << (i+1) << " 次访问耗时: " << duration.count() << " 微秒" << std::endl;
    }
    
    std::cout << "数据库统计 - 数据项数量: " << memory_db->size() << std::endl;
}

void example_factor_calculation() {
    std::cout << "\n=== 因子计算示例 ===" << std::endl;
    
    // 创建测试数据库
    auto db = std::make_shared<MemoryDatabase>();
    
    // 生成测试数据
    MatrixXd amt_minute = MatrixXd::Random(1000, 50) * 1000.0 + MatrixXd::Constant(1000, 50, 100.0);
    MatrixXd limit_status = MatrixXd::Zero(1000, 50);
    
    db->addData("FactorData.Basic_factor.amt_minute", amt_minute);
    db->addData("FactorData.Basic_factor.limit_status_minute", limit_status);
    
    // 暂时注释因子计算，等待具体因子实现
    // m_vpc_mut_ty_log38 factor;
    
    try {
        // 计算因子
        // auto result = factor.calc_single(*db);
        
        std::cout << "因子计算功能已准备就绪!" << std::endl;
        std::cout << "等待具体因子实现..." << std::endl;
        
        // 应用reform
        // auto final_result = factor.reform(result);
        // std::cout << "Reform后结果长度: " << final_result.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "因子计算错误: " << e.what() << std::endl;
    }
}

void example_file_operations() {
    std::cout << "\n=== 文件操作示例 ===" << std::endl;
    
    // 创建测试数据
    Eigen::MatrixXd test_matrix(100, 20);
    test_matrix = Eigen::MatrixXd::Random(100, 20) * 100.0;
    
    // 保存到CSV文件
    try {
        // 保存为CSV格式（简化实现）
        std::ofstream file("test_data.csv");
        for (int i = 0; i < test_matrix.rows(); ++i) {
            for (int j = 0; j < test_matrix.cols(); ++j) {
                file << test_matrix(i, j);
                if (j < test_matrix.cols() - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
        std::cout << "数据已保存到 test_data.csv" << std::endl;
        
        // 保存到二进制文件（简化实现）
        std::ofstream bin_file("test_data.bin", std::ios::binary);
        int rows = test_matrix.rows();
        int cols = test_matrix.cols();
        bin_file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        bin_file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        bin_file.write(reinterpret_cast<const char*>(test_matrix.data()), rows * cols * sizeof(double));
        bin_file.close();
        std::cout << "数据已保存到 test_data.bin" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "文件保存错误: " << e.what() << std::endl;
    }
    
    // 从文件加载数据
    try {
        auto file_db = DatabaseFactory::createCSVFileDatabase(".");
        file_db->registerDataFile("loaded_data", "test_data.csv");
        
        auto loaded_data = file_db->getMatrix("loaded_data");
        std::cout << "从文件加载数据成功，形状: (" << loaded_data.rows() << ", " << loaded_data.cols() << ")" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "文件加载错误: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Database 使用示例程序" << std::endl;
    std::cout << "=====================" << std::endl;
    
    try {
        // 运行各种示例
        example_memory_database();
        example_cached_database();
        example_factor_calculation();
        example_file_operations();
        
        std::cout << "\n所有示例运行完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "程序运行错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

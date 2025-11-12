#include "database.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <limits>
#include <thread>
#include <cstdlib>
#include <cctype>

namespace Tool {

// MemoryDatabase实现
void MemoryDatabase::loadFromFile(const std::string& name, const std::string& filename) {
    // 这里可以实现从不同格式文件加载数据的逻辑
    // 目前提供一个基础框架
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 简单的CSV加载实现
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            // 去除空格
            cell.erase(std::remove_if(cell.begin(), cell.end(), [](unsigned char c) { return std::isspace(c); }), cell.end());
            
            if (cell.empty()) {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            
            try {
                row.push_back(std::stod(cell));
            } catch (const std::exception&) {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        data.push_back(row);
    }
    
    if (data.empty()) {
        throw std::runtime_error("Empty file: " + filename);
    }
    
    // 确保所有行都有相同的列数
    size_t max_cols = 0;
    for (const auto& row : data) {
        max_cols = std::max(max_cols, row.size());
    }
    
    // 转换为Eigen矩阵
    int rows = data.size();
    int cols = max_cols;
    MatrixXd matrix(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j < data[i].size()) {
                matrix(i, j) = data[i][j];
            } else {
                matrix(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    addData(name, matrix);
}

void MemoryDatabase::loadFromCSV(const std::string& name, const std::string& filename) {
    loadFromFile(name, filename);
}

void MemoryDatabase::loadFromCSVParallel(const std::vector<std::pair<std::string, std::string>>& file_pairs) {
    if (file_pairs.empty()) {
        return;
    }
    
    // 创建线程函数：加载单个CSV文件
    // 注意：每个线程写入不同的key，所以不需要加锁
    auto loadSingleFile = [this](const std::string& name, const std::string& filename) {
        try {
            // 加载数据（不直接写入data_map_，先存储在临时变量中）
            MatrixXd matrix = loadFromFileInternal(filename);
            
            // 直接写入，因为每个线程写入不同的key，不会冲突
            data_map_[name] = std::move(matrix);
        } catch (const std::exception& e) {
            std::cerr << "错误: 加载文件失败 " << filename << ": " << e.what() << std::endl;
            throw;
        }
    };
    
    // 创建线程向量
    std::vector<std::thread> threads;
    threads.reserve(file_pairs.size());
    
    // 为每个文件创建线程
    for (const auto& pair : file_pairs) {
        threads.emplace_back(loadSingleFile, pair.first, pair.second);
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
}

// MemoryDatabase内部辅助函数实现（优化版本）
MatrixXd MemoryDatabase::loadFromFileInternal(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 设置大缓冲区提升I/O性能
    constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1MB
    char* buffer = new char[BUFFER_SIZE];
    file.rdbuf()->pubsetbuf(buffer, BUFFER_SIZE);
    
    // 先读取第一行确定列数
    std::string first_line;
    if (!std::getline(file, first_line)) {
        delete[] buffer;
        throw std::runtime_error("Empty file: " + filename);
    }
    
    // 快速计算第一行的列数
    int cols = 1;
    for (char c : first_line) {
        if (c == ',') cols++;
    }
    
    // 计算行数（重置文件指针）
    file.clear();
    file.seekg(0, std::ios::beg);
    int rows = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) rows++;
    }
    
    // 重置文件指针，预分配矩阵
    file.clear();
    file.seekg(0, std::ios::beg);
    MatrixXd matrix(rows, cols);
    
    // 使用更高效的解析方法
    int row_idx = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // 直接解析字符串，避免 stringstream
        const char* start = line.c_str();
        const char* end = start + line.length();
        
        for (int col_idx = 0; col_idx < cols; ++col_idx) {
            // 跳过空格
            while (start < end && std::isspace(*start)) start++;
            
            if (start >= end || *start == ',') {
                matrix(row_idx, col_idx) = std::numeric_limits<double>::quiet_NaN();
                if (start < end) start++; // 跳过逗号
                continue;
            }
            
            // 找到下一个逗号或行尾
            const char* cell_end = start;
            while (cell_end < end && *cell_end != ',') cell_end++;
            
            // 使用 strtod 直接转换（比 stod 快，不需要异常处理）
            char* parse_end;
            double value = std::strtod(start, &parse_end);
            
            if (parse_end == start) {
                // 解析失败
                matrix(row_idx, col_idx) = std::numeric_limits<double>::quiet_NaN();
            } else {
                matrix(row_idx, col_idx) = value;
            }
            
            start = cell_end;
            if (start < end) start++; // 跳过逗号
        }
        row_idx++;
    }
    
    delete[] buffer;
    return matrix;
}

// CSVFileDatabase实现
MatrixXd CSVFileDatabase::getMatrix(const std::string& name) {
    auto it = name_to_file_.find(name);
    if (it == name_to_file_.end()) {
        throw std::runtime_error("Data not found: " + name);
    }
    
    std::string full_path = data_directory_ + "/" + it->second;
    return loadMatrixFromCSV(full_path);
}

std::pair<int, int> CSVFileDatabase::getDataShape(const std::string& name) const {
    auto it = name_to_file_.find(name);
    if (it == name_to_file_.end()) {
        return {-1, -1};
    }
    
    // 这里可以实现快速获取CSV文件维度的方法
    // 目前返回占位值
    return {-1, -1};
}

MatrixXd CSVFileDatabase::loadMatrixFromCSV(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            // 去除空格
            cell.erase(std::remove_if(cell.begin(), cell.end(), [](unsigned char c) { return std::isspace(c); }), cell.end());
            
            if (cell.empty()) {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            
            try {
                row.push_back(std::stod(cell));
            } catch (const std::exception&) {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    if (data.empty()) {
        throw std::runtime_error("Empty CSV file: " + filename);
    }
    
    // 确保所有行都有相同的列数
    size_t max_cols = 0;
    for (const auto& row : data) {
        max_cols = std::max(max_cols, row.size());
    }
    
    // 转换为Eigen矩阵
    int rows = data.size();
    int cols = max_cols;
    MatrixXd matrix(rows, cols);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j < data[i].size()) {
                matrix(i, j) = data[i][j];
            } else {
                matrix(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    return matrix;
}


// 工具函数：创建测试数据
namespace DatabaseUtils {
    /**
     * 创建测试用的内存数据库，包含常用的因子数据
     * @param rows 数据行数
     * @param cols 数据列数
     * @return std::unique_ptr<MemoryDatabase>
     */
    std::unique_ptr<MemoryDatabase> createTestDatabase(int rows = 1000, int cols = 100) {
        auto db = std::unique_ptr<MemoryDatabase>(new MemoryDatabase);
        
        // 生成随机测试数据
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1000.0);
        
        // 创建各种测试数据
        MatrixXd amt_data(rows, cols);
        MatrixXd close_data(rows, cols);
        MatrixXd volume_data(rows, cols);
        MatrixXd limit_status_data(rows, cols);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                amt_data(i, j) = dist(gen);
                close_data(i, j) = 10.0 + dist(gen) * 0.1;
                volume_data(i, j) = dist(gen) * 100;
                limit_status_data(i, j) = (dist(gen) < 50.0) ? 0.0 : 1.0; // 大部分为0，少数为1
            }
        }
        
        // 添加数据到数据库
        db->addData(DataNames::AMT_MINUTE, amt_data);
        db->addData(DataNames::CLOSE_MINUTE, close_data);
        db->addData(DataNames::VOLUME_MINUTE, volume_data);
        db->addData(DataNames::LIMIT_STATUS_MINUTE, limit_status_data);
        
        return db;
    }
}

} // namespace Tool

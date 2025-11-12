#ifndef DATABASE_H
#define DATABASE_H

#include <Eigen/Dense>
#include <string>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>
#include <mutex>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace Tool {

/**
 * 数据库接口类 - 用于因子计算中的数据访问
 * 提供统一的数据访问接口，支持从不同数据源获取矩阵数据
 */
class Database {
public:
    virtual ~Database() = default;
    
    /**
     * 获取指定名称的矩阵数据
     * @param name 数据名称，格式如 "FactorData.Basic_factor.amt_minute"
     * @return Eigen::MatrixXd 矩阵数据
     */
    virtual MatrixXd getMatrix(const std::string& name) = 0;
    
    /**
     * 检查数据是否存在
     * @param name 数据名称
     * @return bool 数据是否存在
     */
    virtual bool hasData(const std::string& name) const = 0;
    
    /**
     * 获取数据维度信息
     * @param name 数据名称
     * @return std::pair<int, int> (行数, 列数)
     */
    virtual std::pair<int, int> getDataShape(const std::string& name) const = 0;
};

/**
 * 内存数据库实现 - 将数据存储在内存中
 * 适用于测试和小规模数据
 */
class MemoryDatabase : public Database {
private:
    std::map<std::string, MatrixXd> data_map_;
    
    /**
     * 内部辅助函数：从文件加载矩阵（不写入data_map_）
     * @param filename 文件路径
     * @return MatrixXd 矩阵数据
     */
    MatrixXd loadFromFileInternal(const std::string& filename);
    
public:
    /**
     * 构造函数
     */
    MemoryDatabase() = default;
    
    /**
     * 添加数据到数据库
     * @param name 数据名称
     * @param data 矩阵数据
     */
    void addData(const std::string& name, const MatrixXd& data) {
        data_map_[name] = data;
    }
    
    /**
     * 从文件加载数据
     * @param name 数据名称
     * @param filename 文件路径
     */
    void loadFromFile(const std::string& name, const std::string& filename);
    
    /**
     * 从CSV文件加载数据
     * @param name 数据名称
     * @param filename CSV文件路径
     */
    void loadFromCSV(const std::string& name, const std::string& filename);
    
    /**
     * 并行从多个CSV文件加载数据（多线程）
     * @param file_pairs 数据名称和文件路径的配对列表
     */
    void loadFromCSVParallel(const std::vector<std::pair<std::string, std::string>>& file_pairs);
    
    // 实现Database接口
    MatrixXd getMatrix(const std::string& name) override {
        auto iterator = data_map_.find(name);
        if (iterator == data_map_.end()) {
            throw std::runtime_error("Data not found: " + name);
        }
        return iterator->second;
    }
    
    bool hasData(const std::string& name) const override {
        return data_map_.find(name) != data_map_.end();
    }
    
    std::pair<int, int> getDataShape(const std::string& name) const override {
        auto iterator = data_map_.find(name);
        if (iterator == data_map_.end()) {
            return {-1, -1};
        }
        return {iterator->second.rows(), iterator->second.cols()};
    }
    
    /**
     * 获取所有数据名称
     * @return std::vector<std::string> 数据名称列表
     */
    std::vector<std::string> getAllDataNames() const {
        std::vector<std::string> names;
        names.reserve(data_map_.size());
        for (const auto& pair : data_map_) {
            names.push_back(pair.first);
        }
        return names;
    }
    
    /**
     * 清空所有数据
     */
    void clear() {
        data_map_.clear();
    }
    
    /**
     * 获取数据数量
     * @return size_t 数据项数量
     */
    size_t size() const {
        return data_map_.size();
    }
};

/**
 * CSV文件数据库实现 - 从CSV文件读取数据
 */
class CSVFileDatabase : public Database {
private:
    std::string data_directory_;
    std::map<std::string, std::string> name_to_file_;
    
public:
    /**
     * 构造函数
     * @param data_directory 数据目录路径
     */
    explicit CSVFileDatabase(const std::string& data_directory) 
        : data_directory_(data_directory) {}
    
    /**
     * 注册CSV数据文件映射
     * @param name 数据名称
     * @param filename CSV文件名
     */
    void registerDataFile(const std::string& name, const std::string& filename) {
        name_to_file_[name] = filename;
    }
    
    // 实现Database接口
    MatrixXd getMatrix(const std::string& name) override;
    
    bool hasData(const std::string& name) const override {
        return name_to_file_.find(name) != name_to_file_.end();
    }
    
    std::pair<int, int> getDataShape(const std::string& name) const override;
    
private:
    /**
     * 从CSV文件读取矩阵数据
     * @param filename 文件路径
     * @return MatrixXd 矩阵数据
     */
    MatrixXd loadMatrixFromCSV(const std::string& filename) const;
};


/**
 * 数据库工厂类 - 用于创建不同类型的数据库实例
 */
class DatabaseFactory {
public:
    /**
     * 创建内存数据库
     * @return std::unique_ptr<MemoryDatabase>
     */
    static std::unique_ptr<MemoryDatabase> createMemoryDatabase() {
        return std::unique_ptr<MemoryDatabase>(new MemoryDatabase);
    }
    
    /**
     * 创建CSV文件数据库
     * @param data_directory 数据目录
     * @return std::unique_ptr<CSVFileDatabase>
     */
    static std::unique_ptr<CSVFileDatabase> createCSVFileDatabase(const std::string& data_directory) {
        return std::unique_ptr<CSVFileDatabase>(new CSVFileDatabase(data_directory));
    }
};

// 常用数据名称常量
namespace DataNames {
    const std::string AMT_MINUTE = "FactorData.Basic_factor.amt_minute";
    const std::string CLOSE_MINUTE = "FactorData.Basic_factor.close_minute";
    const std::string VOLUME_MINUTE = "FactorData.Basic_factor.volume_minute";
    const std::string HIGH_MINUTE = "FactorData.Basic_factor.high_minute";
    const std::string LOW_MINUTE = "FactorData.Basic_factor.low_minute";
    const std::string OPEN_MINUTE = "FactorData.Basic_factor.open_minute";
    const std::string LIMIT_STATUS_MINUTE = "FactorData.Basic_factor.limit_status_minute";
    const std::string CLOSE_INDEX_MINUTE = "FactorData.Basic_factor.close-index_minute";
    const std::string CITICS_INDCODE1 = "FactorData.Basic_factor.citics_indcode1";
    const std::string CITICS_INDCODE2 = "FactorData.Basic_factor.citics_indcode2";
    const std::string CLOSE = "FactorData.Basic_factor.close";
    const std::string MKT_CAP_ARD = "FactorData.Basic_factor.mkt_cap_ard";
    const std::string ADJFACTOR = "FactorData.Basic_factor.adjfactor";
    const std::string FREE_FLOAT_SHARES = "FactorData.Basic_factor.free_float_shares";
    const std::string ACTIVEBUYORDERVOL_MINUTE = "FactorData.Basic_factor.activebuyordervol_minute";
    const std::string ACTIVEBUYORDERAMT_MINUTE = "FactorData.Basic_factor.activebuyorderamt_minute";
    const std::string BUYTRADEVOL_MINUTE = "FactorData.Basic_factor.buytradevol_minute";
    const std::string TRADENUM_MINUTE = "FactorData.Basic_factor.tradenum_minute";
}

} // namespace Tool

#endif // DATABASE_H

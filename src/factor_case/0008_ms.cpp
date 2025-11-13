#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

// 包含项目头文件
#include "../OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include "../Tool/Tool.h"
#include "../Tool/config_reader.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::vector;
using std::endl;

class Factor_0008_MS : public BaseFactor {
public:
    VectorXd calc_single(Tool::Database& database) {
        // 读取输入数据
        MatrixXd data = database.getMatrix("test_data");
        
        // 读取配置参数
        Tool::ConfigReader config("config.ini");
        bool unbiased = config.getBool("0008_ms", "unbiased", true);  // 默认无偏
        
        cout << "=== 0008_MS 因子计算 ===" << endl;
        cout << "输入数据维度: " << data.rows() << " x " << data.cols() << endl;
        cout << "算法类型: " << (unbiased ? "无偏" : "有偏") << endl;
        
        // 计算均值/标准差比值（变异系数倒数）
        VectorXd result;
        BaseFactor::col_mean_over_std(data, result, unbiased);
        
        cout << "各列均值/标准差比值结果:" << endl;
        for (int i = 0; i < result.size(); ++i) {
            if (std::isnan(result[i])) {
                cout << "col_" << i << ": nan" << endl;
            } else {
                cout << "col_" << i << ": " << std::fixed << std::setprecision(6) << result[i] << endl;
            }
        }
        
        return result;
    }
};

int main() {
    try {
        // 读取配置文件
        Tool::ConfigReader config("config.ini");
        std::string input_csv = config.getString("0008_ms", "input_csv", "");
        std::string output_csv = config.getString("0008_ms", "output_csv", "");
        int precision = config.getInt("0008_ms", "precision", 6);
        
        if (input_csv.empty() || output_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少input_csv或output_csv路径" << std::endl;
            return 1;
        }
        
        // 创建因子实例
        Factor_0008_MS factor;
        
        // 创建数据库实例并读取数据
        Tool::MemoryDatabase database;
        database.loadFromCSV("test_data", input_csv);
        
        // 计算因子
        VectorXd result = factor.calc_single(database);
        
        // 输出结果到CSV文件
        std::ofstream output_file(output_csv);
        if (output_file.is_open()) {
            // 第一行写入时间戳
            output_file << "# Generated at: " << Tool::Timestamp::getCurrentTimestamp() << "\n";
            output_file << "factor,value\n";
            for (int i = 0; i < result.size(); ++i) {
                if (std::isnan(result[i])) {
                    output_file << "col_" << i << "_ms,nan\n";
                } else {
                    output_file << "col_" << i << "_ms," << std::fixed << std::setprecision(precision) << result[i] << "\n";
                }
            }
            output_file.close();
            cout << "\n[" << Tool::Timestamp::getCurrentTimestamp() << "] 结果已保存到: " << output_csv << endl;
        }
        
        cout << "\n=== 计算完成 ===" << endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

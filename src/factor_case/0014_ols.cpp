#include "../OnlineBaseFactor/BaseFactor/BaseFactor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include "../Tool/Tool.h"
#include "../Tool/config_reader.h"
#include "../Tool/timestamp.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

class Factor_0014_OLS : public BaseFactor {
public:
        VectorXd calc_single(Tool::Database& database) {
            // 读取输入数据
            MatrixXd x_data = database.getMatrix("x_data");
            VectorXd y_data = database.getMatrix("y_data").col(0);
            VectorXd result;
            Tool::ConfigReader config("config.ini");
            int method = config.getInt("0014_ols", "method", 0);
            bool fit_intercept = config.getBool("0014_ols", "fit_intercept", true);
            
            // 使用高精度计时器记录OLS求解耗时
            Tool::HighPrecisionTimer timer;
            timer.start();
            BaseFactor::ols_solve(x_data, y_data, result, fit_intercept, method);
            timer.stop();
            
            // 输出详细的耗时信息
            cout << "OLS求解耗时: " << timer.getFormattedElapsed() << endl;
            cout << "详细耗时: " << timer.getElapsedMicroseconds() << " μs" << endl;
            cout << "           " << timer.getElapsedMilliseconds() << " ms" << endl;
            cout << "           " << timer.getElapsedSeconds() << " s" << endl;
            
            return result;
        }
};
int main() {
    try {
        // 读取配置文件
        Tool::ConfigReader config("config.ini");
        std::string x_input_csv = config.getString("0014_ols", "x_input_csv", "");
        std::string y_input_csv = config.getString("0014_ols", "y_input_csv", "");
        std::string output_csv = config.getString("0014_ols", "output_csv", "");
        int precision = config.getInt("0014_ols", "precision", 6);
            
        if (output_csv.empty() || x_input_csv.empty() || y_input_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少output_csv或x_input_csv或y_input_csv路径" << std::endl;
            return 1;
        }
        
        // 创建因子实例
        Factor_0014_OLS factor;
        
        // 创建数据库实例并读取数据
        Tool::MemoryDatabase database;
        database.loadFromCSV("x_data", x_input_csv);
        database.loadFromCSV("y_data", y_input_csv);

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
                    output_file << "col_" << i << "_ols,nan\n";
                } else {
                    output_file << "col_" << i << "_ols," << std::fixed << std::setprecision(precision) << result[i] << "\n";
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



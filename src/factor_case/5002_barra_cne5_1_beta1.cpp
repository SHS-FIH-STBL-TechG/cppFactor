#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <climits>

// 包含项目头文件
#include "../tool/config_reader.h"
#include "../tool/timestamp.h"
#include "../tool/database.h"
#include "../tool/profiler.h"
#include "../Factor/BarraCne5/barra_cne5_1_beta1.h"
#include "../Eigen_extra/Eigen_extra.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

// 时间测量辅助函数
double getElapsedMs(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;  // 转换为毫秒
}

int main() {
    try {
        // 读取配置文件
        Tool::ConfigReader config("config.ini");
        
        // 读取输入文件路径
        std::string input_pct_chg = config.getString("5002_cne5_1_beta1", "input_pct_chg", "");
        std::string input_a_mkt_cap = config.getString("5002_cne5_1_beta1", "input_a_mkt_cap", "");
        std::string input_is_valid = config.getString("5002_cne5_1_beta1", "input_is_valid", "");
        std::string output_csv = config.getString("5002_cne5_1_beta1", "output_csv", "");
        
        // 读取配置参数
        int precision = config.getInt("5002_cne5_1_beta1", "precision", 6);
        int stepSize = config.getInt("5002_cne5_1_beta1", "stepSize", 1);
        int lagWindow = config.getInt("5002_cne5_1_beta1", "lagWindow", 100);
        int reformWindow = config.getInt("5002_cne5_1_beta1", "reformWindow", 1);
        
        if (input_pct_chg.empty() || input_a_mkt_cap.empty() || 
            input_is_valid.empty() || output_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少必要的输入/输出文件路径" << std::endl;
            return 1;
        }
        
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 5002_cne5_1_beta1 因子计算 ===" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 配置参数:" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   lagWindow = " << lagWindow << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   stepSize = " << stepSize << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   reformWindow = " << reformWindow << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   precision = " << precision << endl;
        
        // 读取输入数据（多线程并行加载）
        Tool::MemoryDatabase database;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 正在读取输入数据（多线程并行）..." << endl;
        
        {
            PROFILE_SCOPE("数据读取");
            // 使用多线程并行加载多个CSV文件
            std::vector<std::pair<std::string, std::string>> file_pairs = {
                {"pct_chg", input_pct_chg},
                {"a_mkt_cap", input_a_mkt_cap},
                {"is_valid", input_is_valid}
            };
            database.loadFromCSVParallel(file_pairs);
        }
        
        MatrixXd pct_chg = database.getMatrix("pct_chg");
        MatrixXd a_mkt_cap = database.getMatrix("a_mkt_cap");
        MatrixXd is_valid = database.getMatrix("is_valid");
        
        int num_timepoints = static_cast<int>(pct_chg.rows());
        int num_stocks = static_cast<int>(pct_chg.cols());
        
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 数据维度: " << num_timepoints << " x " << num_stocks << endl;
        
        // 验证数据维度一致性
        if (a_mkt_cap.rows() != num_timepoints || a_mkt_cap.cols() != num_stocks) {
            std::cerr << "错误: 市值数据维度不匹配: " << a_mkt_cap.rows() << " x " << a_mkt_cap.cols() << endl;
            return 1;
        }
        if (is_valid.rows() != num_timepoints || is_valid.cols() != num_stocks) {
            std::cerr << "错误: 有效性数据维度不匹配: " << is_valid.rows() << " x " << is_valid.cols() << endl;
            return 1;
        }
        
        // 检查数据量是否足够
        if (num_timepoints < lagWindow) {
            std::cerr << "错误: 时间点数(" << num_timepoints << ")小于初始化窗口大小(" << lagWindow << ")" << endl;
            return 1;
        }
        
        // 创建beta实例
        barra_cne5_1_beta1 beta;
        
        // 准备输出文件
        std::ofstream output_file(output_csv);
        if (!output_file.is_open()) {
            std::cerr << "错误: 无法创建输出文件: " << output_csv << std::endl;
            return 1;
        }
        
        // 写入CSV头部
        output_file << "# Generated at: " << Tool::Timestamp::getCurrentTimestamp() << "\n";
        output_file << "time_index,step,operation,elapsed_ms";
        for (int i = 0; i < num_stocks; ++i) {
            output_file << ",stock_" << i << "_beta";
        }
        output_file << "\n";
        
        // 初始化：使用前lagWindow个时间点的数据
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 开始初始化..." << endl;
        auto init_start = std::chrono::high_resolution_clock::now();
        
        Ma initRet = pct_chg.block(0, 0, lagWindow, num_stocks);
        Ma initCap = a_mkt_cap.block(0, 0, lagWindow, num_stocks);
        Ve initValid = is_valid.row(lagWindow - 1).transpose();  // 使用最后一个时间点的有效性，转换为列向量
        
        {
            PROFILE_SCOPE("barra_cne5_1_beta1::Init");
            int init_result = beta.Init(initRet, initCap, initValid);
        }
        double init_elapsed = getElapsedMs(init_start);
        
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 初始化完成，耗时: " << std::fixed << std::setprecision(3) << init_elapsed << " ms" << endl;
        
        // 输出初始化结果
        const Ve& init_beta = beta.getValue();
        output_file << lagWindow - 1 << ",0,init," << std::fixed << std::setprecision(3) << init_elapsed;
        for (int i = 0; i < num_stocks; ++i) {
            if (i < init_beta.size()) {
                if (std::isnan(init_beta[i])) {
                    output_file << ",nan";
                } else {
                    output_file << "," << std::fixed << std::setprecision(precision) << init_beta[i];
                }
            } else {
                output_file << ",nan";
            }
        }
        output_file << "\n";
        
        // 持续更新：按stepSize逐步处理后续数据
        int update_count = 0;
        int current_time = lagWindow;
        
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 开始持续更新..." << endl;
        
        while (current_time < num_timepoints) {
            // 计算本次更新的数据量（不超过stepSize）
            int update_size = std::min(stepSize, num_timepoints - current_time);
            
            // 提取本次更新的数据
            Ma updateRet = pct_chg.block(current_time, 0, update_size, num_stocks);
            Ma updateCap = a_mkt_cap.block(current_time, 0, update_size, num_stocks);
            Ve updateValid = is_valid.row(current_time + update_size - 1).transpose();  // 使用最后一个时间点的有效性，转换为列向量
            
            // 记录更新时间
            auto update_start = std::chrono::high_resolution_clock::now();
            
            // 调用Update
            {
                PROFILE_SCOPE("barra_cne5_1_beta1::Update");
                beta.Update(updateRet, updateCap, updateValid);
            }
            
            double update_elapsed = getElapsedMs(update_start);
            update_count++;
            
            // 输出更新结果
            const Ve& update_beta = beta.getValue();
            output_file << (current_time + update_size - 1) << "," << update_count << ",update," 
                       << std::fixed << std::setprecision(3) << update_elapsed;
            for (int i = 0; i < num_stocks; ++i) {
                if (i < update_beta.size()) {
                    if (std::isnan(update_beta[i])) {
                        output_file << ",nan";
                    } else {
                        output_file << "," << std::fixed << std::setprecision(precision) << update_beta[i];
                    }
                } else {
                    output_file << ",nan";
                }
            }
            output_file << "\n";
            
            // 每100次更新输出一次进度
            if (update_count % 100 == 0) {
                cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 更新进度: " 
                     << update_count << " 次, 当前时间点: " << (current_time + update_size - 1) 
                     << "/" << num_timepoints << ", 本次耗时: " 
                     << std::fixed << std::setprecision(3) << update_elapsed << " ms" << endl;
            }
            
            current_time += update_size;
        }
        
        output_file.close();
        
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 计算完成 ===" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 总更新次数: " << update_count << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 结果已保存到: " << output_csv << endl;
        
        // 输出性能分析报告
        Tool::Profiler::getInstance().printReport();
        
        // 导出性能报告到CSV
        std::string perf_report_csv = output_csv;
        size_t last_dot = perf_report_csv.find_last_of(".");
        if (last_dot != std::string::npos) {
            perf_report_csv = perf_report_csv.substr(0, last_dot) + "_performance.csv";
        } else {
            perf_report_csv += "_performance.csv";
        }
        Tool::Profiler::getInstance().exportToCSV(perf_report_csv);
        
        // 导出online层性能报告到单独的CSV文件
        std::string online_perf_report_csv = output_csv;
        last_dot = online_perf_report_csv.find_last_of(".");
        if (last_dot != std::string::npos) {
            online_perf_report_csv = online_perf_report_csv.substr(0, last_dot) + "_online_performance.csv";
        } else {
            online_perf_report_csv += "_online_performance.csv";
        }
        
        // 导出以"Online"或"Cache"开头的性能数据
        std::ofstream online_file(online_perf_report_csv);
        if (online_file.is_open()) {
            online_file << "函数名,调用次数,总耗时(ms),平均耗时(ms),最小耗时(ms),最大耗时(ms)\n";
            
            // 获取所有性能数据
            auto all_profiles = Tool::Profiler::getInstance().getAllProfiles();
            std::vector<std::pair<std::string, Tool::ProfileData>> online_profiles;
            
            for (const auto& pair : all_profiles) {
                const std::string& name = pair.first;
                // 检查是否以"Online"或"Cache"开头
                if (name.find("Online") == 0 || name.find("Cache") == 0) {
                    online_profiles.push_back(pair);
                }
            }
            
            // 按总耗时排序
            std::sort(online_profiles.begin(), online_profiles.end(),
                [](const auto& a, const auto& b) {
                    return a.second.total_us > b.second.total_us;
                });
            
            for (const auto& pair : online_profiles) {
                const auto& data = pair.second;
                online_file << data.name << ","
                           << data.call_count << ","
                           << std::fixed << std::setprecision(2) << data.total_us / 1000.0 << ","
                           << data.avg_us / 1000.0 << ","
                           << (data.min_us == LLONG_MAX ? 0 : data.min_us) / 1000.0 << ","
                           << data.max_us / 1000.0 << "\n";
            }
            
            online_file.close();
            std::cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] Online层性能报告已保存到: " << online_perf_report_csv << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


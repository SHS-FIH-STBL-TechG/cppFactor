#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>

// 项目头文件
#include "../Tool/config_reader.h"
#include "../Tool/timestamp.h"
#include "../Tool/database.h"
#include "../Tool/profiler.h"
#include "../Factor/MVpcMutTyLog/m_vpc_mut_ty_log2.h"
#include "../Eigen_extra/Eigen_extra.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

// 辅助函数：计算耗时（毫秒）
double getElapsedMs(const std::chrono::high_resolution_clock::time_point& start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;
}

int main() {
    try {
        // 读取配置
        Tool::ConfigReader config("config.ini");

        std::string input_amt = config.getString("6002_m_vpc_mut_ty_log2", "input_amt", "");
        std::string input_close = config.getString("6002_m_vpc_mut_ty_log2", "input_close", "");
        std::string output_csv = config.getString("6002_m_vpc_mut_ty_log2", "output_csv", "");

        int precision = config.getInt("6002_m_vpc_mut_ty_log2", "precision", 6);
        int stepSize = config.getInt("6002_m_vpc_mut_ty_log2", "stepSize", 1);
        int lagWindow = config.getInt("6002_m_vpc_mut_ty_log2", "lagWindow", 120);

        if (input_amt.empty() || input_close.empty() || output_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少必要的输入/输出路径" << std::endl;
            return 1;
        }

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 6002_m_vpc_mut_ty_log2 因子计算 ===" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 配置参数:" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   lagWindow = " << lagWindow << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   stepSize = " << stepSize << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "]   precision = " << precision << endl;

        // 加载输入数据
        Tool::MemoryDatabase database;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 正在加载输入数据..." << endl;
        {
            PROFILE_SCOPE("数据读取");
            database.loadFromCSVParallel({
                {"FactorData.Basic_factor.amt_minute", input_amt},
                {"FactorData.Basic_factor.close_adj_minute", input_close}
            });
        }

        MatrixXd amt = database.getMatrix("FactorData.Basic_factor.amt_minute");
        MatrixXd close = database.getMatrix("FactorData.Basic_factor.close_adj_minute");

        int num_timepoints = static_cast<int>(amt.rows());
        int num_stocks = static_cast<int>(amt.cols());

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 数据维度: " << num_timepoints << " x " << num_stocks << endl;

        if (close.rows() != num_timepoints || close.cols() != num_stocks) {
            std::cerr << "错误: 收盘价数据维度不匹配: " << close.rows() << " x " << close.cols() << endl;
            return 1;
        }

        if (num_timepoints < lagWindow) {
            std::cerr << "错误: 时间点数(" << num_timepoints << ")小于初始化窗口大小(" << lagWindow << ")" << std::endl;
            return 1;
        }

        m_vpc_mut_ty_log2 factor;

        std::ofstream output_file(output_csv);
        if (!output_file.is_open()) {
            std::cerr << "错误: 无法创建输出文件: " << output_csv << std::endl;
            return 1;
        }

        output_file << "# Generated at: " << Tool::Timestamp::getCurrentTimestamp() << "\n";
        output_file << "time_index,step,operation,elapsed_ms";
        for (int i = 0; i < num_stocks; ++i) {
            output_file << ",stock_" << i << "_mut_ty_log2";
        }
        output_file << "\n";

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 开始初始化..." << endl;
        auto init_start = std::chrono::high_resolution_clock::now();

        Ma initAmt = amt.block(0, 0, lagWindow, num_stocks);
        Ma initClose = close.block(0, 0, lagWindow, num_stocks);

        {
            PROFILE_SCOPE("m_vpc_mut_ty_log2::Init");
            int ret = factor.Init(initAmt, initClose);
            if (ret != 0) {
                std::cerr << "错误: Init 返回非零状态码: " << ret << std::endl;
                return 1;
            }
        }

        double init_elapsed = getElapsedMs(init_start);
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 初始化完成，耗时: " 
             << std::fixed << std::setprecision(3) << init_elapsed << " ms" << endl;

        const Ve& init_value = factor.getValue();
        output_file << lagWindow - 1 << ",0,init," << std::fixed << std::setprecision(3) << init_elapsed;
        for (int i = 0; i < num_stocks; ++i) {
            if (i < init_value.size()) {
                double value = init_value[i];
                if (std::isnan(value)) {
                    output_file << ",nan";
                } else {
                    output_file << "," << std::fixed << std::setprecision(precision) << value;
                }
            } else {
                output_file << ",nan";
            }
        }
        output_file << "\n";

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 开始持续更新..." << endl;
        int update_count = 0;
        int current_time = lagWindow;

        while (current_time < num_timepoints) {
            int update_size = std::min(stepSize, num_timepoints - current_time);

            Ma updateAmt = amt.block(current_time, 0, update_size, num_stocks);
            Ma updateClose = close.block(current_time, 0, update_size, num_stocks);

            auto update_start = std::chrono::high_resolution_clock::now();

            {
                PROFILE_SCOPE("m_vpc_mut_ty_log2::Update");
                factor.Update(updateAmt, updateClose);
            }

            double update_elapsed = getElapsedMs(update_start);
            update_count++;

            const Ve& update_value = factor.getValue();
            output_file << (current_time + update_size - 1) << "," << update_count << ",update,"
                        << std::fixed << std::setprecision(3) << update_elapsed;
            for (int i = 0; i < num_stocks; ++i) {
                if (i < update_value.size()) {
                    double value = update_value[i];
                    if (std::isnan(value)) {
                        output_file << ",nan";
                    } else {
                        output_file << "," << std::fixed << std::setprecision(precision) << value;
                    }
                } else {
                    output_file << ",nan";
                }
            }
            output_file << "\n";

            if (update_count % 100 == 0) {
                cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 更新进度: "
                     << update_count << " 次, 当前时间点: "
                     << (current_time + update_size - 1) << "/" << num_timepoints
                     << ", 本次耗时: "
                     << std::fixed << std::setprecision(3) << update_elapsed << " ms" << endl;
            }

            current_time += update_size;
        }

        factor.Finish();
        output_file.close();

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 计算完成 ===" << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 总更新次数: " << update_count << endl;
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 结果已保存到: " << output_csv << endl;

        Tool::Profiler::getInstance().printReport();

        std::string perf_report_csv = output_csv;
        size_t last_dot = perf_report_csv.find_last_of(".");
        if (last_dot != std::string::npos) {
            perf_report_csv = perf_report_csv.substr(0, last_dot) + "_performance.csv";
        } else {
            perf_report_csv += "_performance.csv";
        }
        Tool::Profiler::getInstance().exportToCSV(perf_report_csv);

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



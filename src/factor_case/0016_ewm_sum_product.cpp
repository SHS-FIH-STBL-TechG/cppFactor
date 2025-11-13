#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <cmath>

#include "../Tool/Tool.h"
#include "../OnlineBaseFactor/OnlineEWMMethod.h"

using Eigen::MatrixXd;
using EigenExtra::Ve;
using std::cout;
using std::endl;

namespace {

Ve columnToVector(const MatrixXd& matrix, int column) {
    std::vector<double> values;
    values.reserve(static_cast<size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        const double value = matrix(row, column);
        if (!std::isnan(value)) {
            values.push_back(value);
        }
    }
    Ve result(static_cast<int>(values.size()));
    for (int i = 0; i < result.size(); ++i) {
        result[i] = values[static_cast<size_t>(i)];
    }
    return result;
}

Ve rowToVector(const MatrixXd& matrix, int row) {
    std::vector<double> values;
    values.reserve(static_cast<size_t>(matrix.cols()));
    for (int col = 0; col < matrix.cols(); ++col) {
        const double value = matrix(row, col);
        if (!std::isnan(value)) {
            values.push_back(value);
        }
    }
    Ve result(static_cast<int>(values.size()));
    for (int i = 0; i < result.size(); ++i) {
        result[i] = values[static_cast<size_t>(i)];
    }
    return result;
}

}

int main() {
    try {
        Tool::ConfigReader config("config.ini");
        const std::string initial_x_csv = config.getString("0016_ewm_sum_product", "initial_x_csv", "");
        const std::string initial_y_csv = config.getString("0016_ewm_sum_product", "initial_y_csv", "");
        const std::string update_x_csv = config.getString("0016_ewm_sum_product", "update_x_csv", "");
        const std::string update_y_csv = config.getString("0016_ewm_sum_product", "update_y_csv", "");
        const std::string weight_csv = config.getString("0016_ewm_sum_product", "weight_csv", "");
        const std::string output_csv = config.getString("0016_ewm_sum_product", "output_csv", "");
        const int precision = config.getInt("0016_ewm_sum_product", "precision", 6);

        if (initial_x_csv.empty() || initial_y_csv.empty() || update_x_csv.empty() ||
            update_y_csv.empty() || weight_csv.empty() || output_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少必要的路径" << std::endl;
            return 1;
        }

        Tool::MemoryDatabase database;
        database.loadFromCSV("initial_x", initial_x_csv);
        database.loadFromCSV("initial_y", initial_y_csv);
        database.loadFromCSV("update_x", update_x_csv);
        database.loadFromCSV("update_y", update_y_csv);
        database.loadFromCSV("weight", weight_csv);

        MatrixXd initial_x = database.getMatrix("initial_x");
        MatrixXd initial_y = database.getMatrix("initial_y");
        MatrixXd update_x = database.getMatrix("update_x");
        MatrixXd update_y = database.getMatrix("update_y");
        MatrixXd weight_matrix = database.getMatrix("weight");

        if (initial_x.cols() == 0 || initial_y.cols() == 0 || weight_matrix.cols() == 0) {
            std::cerr << "错误: 初始数据或权重数据为空" << std::endl;
            return 1;
        }

        Ve initial_values_x = columnToVector(initial_x, 0);
        Ve initial_values_y = columnToVector(initial_y, 0);
        Ve weight_values = columnToVector(weight_matrix, 0);

        auto weight_cache = OnlineBaseFactor::createOnlineBaseF<OnlineWeightCache>(weight_values);
        auto data_cache_x = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values_x);
        auto data_cache_y = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values_y);
        auto ewm_sum_product = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(
            OnlineEWMSumProduct::Window{weight_cache, data_cache_x, data_cache_y});

        std::ofstream output_file(output_csv);
        if (!output_file.is_open()) {
            std::cerr << "错误: 无法写入输出文件: " << output_csv << std::endl;
            return 1;
        }

        output_file << "# Generated at: " << Tool::Timestamp::getCurrentTimestamp() << "\n";
        output_file << "step,operation,value\n";

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 0016_EWM_SUM_PRODUCT 测试 ===" << endl;
        cout << "初始窗口长度: " << initial_values_x.size() << endl;

        const double init_value = ewm_sum_product->getValue();
        output_file << 0 << ",init," << std::fixed << std::setprecision(precision) << init_value << "\n";
        cout << "初始带权乘积和: " << std::fixed << std::setprecision(precision) << init_value << endl;

        const int update_rows = std::min(update_x.rows(), update_y.rows());
        for (int row = 0; row < update_rows; ++row) {
            Ve update_values_x = rowToVector(update_x, row);
            Ve update_values_y = rowToVector(update_y, row);
            if (update_values_x.size() == 0 || update_values_y.size() == 0) {
                continue;
            }
            if (update_values_x.size() != update_values_y.size()) {
                std::cerr << "错误: 更新数据长度不一致" << std::endl;
                return 1;
            }

            const size_t version = static_cast<size_t>(row + 1);
            ewm_sum_product->update(update_values_x, update_values_y, version);
            const double current_value = ewm_sum_product->getValue();
            output_file << (row + 1) << ",update," << std::fixed << std::setprecision(precision) << current_value << "\n";
            cout << "更新 " << (row + 1) << " -> 带权乘积和: " << std::fixed << std::setprecision(precision) << current_value << endl;
        }

        output_file.close();
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 结果已写入: " << output_csv << endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


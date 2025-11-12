#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "../Tool/config_reader.h"
#include "../Tool/database.h"
#include "../Tool/timestamp.h"
#include "../OnlineBaseFactor/OnlineEWMMethod.h"

using Eigen::MatrixXd;
using EigenExtra::Ve;
using std::cout;

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
        const std::string initial_csv = config.getString("0021_ewm_skew", "initial_csv", "");
        const std::string update_csv = config.getString("0021_ewm_skew", "update_csv", "");
        const std::string weight_csv = config.getString("0021_ewm_skew", "weight_csv", "");
        const std::string output_csv = config.getString("0021_ewm_skew", "output_csv", "");
        const int precision = config.getInt("0021_ewm_skew", "precision", 6);

        if (initial_csv.empty() || update_csv.empty() || weight_csv.empty() || output_csv.empty()) {
            std::cerr << "错误: 配置文件中缺少必要的路径\n";
            return 1;
        }

        Tool::MemoryDatabase database;
        database.loadFromCSV("initial", initial_csv);
        database.loadFromCSV("update", update_csv);
        database.loadFromCSV("weight", weight_csv);

        MatrixXd initial_matrix = database.getMatrix("initial");
        MatrixXd update_matrix = database.getMatrix("update");
        MatrixXd weight_matrix = database.getMatrix("weight");

        if (initial_matrix.cols() == 0 || weight_matrix.cols() == 0) {
            std::cerr << "错误: 初始数据或权重数据为空\n";
            return 1;
        }

        Ve initial_values = columnToVector(initial_matrix, 0);
        Ve weight_values = columnToVector(weight_matrix, 0);

        auto weight_cache = OnlineBaseFactor::createOnlineBaseF<OnlineWeightCache>(weight_values);

        auto data_cache_sum = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto ewm_sum = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSum>(
            OnlineEWMSum::Window{weight_cache, data_cache_sum});
        auto ewm_mean = OnlineBaseFactor::createOnlineBaseF<OnlineEWMMean>(
            OnlineEWMMean::Window{weight_cache, ewm_sum});

        auto data_cache_prod_x = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto data_cache_prod_y = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto ewm_sum_product = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct>(
            OnlineEWMSumProduct::Window{weight_cache, data_cache_prod_x, data_cache_prod_y});

        auto data_cache_prod3_x = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto data_cache_prod3_y = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto data_cache_prod3_z = OnlineBaseFactor::createOnlineBaseF<OnlineDataCache>(initial_values);
        auto ewm_sum_product3 = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSumProduct3>(
            OnlineEWMSumProduct3::Window{weight_cache, data_cache_prod3_x, data_cache_prod3_y, data_cache_prod3_z});

        auto ewm_skew = OnlineBaseFactor::createOnlineBaseF<OnlineEWMSkew>(
            OnlineEWMSkew::Window{weight_cache, ewm_mean, ewm_sum_product, ewm_sum_product3});

        std::ofstream output_file(output_csv);
        if (!output_file.is_open()) {
            std::cerr << "错误: 无法写入输出文件: " << output_csv << '\n';
            return 1;
        }

        output_file << "# Generated at: " << Tool::Timestamp::getCurrentTimestamp() << "\n";
        output_file << "step,operation,value\n";

        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] === 0021_EWM_SKEW 测试 ===\n";
        cout << "初始窗口长度: " << initial_values.size() << '\n';

        const double init_value = ewm_skew->getValue();
        output_file << 0 << ",init," << std::fixed << std::setprecision(precision) << init_value << "\n";
        cout << "初始带权偏度: " << std::fixed << std::setprecision(precision) << init_value << '\n';

        for (int row = 0; row < update_matrix.rows(); ++row) {
            Ve update_values = rowToVector(update_matrix, row);
            if (update_values.size() == 0) {
                continue;
            }
            const size_t version = static_cast<size_t>(row + 1);
            ewm_skew->update(update_values, version);
            const double current_value = ewm_skew->getValue();
            output_file << (row + 1) << ",update," << std::fixed << std::setprecision(precision) << current_value << "\n";
            cout << "更新 " << (row + 1) << " -> 带权偏度: " << std::fixed << std::setprecision(precision) << current_value << '\n';
        }

        output_file.close();
        cout << "[" << Tool::Timestamp::getCurrentTimestamp() << "] 结果已写入: " << output_csv << '\n';

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << '\n';
        return 1;
    }

    return 0;
}



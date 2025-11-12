#include "basefactor.h"
#include "../BusinessFactor/businessfactor.h"
#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>
#include <iomanip>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

// ===========================================
// 非参数统计方法 (Non-parametric Statistics)
// ===========================================

void BaseFactor::rankpct(const VectorXd& input_vector, VectorXd& result) {
    std::vector<std::pair<double, int>> values;
    for(int idx = 0; idx < input_vector.size(); ++idx) {
        if(std::isnan(input_vector(idx))) {
            continue;
        }
        values.emplace_back(input_vector(idx), idx);
    }
    if(values.empty()) {
        result = VectorXd::Constant(input_vector.size(), std::numeric_limits<double>::quiet_NaN());
        return;
    }

    // 按值排序
    std::stable_sort(values.begin(), values.end(),
        [](const std::pair<double, int>& first, const std::pair<double, int>& second) {
            return first.first < second.first;
        });
    
    // 计算百分位排名
    result = VectorXd::Constant(input_vector.size(), std::numeric_limits<double>::quiet_NaN());
    int num_values = values.size();
    for(int idx = 0; idx < num_values; ++idx) {
        int position = values[idx].second;
        result(position) = double(idx+1) / num_values;
    }

    // 搜索相同值并平均排名
    for(int idx = 0; idx < num_values-1; idx++) {
        if(values[idx].first == values[idx+1].first) {
            int index_start = idx;
            int index_end = idx+1;
            int count = 2;
            double sum = result(values[index_start].second) + result(values[index_end].second);
            while(index_end < num_values-1) {
                if(values[index_end].first == values[index_end+1].first) {
                    index_end++;
                    count++;
                    sum += result(values[index_end].second);
                } else {
                    break;
                }
            }
            double avg_rank = sum / count;
            for(int j = index_start; j <= index_end; ++j) {
                int position = values[j].second;
                result(position) = avg_rank;
            }
            idx = index_end;
        }    
    }
}

// 行百分位rank - 基于排序的非参数统计
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：MatrixXd(每行百分位排名矩阵，相同维度，NaN值保持为NaN)
void BaseFactor::row_rankpct(const MatrixXd& mat, MatrixXd& result) {
    result = MatrixXd::Constant(mat.rows(), mat.cols(), std::numeric_limits<double>::quiet_NaN());
    
    #pragma omp parallel for
    for(int row = 0; row < mat.rows(); ++row) {
        // 对每一行进行百分位排名
        VectorXd row_ranks;
        rankpct(mat.row(row), row_ranks);
        result.row(row) = row_ranks;
    }
    
    // debug：打印排序后的排名矩阵
    // const int precision = 6;
    // cout << "=== 排序后排名矩阵 ===" << '\n';
    // for(int row = 0; row < mat.rows(); ++row) {
    //     cout << "第" << row << "行: ";
    //     for(int col = 0; col < mat.cols(); ++col) {
    //         if(std::isnan(result(row, col))) {
    //             cout << "NaN ";
    //         } else {
    //             cout << std::fixed << std::setprecision(precision) << result(row, col) << " ";
    //         }
    //     }
    //     cout << '\n';
    // }
    // cout << "========================" << '\n';
}

// rank差分均值 - 基于排序的差分分析
// 输入：mat(二维矩阵，可能包含NaN值)
// 输出：VectorXd(每列排序差分均值向量，基于百分位排名计算相邻元素差值均值)
void BaseFactor::col_rankdm(const MatrixXd& mat, VectorXd& result) {
    // 先进行百分位排名
    MatrixXd ranked;
    row_rankpct(mat, ranked);
    // 然后计算排名数据的差分均值
    col_meandiff(ranked, result);
}

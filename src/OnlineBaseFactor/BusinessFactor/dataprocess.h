#ifndef DATAPROCESS_H
#define DATAPROCESS_H

#include "../Eigen_extra/Eigen_extra.h"
#include <vector>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

// ===========================================
// 业务数据处理类 (Business Data Processing)
// ===========================================

// 专门处理业务层数据预处理和信号处理的类
// 包含从BaseFactor迁移过来的信号处理方法
class BusinessProcess {
public:
    // ---- 信号处理方法 (Signal Processing) ----
    // 输入：data(被过滤矩阵), filter(滤波矩阵), method(滤波方法"day"/"minute")
    // 输出：void (直接修改data矩阵，将满足条件的元素设为NaN)
    static void data_filter2(MatrixXd& data, const MatrixXd& filter, const std::string& method);
    
    // ---- 分钟线数据转换 (Minute Data Transform) ----
    // 输入：data(分钟线数据矩阵), operations(操作列表，如["drop1", "drop4"])
    // 输出：void (直接修改data矩阵)
    // 功能：将242根分钟线转换为240或241根，支持合并、删除等操作
    // 注意：需要根据实际数据源结构实现
    static void minute_data_transform(MatrixXd& data, const vector<string>& operations);
    
    // ---- 时间序列分组处理 (Time Series Grouping) ----
    // 输入：data(时间序列数据), sub_window(窗口大小), daily_split_way(分割方式)
    // 输出：vector<MatrixXd>(按日期分组的数据矩阵列表)
    // 注意：需要根据实际时间索引结构实现
    static vector<MatrixXd> group_by_date(const MatrixXd& data, int sub_window, const string& daily_split_way);
    
    // ---- 日内局部信号计算 (Intraday Local Signal Calculation) ----
    // 输入：group_data(分组数据), min_amt(分钟成交额数据)
    // 输出：VectorXd(局部信号向量)
    // 注意：需要根据实际业务逻辑实现
    static VectorXd calc_intra_subinfo(const MatrixXd& group_data, const MatrixXd& min_amt);
};

#endif // DATAPROCESS_H

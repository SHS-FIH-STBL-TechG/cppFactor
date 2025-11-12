#include "Eigen_extra.h"

namespace EigenExtra {


// 组合：删除整行、整列"无效"元素，返回最终矩阵，并回传保留的行列索引
MatrixXd compressRemoveAllEmptyRowsCols(const MatrixXd& input,
                                       std::vector<int>& keptRowIndex,
                                       std::vector<int>& keptColIndex) {
    MatrixXd noEmptyRows = removeAllEmptyRows(input, keptRowIndex);
    if (noEmptyRows.rows() == 0) {
        keptColIndex.clear();
        return MatrixXd(0, 0);
    }
    MatrixXd noEmptyRowsCols = removeAllEmptyCols(noEmptyRows, {}, keptColIndex);
    return noEmptyRowsCols;
}

} // namespace EigenExtra
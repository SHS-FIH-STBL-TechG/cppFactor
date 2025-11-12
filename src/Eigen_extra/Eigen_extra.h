#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace EigenExtra {

using Eigen::MatrixXd;
using Eigen::VectorXd;

using Ve = VectorXd;
using Ma = MatrixXd;

// 默认有效性判定：仅排除 NaN
inline bool defaultIsValid(double value) {
    return !std::isnan(value);
}

// 删除整行"无效"的行（无效: 谓词判定全部为 false），返回压缩矩阵，并回传保留行索引
template <typename Predicate>
inline MatrixXd removeAllEmptyRows(const MatrixXd& input,
                                   std::vector<int>& keptRowIndex,
                                   Predicate isValid) {
    const int numRows = static_cast<int>(input.rows());
    const int numCols = static_cast<int>(input.cols());

    keptRowIndex.clear();
    keptRowIndex.reserve(numRows);

    for (int row = 0; row < numRows; ++row) {
        bool anyValidInRow = false;
        for (int col = 0; col < numCols; ++col) {
            if (isValid(input(row, col))) { anyValidInRow = true; break; }
        }
        if (anyValidInRow) {
            keptRowIndex.push_back(row);
        }
    }

    if (keptRowIndex.empty()) {
        return MatrixXd(0, numCols);
    }

    MatrixXd output(static_cast<int>(keptRowIndex.size()), numCols);
    for (int i = 0; i < static_cast<int>(keptRowIndex.size()); ++i) {
        output.row(i) = input.row(keptRowIndex[i]);
    }
    return output;
}

// 无谓词重载：默认仅把 NaN 视作无效
inline MatrixXd removeAllEmptyRows(const MatrixXd& input,
                                   std::vector<int>& keptRowIndex) {
    return removeAllEmptyRows(input, keptRowIndex, defaultIsValid);
}

// 只返回无效行索引，不做矩阵拷贝（0拷贝版本）
template <typename Predicate>
inline std::vector<int> getInvalidRowIndices(const MatrixXd& input, Predicate isValid) {
    const int numRows = static_cast<int>(input.rows());
    const int numCols = static_cast<int>(input.cols());
    
    std::vector<int> invalidRows;
    invalidRows.reserve(numRows);
    
    for (int row = 0; row < numRows; ++row) {
        bool anyValidInRow = false;
        for (int col = 0; col < numCols; ++col) {
            if (isValid(input(row, col))) { 
                anyValidInRow = true; 
                break; 
            }
        }
        if (!anyValidInRow) {
            invalidRows.push_back(row);
        }
    }
    
    return invalidRows;
}

// 无谓词重载：默认仅把 NaN 视作无效
inline std::vector<int> getInvalidRowIndices(const MatrixXd& input) {
    return getInvalidRowIndices(input, defaultIsValid);
}

// 只返回有效行索引，不做矩阵拷贝（0拷贝版本）
template <typename Predicate>
inline std::vector<int> getValidRowIndices(const MatrixXd& input, Predicate isValid) {
    const int numRows = static_cast<int>(input.rows());
    const int numCols = static_cast<int>(input.cols());
    
    std::vector<int> validRows;
    validRows.reserve(numRows);
    
    for (int row = 0; row < numRows; ++row) {
        bool anyValidInRow = false;
        for (int col = 0; col < numCols; ++col) {
            if (isValid(input(row, col))) { 
                anyValidInRow = true; 
                break; 
            }
        }
        if (anyValidInRow) {
            validRows.push_back(row);
        }
    }
    
    return validRows;
}

// 无谓词重载：默认仅把 NaN 视作无效
inline std::vector<int> getValidRowIndices(const MatrixXd& input) {
    return getValidRowIndices(input, defaultIsValid);
}

// 删除整列"无效"的列（无效: 谓词判定在保留行中全部为 false），返回压缩矩阵，并回传保留列索引
template <typename Predicate>
inline MatrixXd removeAllEmptyCols(const MatrixXd& input,
                           const std::vector<int>& candidateRows,
                           std::vector<int>& keptColIndex,
                           Predicate isValid) {
    const int numRows = static_cast<int>(input.rows());
    const int numCols = static_cast<int>(input.cols());

    keptColIndex.clear();
    keptColIndex.reserve(numCols);

    const bool useRowSubset = !candidateRows.empty();
    for (int col = 0; col < numCols; ++col) {
        bool anyValidInCol = false;
        if (useRowSubset) {
            for (int idx = 0; idx < static_cast<int>(candidateRows.size()); ++idx) {
                const int row = candidateRows[idx];
                if (isValid(input(row, col))) { anyValidInCol = true; break; }
            }
        } else {
            for (int row = 0; row < numRows; ++row) {
                if (isValid(input(row, col))) { anyValidInCol = true; break; }
            }
        }
        if (anyValidInCol) {
            keptColIndex.push_back(col);
        }
    }

    if (keptColIndex.empty()) {
        return MatrixXd(static_cast<int>(useRowSubset ? candidateRows.size() : numRows), 0);
    }

    const int outRows = static_cast<int>(useRowSubset ? candidateRows.size() : numRows);
    MatrixXd output(outRows, static_cast<int>(keptColIndex.size()));

    if (useRowSubset) {
        for (int i = 0; i < outRows; ++i) {
            const int srcRow = candidateRows[i];
            for (int j = 0; j < static_cast<int>(keptColIndex.size()); ++j) {
                const int srcCol = keptColIndex[j];
                output(i, j) = input(srcRow, srcCol);
            }
        }
    } else {
        for (int i = 0; i < outRows; ++i) {
            for (int j = 0; j < static_cast<int>(keptColIndex.size()); ++j) {
                const int srcCol = keptColIndex[j];
                output(i, j) = input(i, srcCol);
            }
        }
    }

    return output;
}

// 无谓词重载：默认仅把 NaN 视作无效
inline MatrixXd removeAllEmptyCols(const MatrixXd& input,
                                   const std::vector<int>& candidateRows,
                                   std::vector<int>& keptColIndex) {
    return removeAllEmptyCols(input, candidateRows, keptColIndex, defaultIsValid);
}

// 只返回无效列索引，不做矩阵拷贝（0拷贝版本）
template <typename Predicate>
inline std::vector<int> getInvalidColIndices(const MatrixXd& input, 
                                     const std::vector<int>& candidateRows,
                                     Predicate isValid) {
    const int numRows = static_cast<int>(input.rows());
    const int numCols = static_cast<int>(input.cols());
    
    std::vector<int> invalidCols;
    invalidCols.reserve(numCols);
    
    const bool useRowSubset = !candidateRows.empty();
    for (int col = 0; col < numCols; ++col) {
        bool anyValidInCol = false;
        if (useRowSubset) {
            for (int idx = 0; idx < static_cast<int>(candidateRows.size()); ++idx) {
                const int row = candidateRows[idx];
                if (isValid(input(row, col))) { 
                    anyValidInCol = true; 
                    break; 
                }
            }
        } else {
            for (int row = 0; row < numRows; ++row) {
                if (isValid(input(row, col))) { 
                    anyValidInCol = true; 
                    break; 
                }
            }
        }
        if (!anyValidInCol) {
            invalidCols.push_back(col);
        }
    }
    
    return invalidCols;
}

// 无谓词重载：默认仅把 NaN 视作无效
inline std::vector<int> getInvalidColIndices(const MatrixXd& input, 
                                            const std::vector<int>& candidateRows) {
    return getInvalidColIndices(input, candidateRows, defaultIsValid);
}

// 组合：删除整行、整列"无效"元素，返回最终矩阵，并回传保留的行列索引
MatrixXd compressRemoveAllEmptyRowsCols(const MatrixXd& input,
                                       std::vector<int>& keptRowIndex,
                                       std::vector<int>& keptColIndex);

} // namespace EigenExtra

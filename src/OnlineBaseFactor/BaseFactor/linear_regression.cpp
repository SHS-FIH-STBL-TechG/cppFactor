#include "basefactor.h"
#include <iostream>
#include <limits>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ColPivHouseholderQR;
using Eigen::JacobiSVD;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using namespace std;

void BaseFactor::ols_solve(const MatrixXd& X_in,
                           const VectorXd& y_in,
                           VectorXd& beta,
                           bool fit_intercept,
                           int method)
{
    // ---------- Step 1: 基础检查 ----------
    if (X_in.rows() == 0 || X_in.cols() == 0 || y_in.size() == 0) {
        cerr << "[OLS] Error: Empty input matrix or vector." << endl;
        return;
    }
    if (X_in.rows() != y_in.size()) {
        cerr << "[OLS] Error: Dimension mismatch (X.rows != y.size)." << endl;
        return;
    }
    if (!((X_in.array().isFinite()).all() && (y_in.array().isFinite()).all())) {
        cerr << "[OLS] Error: Input contains NaN or Inf." << endl;
        return;
    }

    // ---------- Step 2: 构造 X ----------
    MatrixXd X = X_in;
    if (fit_intercept) {
        X.conservativeResize(X.rows(), X.cols() + 1);
        X.col(X.cols() - 1).setOnes();
    }

    int n = X.rows();
    int p = X.cols();
    if (n <= p) {
        cerr << "[OLS] Error: Not enough samples (n=" << n << ", p=" << p << ")." << endl;
        return;
    }

    // ---------- Step 3: 根据method参数选择求解方法 ----------
    switch (method) {
        case 0: // QR分解方法
            {
                cout << "[OLS] Using QR decomposition method..." << endl;
                ColPivHouseholderQR<MatrixXd> qr(X);
                beta = qr.solve(y_in);
                if (!beta.allFinite()) {
                    cerr << "[OLS] Error: QR decomposition failed (NaN/Inf)." << endl;
                }
            }
            break;
            
        case 1: // SVD分解方法
            {
                cout << "[OLS] Using SVD decomposition method..." << endl;
                JacobiSVD<MatrixXd> svd(X, ComputeThinU | ComputeThinV);
                VectorXd S = svd.singularValues();
                double tol = std::numeric_limits<double>::epsilon() * std::max(X.rows(), X.cols()) * S(0);
                VectorXd invS = S.unaryExpr([&](double s) { return (s > tol) ? 1.0 / s : 0.0; });
                beta = svd.matrixV() * invS.asDiagonal() * svd.matrixU().adjoint() * y_in;
                
                if (!beta.allFinite()) {
                    cerr << "[OLS] Error: SVD decomposition failed (NaN/Inf)." << endl;
                }
            }
            break;
            
        case 2: // Ridge正则化方法
            {
                cout << "[OLS] Using Ridge regularization method..." << endl;
                double lambda = 1e-6;
                MatrixXd XtX = X.transpose() * X;
                XtX += lambda * MatrixXd::Identity(p, p);
                VectorXd Xty = X.transpose() * y_in;
                beta = XtX.ldlt().solve(Xty);
                
                if (!beta.allFinite()) {
                    cerr << "[OLS] Error: Ridge regularization failed (NaN/Inf)." << endl;
                }
            }
            break;
        case 3: // 正规方程标准解
            {
                cout << "[OLS] Using normal equation standard solution..." << endl;
                MatrixXd XtX = X.transpose() * X;
                VectorXd Xty = X.transpose() * y_in;
                beta = XtX.ldlt().solve(Xty);
                
                if (!beta.allFinite()) {
                    cerr << "[OLS] Error: Normal equation standard solution failed (NaN/Inf)." << endl;
                }
            }
            break;
            
        default:
            cerr << "[OLS] Error: Invalid method parameter (" << method << "). Use 0, 1, or 2." << endl;
            return;
    }

    // ---------- Step 6: 最终验证 ----------
    if (!beta.allFinite()) {
        cerr << "[OLS] Error: All methods failed to produce valid coefficients." << endl;
    }
}



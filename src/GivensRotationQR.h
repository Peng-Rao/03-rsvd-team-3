#ifndef GIVENSROTATIONQR_H
#define GIVENSROTATIONQR_H

#include "MatrixTraits.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <omp.h>

namespace Eigen {
    /**
     * @brief Base class template (default for dense matrices)
     */
    template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
    // template<typename MatrixType>
    class GivensRotationQR {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;

        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

        GivensRotationQR() = default;

        /**
         * @brief Compute the QR decomposition using Given rotations
         *
         * @param matrix The input matrix to decompose
         * @return Reference to this object
         */
        GivensRotationQR& compute(const MatrixType& matrix) {
            Index rows = matrix.rows();
            Index cols = matrix.cols();
            m_matrixR = matrix;
            m_matrixQ = DenseMatrix::Identity(rows, rows);

            for (Index j = 0; j < cols; ++j) {
                for (Index i = rows - 1; i > j; --i) {
                    Scalar a = m_matrixR(i - 1, j);
                    Scalar b = m_matrixR(i, j);
                    Scalar c, s;
                    computeGivensRotation(a, b, c, s);

                    applyGivensRotation(m_matrixR, i - 1, i, c, s);
                    applyGivensRotation(m_matrixQ, i - 1, i, c, s, true);
                }
            }

            return *this;
        }

        /**
         * @brief Access the orthogonal matrix Q
         * @return The Q matrix
         */
        const DenseMatrix& matrixQ() const { return m_matrixQ; }

        /**
         * @brief Access the upper triangular matrix R
         * @return The R matrix
         */
        const DenseMatrix& matrixR() const { return m_matrixR; }

    private:
        /**
         * @brief Compute the Given rotation coefficients
         *
         * @param a First value
         * @param b Second value
         * @param c Output cosine
         * @param s Output sine
         */
        void computeGivensRotation(Scalar a, Scalar b, Scalar& c, Scalar& s) {
            if (b == Scalar(0)) {
                c = Scalar(1);
                s = Scalar(0);
            } else {
                Scalar r = std::hypot(a, b);
                c = a / r;
                s = -b / r;
            }
        }
        /**
         * @brief Apply a Given rotation to a matrix
         *
         * @param matrix The matrix to apply the rotation to
         * @param i Row index 1
         * @param k Row index 2
         * @param c Cosine of the rotation
         * @param s Sine of the rotation
         * @param isQ If true, apply the transpose for Q
         */
        void applyGivensRotation(DenseMatrix& matrix, Index i, Index k, Scalar c, Scalar s, bool isQ = false) {
            if (isQ) {
                /*
                 * Apply the rotation to the columns of the matrix
                 * Q = I * G_1^T * G_2^T * ... * G_n^T
                 * Instead of computing the transpose of the rotation matrix, we apply the rotation to the columns
                 * QT = G_1 * G_2 * ... * G_n * I
                 */
                auto temp_i = matrix.col(i).eval();
                auto temp_k = matrix.col(k).eval();
                matrix.col(i) = c * temp_i - s * temp_k;
                matrix.col(k) = s * temp_i + c * temp_k;
            } else {
                /*
                 * Apply the rotation to the rows of the matrix
                 * R = G_n * ... * G_2 * G_1 * A
                 */
                auto temp_i = matrix.row(i).eval();
                auto temp_k = matrix.row(k).eval();
                matrix.row(i) = c * temp_i - s * temp_k;
                matrix.row(k) = s * temp_i + c * temp_k;
            }
        }

        DenseMatrix m_matrixQ;
        DenseMatrix m_matrixR;
    };

    // /**
    //  * @brief sparse matrices
    //  */
    template<typename MatrixType>
    class GivensRotationQR<MatrixType, true> {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;

        using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

        GivensRotationQR() = default;

        GivensRotationQR& compute(const MatrixType& matrix) {
            Index rows = matrix.rows();
            Index cols = matrix.cols();
            m_matrixR = matrix;
            m_matrixQ.resize(rows, rows);
            m_matrixQ.setIdentity();

            for (Index j = 0; j < cols; ++j) {
                for (Index i = rows - 1; i > j; --i) {
                    Scalar a = m_matrixR.coeff(i - 1, j);
                    Scalar b = m_matrixR.coeff(i, j);
                    if (std::abs(a) < 1e-9 && std::abs(b) < 1e-9) {
                        continue;
                    }

                    Scalar c, s;
                    computeGivensRotation(a, b, c, s);

                    SparseMatrixType G(rows, rows);
                    G.setIdentity();
                    G.coeffRef(i - 1, i - 1) = c;
                    G.coeffRef(i - 1, i)     = -s;
                    G.coeffRef(i, i - 1)     = s;
                    G.coeffRef(i, i)         = c;
                    G.makeCompressed();

                    m_matrixR = (G * m_matrixR).pruned();

                    SparseMatrixType Gt = G.transpose();
                    Gt.makeCompressed();
                    m_matrixQ = (m_matrixQ * Gt).pruned();
                }
            }

            return *this;
        }

        const SparseMatrixType& matrixQ() const { return m_matrixQ; }
        const SparseMatrixType& matrixR() const { return m_matrixR; }

    private:
        void computeGivensRotation(Scalar a, Scalar b, Scalar& c, Scalar& s) {
            if (b == Scalar(0)) {
                c = Scalar(1);
                s = Scalar(0);
            } else {
                Scalar r = std::hypot(a, b);
                c = a / r;
                s = -b / r;
            }
        }

        SparseMatrixType m_matrixQ;
        SparseMatrixType m_matrixR;
    };

} // namespace Eigen

#endif // GIVENSROTATIONQR_H
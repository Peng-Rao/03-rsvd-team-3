#ifndef POWERMETHODSVD_H
#define POWERMETHODSVD_H

#include "MatrixTraits.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <iostream>
#include <random>

namespace Eigen {
    /**
     * @brief Base class template (default for dense matrices)
     */
    template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
    class PowerMethodSVD {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        PowerMethodSVD() = default;

        /**
         * @brief Compute a rank-k SVD approximation of 'matrix' using repeated 1-vector
         *        power-method and deflation.  We follow the pseudocode in your picture:
         *
         *        for each rank-1:
         *           1) pick x ~ N(0,1)
         *           2) s = log(4 log(2n/delta)/(epsDelta)) / (2 lambda)
         *           3) repeat s times: x <- A^T A x;  x <- x / ||x||
         *           4) sigma = ||A x||; u = (A x)/sigma
         *           5) deflate: A <- A - sigma * u * x^T
         *
         * @param matrix  The input matrix
         * @param rank    Number of singular values/vectors to compute
         * @param epsDelta tolerance for convergence
         * @param delta   scaling factor
         * @param lambda  regularization parameter
         */
        PowerMethodSVD &compute(const MatrixType &matrix, Index rank, const double epsDelta = 1e-9,
                                const double delta = 0.05, const double lambda = 0.1) {
            // Copy into a modifiable matrix for deflation
            DenseMatrix A = matrix.template cast<Scalar>();

            Index m = A.rows();
            Index n = A.cols();

            // We only compute up to the smaller dimension or user-specified rank
            Index k = std::min(rank, std::min(m, n));

            // Allocate storage for S, U, V
            m_singularValues.resize(k);
            m_matrixU.resize(m, k);
            m_matrixV.resize(n, k);

            using std::ceil;
            using std::log;
            const double insideLog = 4.0 * log((2.0 * n) / delta) / epsDelta;
            const double sReal = log(insideLog) / (2.0 * lambda);
            const int s = static_cast<int>(ceil(sReal));

            // For each rank-1 piece, do:
            for (Index i = 0; i < k; ++i) {
                Scalar sigma;
                DenseVector u(m), v(n);

                // Extract top singular value/vector from the current deflated A
                bool success = extractLargestSingularValue(A, sigma, u, v, s);
                if (!success) {
                    // If we fail, shrink the storage and stop
                    m_singularValues.conservativeResize(i);
                    m_matrixU.conservativeResize(m, i);
                    m_matrixV.conservativeResize(n, i);
                    break;
                }

                // Store results
                m_singularValues(i) = sigma;
                m_matrixU.col(i) = u;
                m_matrixV.col(i) = v;

                // Deflate: A <- A - sigma * u * v^T
                A.noalias() -= sigma * (u * v.transpose());
            }

            return *this;
        }

        /**
         * @return Vector of singular values
         */
        const DenseVector &singularValues() const { return m_singularValues; }

        /**
         * @return The left singular vectors (matrix U)
         */
        const DenseMatrix &matrixU() const { return m_matrixU; }

        /**
         * @return The right singular vectors (matrix V)
         */
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Extract the largest singular value/vector from A by
         *        (a) one-vector power method on (A^T A)
         *        (b) computing sigma = ||A v||
         *        (c) u = (A v)/sigma
         *
         * The iteration count 's' is chosen from:
         *   s = log( 4 log(2n/delta) / (epsDelta) ) / (2 lambda)
         *
         * @param A        The current (deflated) matrix
         * @param sigma    (output) the largest singular value found
         * @param u        (output) left singular vector
         * @param v        (output) right singular vector
         * @param s       Number of iterations for power method
         * @return True if we found a non-trivial singular vector (sigma>0),
         *         false otherwise
         */
        bool extractLargestSingularValue(const DenseMatrix &A, Scalar &sigma, DenseVector &u, DenseVector &v,
                                         const int s) {

            Index n = A.cols();
            // 1) random initialization for v
            v = randomVector(n);

            // 2) compute s

            // 3) do s iterations of v <- A^T A v, normalize
            for (int i = 0; i < s; ++i) {
                DenseVector temp = A.transpose() * (A * v);
                double normTemp = temp.norm();
                if (normTemp < 1e-14) {
                    return false; // A^T A v ~ 0 => degenerate
                }
                v = temp / normTemp;
            }

            // 4) sigma = ||A v||
            DenseVector Av = A * v;
            sigma = Av.norm();
            if (sigma < 1e-14) {
                return false;
            }

            //    u = (A v)/sigma
            u = Av / sigma;
            return true;
        }

        /**
         * @brief Generate a random vector of length 'n' ~ Normal(0,1)
         */
        DenseVector randomVector(Index n) const {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseVector vec(n);
            for (Index i = 0; i < n; ++i) {
                vec(i) = dist(gen);
            }
            return vec;
        }

        DenseMatrix m_matrixU; ///< Left singular vectors
        DenseMatrix m_matrixV; ///< Right singular vectors
        DenseVector m_singularValues; ///< The singular values
    };

    // Specialization for Sparse Matrices
    /**
     * @brief Sparse specialization of PowerMethodSVD, consistent with the dense algorithm.
     */
    template<typename MatrixType>
    class PowerMethodSVD<MatrixType, true> {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;
        using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

        PowerMethodSVD() = default;

        /**
         * @brief Compute a rank-k SVD approximation of the sparse matrix 'matrix' using
         *        repeated 1-vector power-method and deflation, similar to the dense version:
         *        for each rank-1:
         *           1) pick v ~ N(0,1)
         *           2) s = log(4 log(2n/delta)/(epsDelta)) / (2 lambda)
         *           3) repeat s times: v <- A^T (A v);  v <- v / ||v||
         *           4) sigma = ||A v||; u = (A v)/sigma
         *           5) deflate: A <- A - sigma * u * v^T
         *
         * @param matrix    Input sparse matrix
         * @param rank      Number of singular values/vectors to compute
         * @param epsDelta  Tolerance scaling
         * @param delta     Probability-like factor
         * @param lambda    Regularization parameter controlling # of iterations
         * @return          Reference to this for chaining
         */
        PowerMethodSVD &compute(const MatrixType &matrix, Index rank, double epsDelta = 1e-9, double delta = 0.05,
                                double lambda = 0.1) {
            // Copy the sparse matrix (so we can deflate)
            SparseMatrixType A_sparse = matrix;
            A_sparse.makeCompressed();

            Index m = A_sparse.rows();
            Index n = A_sparse.cols();
            Index k = std::min(rank, std::min(m, n));

            m_singularValues.resize(k);
            m_matrixU.resize(m, k);
            m_matrixV.resize(n, k);

            using std::ceil;
            using std::log;
            const double insideLog = 4.0 * log((2.0 * n) / delta) / epsDelta;
            const double sReal = log(insideLog) / (2.0 * lambda);
            const int s = static_cast<int>(ceil(sReal));

            // Extract each rank-1 contribution
            for (Index i = 0; i < k; ++i) {
                Scalar sigma;
                DenseVector u(m), v(n);

                bool success = extractLargestSingularValueSparse(A_sparse, sigma, u, v, s);
                if (!success) {
                    // If we fail, break early
                    m_singularValues.conservativeResize(i);
                    m_matrixU.conservativeResize(m, i);
                    m_matrixV.conservativeResize(n, i);
                    break;
                }

                // Store the results
                m_singularValues(i) = sigma;
                m_matrixU.col(i) = u;
                m_matrixV.col(i) = v;

                // Deflate A <- A - sigma * u * v^T
                DenseMatrix uvT = u * v.transpose(); // (m x n)
                SparseMatrixType uvT_sparse = uvT.sparseView();
                A_sparse = A_sparse - sigma * uvT_sparse;

                // Optionally prune small entries for numerical stability
                A_sparse.prune(Scalar(1e-12));
                A_sparse.makeCompressed();
            }

            return *this;
        }

        const DenseVector &singularValues() const { return m_singularValues; }
        const DenseMatrix &matrixU() const { return m_matrixU; }
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Same core power-method for the largest singular value of A (sparse),
         *        matching the dense approach:
         *         1) random v
         *         2) s iterations: v <- A^T(A v), v <- v/||v||
         *         3) sigma = ||A v||
         *         4) u = A v / sigma
         *
         * @param A      Current deflated matrix
         * @param sigma  (output) largest singular value
         * @param u      (output) left singular vector
         * @param v      (output) right singular vector
         * @param s      Number of power iterations
         * @return       True if a non-trivial singular vector is found
         */
        bool extractLargestSingularValueSparse(const SparseMatrixType &A, Scalar &sigma, DenseVector &u, DenseVector &v,
                                               int s) {
            const Index n = A.cols();
            // 1) Random initialization of v
            v = randomVector(n);

            // 2) Power iterations
            for (int i = 0; i < s; ++i) {
                DenseVector temp = A.transpose() * (A * v);
                double normTemp = temp.norm();
                if (normTemp < 1e-14) {
                    return false; // Degenerate direction
                }
                v = temp / normTemp;
            }

            // 3) sigma = ||A v||
            DenseVector Av = A * v;
            sigma = Av.norm();
            if (sigma < 1e-14) {
                return false;
            }

            // 4) u = (A v)/sigma
            u = Av / sigma;
            return true;
        }

        /**
         * @brief Generate a random Gaussian vector of length 'n'
         */
        DenseVector randomVector(Index n) const {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(Scalar(0), Scalar(1));

            DenseVector vec(n);
            for (Index i = 0; i < n; ++i) {
                vec(i) = dist(gen);
            }
            return vec;
        }

        DenseMatrix m_matrixU; ///< Left singular vectors
        DenseMatrix m_matrixV; ///< Right singular vectors
        DenseVector m_singularValues; ///< Singular values
    };

} // namespace Eigen

#endif // POWERMETHODSVD_H

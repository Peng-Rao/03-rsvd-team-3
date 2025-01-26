#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H

#include "MatrixTraits.h"
#include "PowerMethodSVD.h"
#include "GivensRotationQR.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <random>

namespace Eigen {

    /**
     * @brief A class to perform Randomized Singular Value Decomposition (RSVD), default for dense matrices
     */
    template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
    class RandomizedSVD {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        /**
         * @brief Default constructor
         */
        RandomizedSVD() = default;

        /**
         * @brief Compute the RSVD of a matrix
         *
         * @param matrix The input matrix to decompose
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations for accuracy (default: 2)
         * @return Reference to this RSVD object
         */
        RandomizedSVD &compute(const MatrixType &matrix, Index rank, Index powerIterations = 2) {
            randomProjection(matrix, rank, powerIterations);
            return *this;
        }

        /**
         * @brief Get the singular values
         * @return A vector containing the singular values
         */
        const DenseVector &singularValues() const { return m_singularValues; }

        /**
         * @brief Get the left singular vectors (U matrix)
         * @return The U matrix
         */
        const DenseMatrix &matrixU() const { return m_matrixU; }

        /**
         * @brief Get the right singular vectors (V matrix)
         * @return The V matrix
         */
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Generate a random matrix
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return A random matrix
         */
        DenseMatrix generateRandomMatrix(Index rows, Index cols) {
            std::normal_distribution<Scalar> dist(0.0, 1.0);
            std::mt19937 m_gen;
            return DenseMatrix::NullaryExpr(rows, cols, [&]() { return dist(m_gen); });
        }

        /**
         * @brief Perform the randomized projection
         *
         * @param matrix The input matrix
         * @param k Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const MatrixType &matrix, Index k, Index powerIterations) {
            // 1) Oversampling parameter
            // Typically p = 5-20 for better stability. Here, we use 10.
            Index p = 10;
            Index ncols = matrix.cols();
            // We'll project onto (k + p) dimensions
            Index targetRank = k + p;

            // 2) Generate random Gaussian matrix Omega (n x (k+p))
            DenseMatrix Omega = generateRandomMatrix(ncols, targetRank);

            // 3) Initial sketch Y0 = A * Omega
            DenseMatrix Y = matrix * Omega;

            // 4) Power iterations (with re-orthonormalization)
            for (Index i = 0; i < powerIterations; ++i) {
                // Orthogonalize Y
                HouseholderQR<DenseMatrix> qrY(Y);
                DenseMatrix QY = qrY.householderQ() * DenseMatrix::Identity(Y.rows(), targetRank);

                // Z = A^T * QY
                DenseMatrix Z = matrix.transpose() * QY; // (n x m) * (m x (k+p)) -> (n x (k+p))

                // Y = A * Z
                Y = matrix * Z;                          // (m x n) * (n x (k+p)) -> (m x (k+p))
            }

            // 5) Final orthonormal basis Q
            HouseholderQR<DenseMatrix> qrFinal(Y);
            DenseMatrix Q = qrFinal.householderQ() * DenseMatrix::Identity(Y.rows(), targetRank); // (m x (k+p))

            // 6) Form the smaller matrix B = Q^T * A (size: (k+p) x n)
            DenseMatrix B = Q.transpose() * matrix;

            // 7) SVD on the small matrix B
            BDCSVD<DenseMatrix> svd(B, ComputeThinU | ComputeThinV);
            const DenseVector &fullS = svd.singularValues();
            Index actualRank         = std::min<Index>(k, fullS.size());

            // 8) Truncate to rank k
            m_singularValues = fullS.head(actualRank);
            m_matrixU        = Q * svd.matrixU().leftCols(actualRank); // (m x (k+p)) * ((k+p) x k) => (m x k)
            m_matrixV        = svd.matrixV().leftCols(actualRank);     // (n x (k+p)) * ((k+p) x k) => (n x k)
        }

        // Member variables to store results
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };


    // Specialization for Sparse Matrices
    template<typename MatrixType>
    class RandomizedSVD<MatrixType, true> {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;

        using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

        /**
         * @brief Default constructor
         */
        RandomizedSVD() = default;

        /**
         * @brief Compute the RSVD of a sparse matrix
         *
         * @param matrix The input sparse matrix to decompose
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations for accuracy (default: 2)
         * @return Reference to this RSVD object
         */
        RandomizedSVD &compute(const SparseMatrixType &matrix, Index rank, Index powerIterations = 2) {
            randomProjection(matrix, rank, powerIterations);
            return *this;
        }

        /**
         * @brief Get the singular values
         * @return A vector containing the singular values
         */
        const DenseVector &singularValues() const { return m_singularValues; }

        /**
         * @brief Get the left singular vectors (U matrix)
         * @return The U matrix
         */
        const DenseMatrix &matrixU() const { return m_matrixU; }

        /**
         * @brief Get the right singular vectors (V matrix)
         * @return The V matrix
         */
        const DenseMatrix &matrixV() const { return m_matrixV; }

    private:
        /**
         * @brief Generate a random Gaussian dense matrix
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return A random dense matrix with entries drawn from a standard normal distribution
         */
        DenseMatrix generateRandomMatrix(Index rows, Index cols) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseMatrix mat(rows, cols);
            for (Index i = 0; i < rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    mat(i, j) = dist(gen);
                }
            }
            return mat;
        }

        /**
         * @brief Perform the randomized projection for sparse matrices
         *
         * @param matrix The input sparse matrix
         * @param k Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const SparseMatrixType &matrix, Index k, Index powerIterations) {
            Index p = 10;
            Index ncols = matrix.cols();
            Index targetRank = k + p;

            DenseMatrix Omega = generateRandomMatrix(ncols, targetRank);

            DenseMatrix Y = matrix * Omega;  // (m x (k+p))

            for (Index i = 0; i < powerIterations; ++i) {
                SparseMatrixType Ysp = Y.sparseView();

                GivensRotationQR<SparseMatrixType> qr;
                qr.compute(Ysp);

                DenseMatrix QY = qr.matrixQ() * DenseMatrix::Identity(Ysp.rows(), targetRank);

                // (d) Z = A^T * QY
                DenseMatrix Z = matrix.transpose() * QY; // (n x m) * (m x targetRank) => (n x targetRank)

                // (e) Y = A * Z
                Y = matrix * Z; // (m x n) * (n x targetRank) => (m x targetRank)
            }

            {
                SparseMatrixType Ysp = Y.sparseView();
                GivensRotationQR<SparseMatrixType> qrFinal;
                qrFinal.compute(Ysp);
                m_finalQ = qrFinal.matrixQ() * DenseMatrix::Identity(Ysp.rows(), targetRank);
            }

            DenseMatrix B = m_finalQ.transpose() * matrix;

            BDCSVD<DenseMatrix> svd(B, ComputeThinU | ComputeThinV);
            const DenseVector &fullS = svd.singularValues();
            Index actualRank         = std::min<Index>(k, fullS.size());

            m_singularValues = fullS.head(actualRank);
            m_matrixU = m_finalQ * svd.matrixU().leftCols(actualRank);
            m_matrixV = svd.matrixV().leftCols(actualRank);
        }

        // Member variables to store results
        DenseMatrix m_finalQ;
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };


} // namespace Eigen

#endif // RANDOMIZEDSVD_H
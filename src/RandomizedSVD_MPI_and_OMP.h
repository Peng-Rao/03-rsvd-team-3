#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H  //guards to protect against multiple inclusions of the same header file

#include "MatrixTraits.h"
// #include "PowerMethodSVD.h"
// #include "GivensRotationQR_MPI.h"
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <random>
#include <mpi.h>
#include <iostream>
#include <omp.h>
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
        
        /**
         * @brief Generate a random matrix using MPI
         *
         * @param rows Number of rows
         * @param cols Number of columns
         * @return A random matrix
         */
        DenseMatrix generateRandomMatrix(Index rows, Index cols) {
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            if (world_rank == 0) {
                std::cout << "Starting generateRandomMatrix..." << std::endl;
            }

            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);

            Index local_rows = rows / world_size;
            if (world_rank < rows % world_size) {
                local_rows++;
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseMatrix local_mat(local_rows, cols);

            #pragma omp parallel for collapse(2)
            for (Index i = 0; i < local_rows; ++i) {
                for (Index j = 0; j < cols; ++j) {
                    local_mat(i, j) = dist(gen);
                }
            }

            DenseMatrix global_mat(rows, cols);
            MPI_Gather(local_mat.data(), local_rows * cols, MPI_DOUBLE,
                    global_mat.data(), local_rows * cols, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

            return global_mat;
        }
    private:
        /**
         * @brief Perform the randomized projection
         *
         * @param matrix The input matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const MatrixType &matrix, Index rank, Index powerIterations) {
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            // Step 1: Generate a random Gaussian matrix using MPI
            DenseMatrix randomMatrix = generateRandomMatrix(matrix.cols(), rank);

            m_singularValues.resize(rank);
            m_matrixU.resize(matrix.rows(), rank);
            m_matrixV.resize(matrix.cols(), rank);

            if (world_rank == 0) {
                // Step 2: Form the sketch matrix (single process)
                DenseMatrix sketch = matrix * randomMatrix;

                // Step 3: Perform power iterations to refine the sketch
                for (Index i = 0; i < powerIterations; ++i) {
                    sketch = matrix * (matrix.transpose() * sketch);
                }

                // Step 4: Compute the QR decomposition of the sketch
                HouseholderQR<DenseMatrix> qr(sketch);
                DenseMatrix Q = qr.householderQ() * DenseMatrix::Identity(sketch.rows(), rank);

                // Step 5: Project the original matrix onto the low-dimensional subspace
                DenseMatrix B = Q.transpose() * matrix;

                // Step 6: Perform SVD on the small matrix B
                JacobiSVD<DenseMatrix> svd;
                svd.compute(B, ComputeThinU | ComputeThinV);
                m_singularValues = svd.singularValues();
                m_matrixU = Q * svd.matrixU();
                m_matrixV = svd.matrixV();
            }

            // Broadcast results to all processes
            MPI_Bcast(m_singularValues.data(), rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(m_matrixU.data(), matrix.rows() * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(m_matrixV.data(), matrix.cols() * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Member variables to store results
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };



} // namespace Eigen

#endif // RANDOMIZEDSVD_H
#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H

#include "PowerMethodSVD.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>
#include <random>

namespace Eigen {

    /**
     * @brief A class to perform Randomized Singular Value Decomposition (RSVD)
     */
    template<typename MatrixType, int Options = ComputeThinU | ComputeThinV>
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
         * @brief Compute the RSVD of a matrix using MPI
         */
        RandomizedSVD &compute_mpi(const MatrixType &matrix, Index rank, Index powerIterations = 2) {
            randomProjection_mpi(matrix, rank, powerIterations);
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
         * @brief Perform the randomized projection
         *
         * @param matrix The input matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const MatrixType &matrix, Index rank, Index powerIterations) {
            // Step 1: Generate a random Gaussian matrix
            DenseMatrix randomMatrix = generateRandomMatrix(matrix.cols(), rank);

            // Step 2: Form the sketch matrix
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
            PowerMethodSVD<DenseMatrix> svd;
            svd.compute(B, rank, 1000, Scalar(1e-6));
            m_singularValues = svd.singularValues();
            m_matrixU = Q * svd.matrixU();
            m_matrixV = svd.matrixV();
        }

        /**
         * @brief Perform the randomized projection using MPI
         *
         * @param matrix The input matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection_mpi(const MatrixType &matrix, Index rank, Index powerIterations) {
            int world_size, world_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            Index rows = matrix.rows();
            Index cols = matrix.cols();

            // Step 1: Generate a random Gaussian matrix on rank 0 and broadcast
            DenseMatrix randomMatrix(cols, rank);
            if (world_rank == 0) {
                randomMatrix = DenseMatrix::Random(cols, rank);
            }
            MPI_Bcast(randomMatrix.data(), cols * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Step 2: Form the sketch matrix and reduce
            DenseMatrix sketch = matrix * randomMatrix;
            DenseMatrix global_sketch(rows, rank);
            MPI_Allreduce(MPI_IN_PLACE, sketch.data(), rows * rank,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            global_sketch = sketch;

            // Step 3: Perform power iterations to refine the sketch
            for (Index i = 0; i < powerIterations; ++i) {
                sketch = matrix.transpose() * global_sketch;
                sketch = matrix * sketch;
                MPI_Allreduce(MPI_IN_PLACE, sketch.data(), rows * rank,
                             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                global_sketch = sketch;
            }

            // Step 4: Compute the QR decomposition of the sketch
            HouseholderQR<DenseMatrix> qr(global_sketch);
            DenseMatrix Q = qr.householderQ() * DenseMatrix::Identity(rows, rank);

            // Step 5: Project the original matrix onto the low-dimensional subspace and reduce
            DenseMatrix B = Q.transpose() * matrix;
            DenseMatrix global_B = B;
            MPI_Allreduce(MPI_IN_PLACE, global_B.data(), rank * cols,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Step 6: Perform SVD on the small matrix B
            BDCSVD<DenseMatrix> svd(global_B, Options);
            m_singularValues = svd.singularValues();
            m_matrixU = Q * svd.matrixU();
            m_matrixV = svd.matrixV();
        }

        // Member variables to store results
        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };

} // namespace Eigen

#endif // RANDOMIZEDSVD_H

#ifndef RANDOMIZEDSVD_H
#define RANDOMIZEDSVD_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <mpi.h>

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
            int world_size, world_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            Index rows = matrix.rows();
            Index cols = matrix.cols();

            // Generate random matrix on rank 0 and broadcast
            DenseMatrix randomMatrix(cols, rank);
            if (world_rank == 0) {
                randomMatrix = DenseMatrix::Random(cols, rank);
            }
            MPI_Bcast(randomMatrix.data(), cols * rank, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Form and reduce sketch
            DenseMatrix sketch = matrix * randomMatrix;  // (rows x cols) * (cols x rank) = (rows x rank)
            DenseMatrix global_sketch(rows, rank);
            MPI_Allreduce(MPI_IN_PLACE, sketch.data(), rows * rank,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            global_sketch = sketch;

            // Power iterations
            for (Index i = 0; i < powerIterations; ++i) {
                sketch = matrix.transpose() * global_sketch;  // (cols x rows) * (rows x rank) = (cols x rank)
                sketch = matrix * sketch;  // (rows x cols) * (cols x rank) = (rows x rank)
                MPI_Allreduce(MPI_IN_PLACE, sketch.data(), rows * rank,
                             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                global_sketch = sketch;
            }

            // QR decomposition
            HouseholderQR<DenseMatrix> qr(global_sketch);
            DenseMatrix Q = qr.householderQ();  // (rows x rank)

            // Project and reduce
            DenseMatrix B = Q.transpose() * matrix;  // (rank x rows) * (rows x cols) = (rank x cols)
            DenseMatrix global_B = B;
            MPI_Allreduce(MPI_IN_PLACE, global_B.data(), rank * cols,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Final SVD
            BDCSVD<DenseMatrix> svd(global_B, Options);
            m_singularValues = svd.singularValues();
            m_matrixU = Q * svd.matrixU();
            m_matrixV = svd.matrixV();

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
         * @brief Perform the randomized projection
         *
         * @param matrix The input matrix
         * @param rank Desired rank of the approximation
         * @param powerIterations Number of power iterations
         */
        void randomProjection(const MatrixType &matrix, Index rank, Index powerIterations) {
            // Step 1: Generate a random Gaussian matrix
            DenseMatrix randomMatrix = DenseMatrix::Random(matrix.cols(), rank);

            // Step 2: Form the sketch matrix
            DenseMatrix sketch = matrix * randomMatrix;

            // Step 3: Perform power iterations to refine the sketch
            for (Index i = 0; i < powerIterations; ++i) {
                sketch = matrix * (matrix.transpose() * sketch);
            }

            // Step 4: Compute the QR decomposition of the sketch
            HouseholderQR<DenseMatrix> qr(sketch);
            DenseMatrix Q = qr.householderQ();

            // Step 5: Project the original matrix onto the low-dimensional subspace
            DenseMatrix B = Q.transpose() * matrix;

            // Step 6: Perform SVD on the small matrix B
            BDCSVD<DenseMatrix> svd(B, Options);
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

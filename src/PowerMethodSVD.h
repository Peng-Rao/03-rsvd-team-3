#ifndef POWERMETHODSVD_H
#define POWERMETHODSVD_H

#include <Eigen/Core>
#include <random>

namespace Eigen {

    /**
     * @brief A class to perform SVD using Power Method and Deflation
     */
    template<typename MatrixType, int Options = ComputeThinU | ComputeThinV>
    class PowerMethodSVD {
    public:
        using Scalar = typename MatrixType::Scalar;
        using Index = typename MatrixType::Index;
        using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
        using DenseVector = Matrix<Scalar, Dynamic, 1>;

        PowerMethodSVD() = default;

        /**
         * @brief Compute the SVD of a matrix up to the given rank using power method + deflation
         *
         * @param matrix The input matrix to decompose
         * @param rank Desired rank of the approximation
         * @param maxIters Maximum iterations for each singular value extraction
         * @param tol Tolerance for convergence
         * @return Reference to this object
         */
        PowerMethodSVD &compute(const MatrixType &matrix,
                                Index rank,
                                Index maxIters = 1000,
                                Scalar tol = 1e-6) {

            // Make a copy of the matrix since we will deflate it
            DenseMatrix A = matrix.template cast<Scalar>();
            Index m = A.rows();
            Index n = A.cols();
            Index k = std::min(rank, std::min(m, n));

            m_singularValues.resize(k);
            m_matrixU.resize(m, k);
            m_matrixV.resize(n, k);

            for (Index i = 0; i < k; ++i) {
                // Extract the largest singular value/vector via power method
                Scalar sigma;
                DenseVector u(m), v(n);
                bool success = extractLargestSingularValue(A, sigma, u, v, maxIters, tol);

                if (!success) {
                    // If we fail to converge, we just break early
                    m_singularValues.conservativeResize(i);
                    m_matrixU.conservativeResize(m, i);
                    m_matrixV.conservativeResize(n, i);
                    break;
                }

                m_singularValues(i) = sigma;
                m_matrixU.col(i) = u;
                m_matrixV.col(i) = v;

                // Deflate the matrix
                A = A - sigma * u * v.transpose();
            }

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
         * @brief Use power method to find the largest singular value/vector of A
         *
         * This finds the largest singular value sigma, and corresponding left and right
         * singular vectors u, v of A.
         *
         * Process:
         * 1. Start with a random vector v.
         * 2. Compute u = A * v, normalize u.
         * 3. Compute v = A^T * u, normalize v.
         * 4. Repeat until convergence or maxIters.
         *
         * Once converged, sigma = u^T * (A * v).
         *
         * @param A Input matrix
         * @param sigma (output) largest singular value
         * @param u (output) left singular vector
         * @param v (output) right singular vector
         * @param maxIters Maximum iterations
         * @param tol Convergence tolerance
         * @return True if converged, false otherwise
         */
        bool extractLargestSingularValue(const DenseMatrix &A,
                                         Scalar &sigma,
                                         DenseVector &u,
                                         DenseVector &v,
                                         Index maxIters,
                                         Scalar tol) {
            Index m = A.rows();
            Index n = A.cols();

            // Initialize v with random values
            v = randomVector(n);
            v.normalize();

            // Temporary vectors
            DenseVector u_new(m), v_new(n);
            sigma = 0;

            for (Index iter = 0; iter < maxIters; ++iter) {
                // Compute u = A * v
                u_new = A * v;
                Scalar u_norm = u_new.norm();
                if (u_norm < tol) {
                    return false; // A * v is zero vector, no meaningful singular value
                }
                u_new /= u_norm;

                // Compute v = A^T * u
                v_new = A.transpose() * u_new;
                Scalar v_norm = v_new.norm();
                if (v_norm < tol) {
                    return false; // A^T * u is zero, can't proceed
                }
                v_new /= v_norm;

                // Check convergence of v
                Scalar diff = (v_new - v).norm();
                v = v_new;
                u = u_new;

                if (diff < tol) {
                    // Converged
                    break;
                }

                if (iter == maxIters - 1) {
                    // Did not converge
                    return false;
                }
            }

            // Once converged, compute sigma = ||A * v||
            DenseVector Av = A * v;
            sigma = Av.norm();

            // Ensure consistency: we want u = (A * v) / sigma
            if (sigma > Scalar(0)) {
                u = Av / sigma;
            } else {
                return false;
            }

            return true;
        }

        /**
         * @brief Generate a random vector of length n
         */
        DenseVector randomVector(Index n) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<Scalar> dist(0.0, 1.0);

            DenseVector vec(n);
            for (Index i = 0; i < n; ++i) {
                vec(i) = dist(gen);
            }
            return vec;
        }

        DenseMatrix m_matrixU;
        DenseMatrix m_matrixV;
        DenseVector m_singularValues;
    };

}
#endif //POWERMETHODSVD_H
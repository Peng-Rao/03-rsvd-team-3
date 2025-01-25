#ifndef POWERMETHODSVD_H
#define POWERMETHODSVD_H

#include "MatrixTraits.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

namespace Eigen {
/**
 * @brief Base class template (default for dense matrices)
 */
template<typename MatrixType, bool IsSparse = is_sparse_matrix<MatrixType>::value>
class PowerMethodSVD
{
public:
    using Scalar      = typename MatrixType::Scalar;
    using Index       = typename MatrixType::Index;
    using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
    using DenseVector = Matrix<Scalar, Dynamic, 1>;

    PowerMethodSVD() = default;

    /**
     * @brief Compute the SVD of a dense matrix up to the given rank using a
     *        one-at-a-time power method + deflation + re-orthonormalization.
     *
     * @param matrix   The input dense matrix to decompose
     * @param rank     Desired rank of the approximation
     * @param maxIters Maximum iterations for each singular value extraction
     * @param tol      Tolerance for convergence
     */
    PowerMethodSVD &compute(const MatrixType &matrix,
                            Index rank,
                            Index maxIters = 1000,
                            Scalar tol = Scalar(1e-6))
    {
        // Copy the matrix for deflation
        DenseMatrix A = matrix.template cast<Scalar>();
        Index m = A.rows();
        Index n = A.cols();
        Index k = std::min(rank, std::min(m, n));

        m_singularValues.resize(k);
        m_matrixU.resize(m, k);
        m_matrixV.resize(n, k);

        for (Index i = 0; i < k; ++i)
        {
            // Extract the largest singular value/vector, re-orthogonalizing
            // against previously found singular vectors
            Scalar sigma;
            DenseVector u(m), v(n);

            bool success = extractLargestSingularValue(
                A, sigma, u, v, maxIters, tol,
                m_matrixU, m_matrixV, i  // pass in existing columns
            );

            if (!success)
            {
                // If we fail to converge, shrink the storage and break
                m_singularValues.conservativeResize(i);
                m_matrixU.conservativeResize(m, i);
                m_matrixV.conservativeResize(n, i);
                break;
            }

            m_singularValues(i) = sigma;
            m_matrixU.col(i)    = u;
            m_matrixV.col(i)    = v;

            // Deflate the matrix: A <- A - sigma * u * v^T
            A.noalias() -= sigma * (u * v.transpose());
        }

        return *this;
    }

    /**
     * @brief Get the singular values
     */
    const DenseVector &singularValues() const { return m_singularValues; }

    /**
     * @brief Get the left singular vectors (U matrix)
     */
    const DenseMatrix &matrixU() const { return m_matrixU; }

    /**
     * @brief Get the right singular vectors (V matrix)
     */
    const DenseMatrix &matrixV() const { return m_matrixV; }

private:

    /**
     * @brief Use power method to find the largest singular value/vector of A (Dense),
     *        re-orthonormalizing the new vectors against previously found ones.
     *
     * @param A         Input dense matrix (already deflated up to current i)
     * @param sigma     (output) largest singular value
     * @param u         (output) left singular vector
     * @param v         (output) right singular vector
     * @param maxIters  Maximum iterations
     * @param tol       Convergence tolerance
     * @param Uprev     The U matrix so far (columns = previously found left sing. vectors)
     * @param Vprev     The V matrix so far (columns = previously found right sing. vectors)
     * @param numFound  How many singular vectors have been found so far
     *
     * @return True if converged, false otherwise
     */
    bool extractLargestSingularValue(const DenseMatrix &A,
                                     Scalar &sigma,
                                     DenseVector &u,
                                     DenseVector &v,
                                     Index maxIters,
                                     Scalar tol,
                                     const DenseMatrix &Uprev,
                                     const DenseMatrix &Vprev,
                                     Index numFound)
    {
        Index m = A.rows();
        Index n = A.cols();

        // Initialize v with random values, then orthonormalize it
        v = randomVector(n);
        reorthonormalizeVector(v, Vprev, numFound);
        if(v.norm() < tol) return false;
        v.normalize();

        DenseVector u_new(m), v_new(n);

        sigma = Scalar(0);

        for (Index iter = 0; iter < maxIters; ++iter)
        {
            // 1) Compute u_new = A * v
            u_new.noalias() = A * v;

            //    Re-orthonormalize u_new against previously found columns of Uprev
            reorthonormalizeVector(u_new, Uprev, numFound);

            Scalar u_norm = u_new.norm();
            if (u_norm < tol)
            {
                // A*v is nearly zero => no meaningful singular vector
                return false;
            }
            u_new /= u_norm;

            // 2) Compute v_new = A^T * u_new
            v_new.noalias() = A.transpose() * u_new;

            //    Re-orthonormalize v_new against previously found columns of Vprev
            reorthonormalizeVector(v_new, Vprev, numFound);

            Scalar v_norm = v_new.norm();
            if (v_norm < tol)
            {
                // A^T*u_new is nearly zero => can't proceed
                return false;
            }
            v_new /= v_norm;

            // Check convergence of v
            Scalar diff = (v_new - v).norm();
            v = v_new;
            u = u_new;

            if (diff < tol)
            {
                // Converged
                break;
            }

            if (iter == maxIters - 1)
            {
                // Did not converge within maxIters
                return false;
            }
        }

        // Once converged, compute sigma = ||A * v||
        DenseVector Av = A * v;
        sigma = Av.norm();

        // Ensure consistency: we want u = (A * v) / sigma
        if (sigma > Scalar(0))
        {
            u = Av / sigma;
            // Re-orthonormalize one last time
            reorthonormalizeVector(u, Uprev, numFound);
            u.normalize();
        }
        else
        {
            return false;
        }

        return true;
    }

    /**
     * @brief Re-orthonormalize 'vec' against the first 'numCols' columns of 'basis'.
     *
     * This step is crucial to keep new singular vectors orthogonal to previously found ones
     * in finite precision.  We do a simple modified Gram-Schmidt loop:
     *
     *    vec <- vec - (basis.col(j)^T vec) * basis.col(j)
     *
     * for j=0..(numCols-1).
     */
    void reorthonormalizeVector(DenseVector &vec,
                                const DenseMatrix &basis,
                                Index numCols) const
    {
        for (Index j = 0; j < numCols; ++j)
        {
            Scalar proj = basis.col(j).dot(vec);
            vec -= proj * basis.col(j);
        }
    }

    /**
     * @brief Generate a random vector of length n using normal distribution
     */
    DenseVector randomVector(Index n) const
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<Scalar> dist(0.0, 1.0);

        DenseVector vec(n);
        for (Index i = 0; i < n; ++i)
            vec(i) = dist(gen);
        return vec;
    }

    DenseMatrix m_matrixU;
    DenseMatrix m_matrixV;
    DenseVector m_singularValues;
};

// Specialization for Sparse Matrices
template<typename MatrixType>
class PowerMethodSVD<MatrixType, true> {
public:
    using Scalar = typename MatrixType::Scalar;
    using Index = typename MatrixType::Index;
    using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
    using DenseVector = Matrix<Scalar, Dynamic, 1>;

    // Determine storage order based on MatrixType
    static constexpr int StorageOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor;

    using SparseMatrixType = SparseMatrix<Scalar, StorageOrder, Index>;

    PowerMethodSVD() = default;

    /**
     * @brief Compute the SVD of a sparse matrix up to the given rank using power method + deflation
     *
     * @param matrix The input sparse matrix to decompose
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
        SparseMatrixType A_sparse = matrix;
        A_sparse.makeCompressed();

        Index m = A_sparse.rows();
        Index n = A_sparse.cols();
        Index k = std::min(rank, std::min(m, n));

        m_singularValues.resize(k);
        m_matrixU.resize(m, k);
        m_matrixV.resize(n, k);

        for (Index i = 0; i < k; ++i) {
            // Extract the largest singular value/vector via power method
            Scalar sigma;
            DenseVector u(m), v(n);
            bool success = extractLargestSingularValueSparse(A_sparse, sigma, u, v, maxIters, tol);

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
            // Subtract sigma * u * v.transpose() from A_sparse
            // Since u and v are dense, compute their outer product and convert to sparse
            DenseMatrix uvT = u * v.transpose();
            SparseMatrixType uvT_sparse = uvT.sparseView();

            A_sparse = A_sparse - sigma * uvT_sparse;
            A_sparse.prune(1e-12); // Remove near-zero elements to maintain sparsity
            A_sparse.makeCompressed();
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
     * @brief Use power method to find the largest singular value/vector of A (Sparse)
     *
     * @param A Input sparse matrix
     * @param sigma (output) largest singular value
     * @param u (output) left singular vector
     * @param v (output) right singular vector
     * @param maxIters Maximum iterations
     * @param tol Convergence tolerance
     * @return True if converged, false otherwise
     */
    bool extractLargestSingularValueSparse(const SparseMatrixType &A,
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

} // namespace Eigen

#endif // POWERMETHODSVD_H
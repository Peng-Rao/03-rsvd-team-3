#ifndef GIVENSROTATIONQR_H
#define GIVENSROTATIONQR_H

#include <Eigen/Core>

namespace Eigen {

/**
 * @brief A class to perform QR decomposition using Given rotations
 */
template<typename MatrixType>
class GivensRotationQR {
public:
    using Scalar = typename MatrixType::Scalar;
    using Index = typename MatrixType::Index;
    using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

    /**
     * @brief Default constructor
     */
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
        m_matrixR = matrix; // Copy input matrix
        m_matrixQ = DenseMatrix::Identity(rows, rows); // Initialize Q as identity

        // Perform Given rotations
        for (Index j = 0; j < cols; ++j) {
            for (Index i = rows - 1; i > j; --i) {
                Scalar a = m_matrixR(i - 1, j);
                Scalar b = m_matrixR(i, j);
                Scalar c, s;
                computeGivensRotation(a, b, c, s);

                // Apply the rotation to R
                applyGivensRotation(m_matrixR, i - 1, i, c, s);

                // Apply the rotation to Q
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
        // Create temporary copies of the rows to prevent data dependency issues
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

    DenseMatrix m_matrixQ; // Orthogonal matrix Q
    DenseMatrix m_matrixR; // Upper triangular matrix R
};

} // namespace Eigen
#endif // GIVENSROTATIONQR_H
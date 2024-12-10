#include <iostream>
#include <vector>
#include <cmath>
#include <ranges>

// Function to perform the Givens rotation for two elements a and b
void givensRotation(double a, double b, double& c, double& s) {
    double r = std::sqrt(a * a + b * b); //distance
    c = a / r; //cosine
    s = -b / r; //sine
}
//size_t is an unsigned integer type that represents the size of an object or the index of an array

// Function to apply the Givens rotation to a matrix A
void applyGivensRotation(std::vector<std::vector<double>>& A, size_t i, size_t j, double c, double s) {
    //for cycle across columns of A, rotate two dimensions each time and fix all other dimensions
    for (size_t k = 0; k < A[0].size(); ++k) {
        double temp1 = c * A[i][k] - s * A[j][k];
        double temp2 = s * A[i][k] + c * A[j][k];
        A[i][k] = temp1;
        A[j][k] = temp2;
    }
}

// Function to perform QR decomposition using Givens rotations
void qrFactorizationGivenRotation(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R) {
    size_t m = A.size(); //number of rows
    size_t n = A[0].size(); //number of columns
    
    // Initialize R as a copy of A
    R = A;
    
    // Initialize Q as an identity matrix
    Q.resize(m, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < m; ++i) {
        Q[i][i] = 1.0;
    }
    
    // Perform Givens rotations to zero out the lower triangular elements of R
    for (size_t j = 0; j < n; ++j) { //iteration across columns
        for (size_t i = m - 1; i > j; --i) { //iteration across rows for each column starting from the lowest position in the column
            //check if subdiagonal value is already zero and exit for cycle!!
            double c, s;
            givensRotation(R[j][j], R[i][j], c, s); //compute given rotation using pivot element and the element
                                                    //that needs to be reduced to zero
            
            // Apply the Givens rotation to both R and Q
            applyGivensRotation(R, i - 1, i, c, s);
            applyGivensRotation(Q, i - 1, i, c, s);
        }
    }
}

// Function to print a matrix
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example matrix A (3x3 matrix)
    std::vector<std::vector<double>> A = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}
    };
    std::cout << "Matrix A:" << std::endl;
    printMatrix(A);
    // Vectors to store Q and R
    std::vector<std::vector<double>> Q, R;
    
    // Perform QR factorization using Givens rotations
    qrFactorizationGivenRotation(A, Q, R);
    
    // Output Q and R
    std::cout << "Matrix Q (orthogonal/rotation):" << std::endl;
    printMatrix(Q);
    
    std::cout << "Matrix R (upper triangular/scaling):" << std::endl;
    printMatrix(R);
    
    // Optionally, reconstruct A from Q * R
    std::vector<std::vector<double>> A_reconstructed(A.size(), std::vector<double>(A[0].size(), 0));
    
    // Matrix multiplication Q * R
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            for (size_t k = 0; k < A[0].size(); ++k) {
                A_reconstructed[i][j] += Q[i][k] * R[k][j];
            }
        }
    }
    
    std::cout << "Reconstructed A (from Q * R):" << std::endl;
    printMatrix(A_reconstructed);

    return 0;
}
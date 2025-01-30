//
// Created by 黄家励 on 2025/1/30.
//

#ifndef PCA_H
#define PCA_H

#endif //PCA_H

#include <iostream>
#include <vector>
#include <Eigen/dense>
#include "RandomizedSVD.h"

using namespace std;

namespace Eigen{
    class PCA {
    public:
        PCA() = default;
        PCA &compute(const MatrixXd &data, int dimension){
            dataReduced(data, dimension);
            return *this;
        }
        PCA &computeBySVD(const MatrixXd &data, int dimension){
            dataReducedBySVD(data, dimension);
            return *this;
        }
        PCA &computeByRSVD(const MatrixXd &data, int dimension, int rank){
            dataReducedByRSVD(data, dimension, rank);
            return *this;
        }
        const MatrixXd &reducedMatrix() const {return reducedMat;}
        ~PCA() = default;

    private:
        MatrixXd reducedMat;
        void dataReduced(const MatrixXd &data, int dimension) {
            // Step 1: data centralisation
            Eigen::VectorXd mean = data.colwise().mean(); // get mean value of each column
            Eigen::MatrixXd centered = data.rowwise() - mean.transpose(); // each row detract mean

            // Step 2: Compute the covariance matrix
            MatrixXd cov_matrix = (centered.adjoint() * centered) / (data.rows() - 1);

            // Step 3: get eigenvalue and eigenvector of covariance matrix
            SelfAdjointEigenSolver<MatrixXd> eigen_solver(cov_matrix);
            VectorXd eigen_values = eigen_solver.eigenvalues();
            MatrixXd eigen_vectors = eigen_solver.eigenvectors();

            // Step 4: ordering eigenvectors by eigenvalue
            vector<pair<double, VectorXd>> eigen_pairs;
            for (int i = 0; i < eigen_values.size(); ++i) {
                eigen_pairs.emplace_back(eigen_values(i), eigen_vectors.col(i));
            }
            sort(eigen_pairs.rbegin(), eigen_pairs.rend(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            // Step 5: get previous num components
            MatrixXd projection_matrix(data.cols(), dimension);
            for (std::vector<std::pair<double, Eigen::VectorXd>>::size_type i = 0; i < dimension; ++i) {
                projection_matrix.col(i) = eigen_pairs[i].second;
            }

            // Step 6: data projection
            this->reducedMat = centered * projection_matrix;

        }
        void dataReducedBySVD(const MatrixXd &data, int dimension) {
            // Step 1: data centralisation
            Eigen::VectorXd mean = data.colwise().mean(); // get mean value of each column
            Eigen::MatrixXd centered = data.rowwise() - mean.transpose(); // each row detract mean

            // Step 2: Compute the eigen vector by SVD
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

            // Step 3: get principal components
            Eigen::MatrixXd principal_components = svd.matrixV().leftCols(dimension);

            // Step 4: data projection
            this->reducedMat = centered * principal_components;
        }

        void dataReducedByRSVD(const MatrixXd &data, int dimension, int rank) {
            if(dimension > rank) {
                cout << "Warning: Reduced dimension must be smaller and equal to random matrix size." << endl;
                return;
            }
            // Step 1: data centralisation
            Eigen::VectorXd mean = data.colwise().mean();  // get mean value of each col
            Eigen::MatrixXd centered = data.rowwise() - mean.transpose();

            // Step 2: Compute the eigen vector by rSVD
            Eigen::RandomizedSVD<Eigen::MatrixXd> rSVD;
            rSVD.compute(centered, rank);

            // Step 3: get principal components
            Eigen::MatrixXd principal_components = rSVD.matrixV().leftCols(dimension);

            // Step 4: data projection
            this->reducedMat = centered * principal_components;
        }


    };


}

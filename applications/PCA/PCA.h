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

            // Step 2: Compute the eigen vector by SVD
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);

            // Step 3: get principal components
            Eigen::MatrixXd principal_components = svd.matrixV().leftCols(dimension);

            // Step 4: data projection
            this->reducedMat = centered * principal_components;
        }

        void dataReducedByRSVD(const MatrixXd &data, int dimension, int rank) {
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

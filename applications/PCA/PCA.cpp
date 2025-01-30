//
// Created by 黄家励 on 2025/1/27.
//
#include <iostream>
#include "PCA.h"
using namespace std;


int main(){
    // data example
    Eigen::MatrixXd data(5, 3);
    data << 2.5, 2.4, 1.5,
            0.5, 0.7, 0.3,
            2.2, 2.9, 1.6,
            1.9, 2.2, 1.1,
            3.1, 3.0, 1.8;
    cout << "orginal data:\n" << data << endl;
    Eigen::MatrixXd row = data.rowwise();
    cout << data.rowwise() << endl;
    Eigen::PCA pca_by_rsvd;
    pca_by_rsvd.computeByRSVD(data,2, 3);
    // reduced data
    Eigen::MatrixXd reduced_data1 = pca_by_rsvd.reducedMatrix();
    cout << "reduced data:\n" << reduced_data1 << endl;

    return 0;
}
//
// Created by 黄家励 on 2025/1/27.
//
#include <iostream>
#include <ctime>
#include "PCA.h"
using namespace std;


int main(){
    // data example
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(100,500);
    int dimension = 50;
    int rank = dimension;

    clock_t start, end;
    double elapse;


    Eigen::PCA pca;
    start = clock();
    pca.compute(data,dimension);
    end = clock();
    elapse = double(end - start) / CLOCKS_PER_SEC * 1000;
    Eigen::MatrixXd reduced_data = pca.reducedMatrix();
    // cout << "reduced data:\n" << reduced_data << endl << endl;
    cout << "Common PCA time elapse on: "<< elapse << " ms" << endl;


    Eigen::PCA pca_by_svd;
    start = clock();
    pca_by_svd.computeBySVD(data,dimension);
    end = clock();
    elapse = double(end - start) / CLOCKS_PER_SEC * 1000;
    Eigen::MatrixXd reduced_data1 = pca.reducedMatrix();
    // cout << "reduced data:\n" << reduced_data1 << endl << endl;
    cout << "PCA by SVD time elapse on: "<< elapse << " ms" << endl;


    Eigen::PCA pca_by_rsvd;
    start = clock();
    pca_by_rsvd.computeByRSVD(data,dimension, rank);
    end = clock();
    elapse = double(end - start) / CLOCKS_PER_SEC * 1000;
    Eigen::MatrixXd reduced_data2 = pca_by_rsvd.reducedMatrix();
    // cout << "reduced data:\n" << reduced_data2 << endl;
    cout << "PCA by rSVD time elapse on: "<< elapse << " ms" << endl;



    return 0;
}
#pragma once
#include <iostream>
#include "VectorMatrix.h"
using namespace std;

#define INF 1000000000

double MSE(vi y, vi y0) {
    vi tmp = y - y0;
    double sum = 0;
    for (auto t : tmp)
        sum += t * t;
    return (1. / y.size()) * sum;
}

vd gradientMSE(vvi AT, vi dy) {
    return (AT * dy) * (2. / dy.size());
}

int Algo1(vvi A, vi y0, vi& x, vd& error, int xmin = 0, int step = 1, int K = 1000000) {

    if (A.empty() || y0.empty())
        throw std::invalid_argument("Wrong argument!");

    if (A.size() != y0.size())
        throw std::out_of_range("Wrong vector size!");

    int maxM = getMaxMatrixEl(A), minM = getMinMatrixEl(A);
    if (maxM < 0 || maxM > 1 || minM < 0 || minM > 1)
        throw std::out_of_range("Wrong matriA value range!");

    if (getMinVectorEl(y0) < 0)
        throw std::out_of_range("Wrong vector value range!");

    if (step <= 0)
        throw std::out_of_range("Wrong step value!");

    if (xmin < 0)
        throw std::out_of_range("Wrong wmin value!");

    int n = A.size();
    int m = A[0].size();

    error.resize(0);

    vvi AT = matrixTranspose(A);

    x.resize(A[0].size(), xmin);
    for (int i = 0; i < m; i++) {
        int max = xmin;
        for (int j = 0; j < n; j++) {
            int tmp = A[j][i] * y0[j];
            if (tmp > max)
                max = tmp;
        }
        x[i] = max;
    }

    for(int k=0; k<K; k++) {
        
        vi y = A * x;
        vd curGrad = gradientMSE(AT, (y - y0));

        error.push_back(MSE(y, y0));

        int indW = -1;
        double maxDGrad = -1;

        for (int i = 0; i < m; i++) {
            if (x[i] - step >= xmin) {
                x[i] -= step;
                vi tmp = A * x - y0;
                if (getMinVectorEl(tmp) >= 0) {
                    double tmpDGrad = L1(gradientMSE(AT, tmp) - curGrad);
                    // Выбираем wi с наибольшим прирощением произврлной
                    if (tmpDGrad > maxDGrad)
                        indW = i, maxDGrad = tmpDGrad;
                }
                x[i] += step;
            }
        }

        if (indW == -1) {
            return k;
        }

        x[indW] -= step;
    }
}

int Algo2(vvi A, vi y0, vi& x, vd& error, int xmin = 0, int K = 1000000) {

    if (A.empty() || y0.empty())
        throw std::invalid_argument("Wrong argument!");

    if (A.size() != y0.size())
        throw std::out_of_range("Wrong vector size!");

    int maxM = getMaxMatrixEl(A), minM = getMinMatrixEl(A);
    if (maxM < 0 || maxM > 1 || minM < 0 || minM > 1)
        throw std::out_of_range("Wrong matriA value range!");

    if (getMinVectorEl(y0) < 0)
        throw std::out_of_range("Wrong vector value range!");

    if (xmin < 0)
        throw std::out_of_range("Wrong wmin value!");

    int n = A.size();
    int m = A[0].size();
    int step;

    error.resize(0);

    vvi AT = matrixTranspose(A);

    x.resize(A[0].size(), xmin);
    for (int i = 0; i < m; i++) {
        int max = xmin;
        for (int j = 0; j < n; j++) {
            int tmp = A[j][i] * y0[j];
            if (tmp > max)
                max = tmp;
        }
        x[i] = max;
    }

    for (int k = 0; k < K; k++) {

        vi y = A * x;
        vd curGrad = gradientMSE(AT, (y - y0));

        error.push_back(MSE(y, y0));
        iter++;

        int indW = -1;
        double maxDGrad = -1;
        int stepW = 0;

        for (int i = 0; i < m; i++) {
            vi yTmp(n, INF);
            for (int j = 0; j < n; j++) {
                if (A[j][i]) yTmp[j] = y[j];
            }
            step = min(x[i] - xmin, getMinVectorEl(yTmp - y0));

            if (step > 0) {
                x[i] -= step;
                vi tmp = A * x - y0;
                if (getMinVectorEl(tmp) >= 0) {
                    double tmpDGrad = L1(gradientMSE(AT, tmp) - curGrad);
                    if (tmpDGrad > maxDGrad)
                        indW = i, maxDGrad = tmpDGrad, stepW = step;
                }
                x[i] += step;
            }
        }

        if (indW == -1) {
            return k;
        }

        x[indW] -= stepW;
    }
}

int Algo3(vvi A, vi y0, vi& x, vd& error, int xmin = 0, int step = 1, int K = 1000000) {

    if (A.empty() || y0.empty())
        throw std::invalid_argument("Wrong argument!");

    if (A.size() != y0.size())
        throw std::out_of_range("Wrong vector size!");

    int maAM = getMaxMatrixEl(A), minM = getMinMatrixEl(A);
    if (maAM < 0 || maAM > 1 || minM < 0 || minM > 1)
        throw std::out_of_range("Wrong matriA value range!");

    if (getMinVectorEl(y0) < 0)
        throw std::out_of_range("Wrong vector value range!");

    if (step <= 0)
        throw std::out_of_range("Wrong step value!");

    if (xmin < 0)
        throw std::out_of_range("Wrong wmin value!");

    int n = A.size();
    int m = A[0].size();

    error.resize(0);

    x.resize(A[0].size(), xmin);
    for (int i = 0; i < m; i++) {
        int max = xmin;
        for (int j = 0; j < n; j++) {
            int tmp = A[j][i] * y0[j];
            if (tmp > max)
                max = tmp;
        }
        x[i] = max;
    }

    for (int k = 0; k < K; k++) {


         vi y = A * x;

        error.push_back(MSE(y, y0));

        int indW = -1;
        int sumW = -1;

        for (int i = 0; i < m; i++) {
            if (x[i] - step >= xmin) {
                x[i] -= step;
                vi tmp = A * x - y0;
                if (getMinVectorEl(tmp) >= 0) {
                    int tmp = 0;
                    for (int j = 0; j < n; j++)
                        if (A[j][i]) tmp += 1;
                    if (tmp > sumW) indW = i, sumW = tmp;
                }
                x[i] += step;
            }
        }

        if (indW == -1) {
            return k;
        }

        x[indW] -= step;
    }
}
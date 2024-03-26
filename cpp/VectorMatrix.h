#pragma once
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

typedef vector<vector<int>> vvi;
typedef vector<int> vi;
typedef vector<vector<double>> vvd;
typedef vector<double> vd;

template <typename T>
std::istream& operator >> (std::istream& in, vector<vector<T>>& m) {
    const size_t rows = m.size();
    const size_t columns = m[0].size();
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != columns; ++j) {
            in >> m[i][j];
        }
    }
    return in;
}

template <typename T>
std::istream& operator >> (std::ifstream& fin, vector<vector<T>>& m) {
    const size_t rows = m.size();
    const size_t columns = m[0].size();
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != columns; ++j) {
            fin >> m[i][j];
        }
    }
    return fin;
}

template <typename T>
std::ostream& operator << (std::ostream& out, vector<vector<T>>& m) {
    const size_t rows = m.size();
    const size_t columns = m[0].size();
    for (size_t i = 0; i != rows; ++i) {
        for (size_t j = 0; j != columns; ++j) {
            out << m[i][j] << "\t";
        }
        out << "\n";
    }
    return out;
}

template <typename T>
std::istream& operator >> (std::istream& in, vector<T>& m) {
    const size_t rows = m.size();
    for (size_t i = 0; i != rows; ++i) 
        in >> m[i];
    return in;
}

template <typename T>
std::istream& operator >> (std::ifstream& fin, vector<T>& m) {
    const size_t rows = m.size();
    for (size_t i = 0; i != rows; ++i)
            fin >> m[i];
    return fin;
}

template <typename T>
std::ostream& operator << (std::ostream& out, vector<T> m) {
    const size_t rows = m.size();
    for (size_t i = 0; i != rows; ++i) 
        out << m[i] << " ";
        out << "\n";
    return out;
}

template <typename T>
std::ostream& operator << (std::ofstream& fout, vector<T> m) {
    const size_t rows = m.size();
    for (size_t i = 0; i != rows; ++i)
        fout << m[i] << "\n";
    return fout;
}

vi getMatrixRow(vvi m, int row) {
    int n = m[0].size();
    vi res(n);
    for (int i = 0; i < n; i++)
        res[i] = m[row][i];
    return res;
}

vi getMatrixCol(vvi m, int col) {
    int n = m.size();
    vi res(n);
    for (int i = 0; i < n; i++)
        res[i] = m[i][col];
    return res;
}

vi operator * (vvi m, vi v) {
    if (m[0].size() != v.size())
        throw std::invalid_argument("Vector has incorect size!");

    vi res(m.size());
    for (int i = 0; i < m.size(); i++) {
        res[i] = 0;
        for (int j = 0; j < m[0].size(); j++)
            res[i] += m[i][j] * v[j];
    }
    return res;
}

vi operator * (vi m1, int val) {
    vector<int> tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] * val;
    return tmp;
}

vi operator * (int val, vi m1) {
    vector<int> tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] * val;
    return tmp;
}

vd operator * (vi m1, double val) {
    vd tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] * val;
    return tmp;
}

int operator * (vi m1, vi m2) {
    int tmp = 0;
    for (size_t i = 0; i < m1.size(); i++)
        tmp += m1[i] * m2[i];
    return tmp;
}

vd operator * (double val, vi m1) {
    vd tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] * val;
    return tmp;
}

vi operator - (vi m1, vi m2) {
    if(m1.size()!=m2.size())
        throw std::invalid_argument("Vectores have incorect size!");
    vi tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] - m2[i];
    return tmp;
}

vd operator - (vd m1, vd m2) {
    if (m1.size() != m2.size())
        throw std::invalid_argument("Vectores have incorect size!");
    vd tmp(m1.size());
    for (size_t i = 0; i < m1.size(); i++)
        tmp[i] = m1[i] - m2[i];
    return tmp;
}

template <typename T>
T L1(vector<T> v) {
    T sum = 0;
    for (auto i : v)
        sum += abs(i);
    return sum;
}

template <typename T>
double L2(vector<T> v) {
    int sum = 0;
    for (auto i : v)
        sum += (i*i);
    return sqrt(sum);
}

vvi matrixTranspose(vvi X) {
    int n = X.size();
    int m = X[0].size();

    vvi res(m, vi(n, 0));

    for (int j = 0; j < m; j++)
        for (int i = 0; i < n; i++)
            res[j][i] = X[i][j];

    return res;
}

int getMaxMatrixEl(vvi m) {
    int res = m[0][0];
    for (int i = 0; i < m.size(); i++)
        for (int j = 0; j < m[0].size(); j++)
            if (res < m[i][j])
                res = m[i][j];
    return res;
}

int getMinMatrixEl(vvi m) {
    int res = m[0][0];
    for (int i = 0; i < m.size(); i++)
        for (int j = 0; j < m[0].size(); j++)
            if (res > m[i][j])
                res = m[i][j];
    return res;
}

template <typename T>
T getMaxVectorEl(vector<T> m) {
    T res = m[0];
    for (int i = 0; i < m.size(); i++)
            if (res < m[i])
                res = m[i];
    return res;
}

template <typename T>
T getMinVectorEl(vector<T> m) {
    T res = m[0];
    for (int i = 0; i < m.size(); i++)
            if (res > m[i])
                res = m[i];
    return res;
}
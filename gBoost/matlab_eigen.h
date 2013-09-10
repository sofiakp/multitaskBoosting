#ifndef MATLAB_EIGEN_H_
#define MATLAB_EIGEN_H_

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "mex.h"
#include "matrix.h"

#define EIGEN_DONT_PARALLELIZE // prevent Eigen from using OpenMP to multithread its operations

using std::cout;
using std::endl;

using Eigen::SparseMatrix;
using Eigen::MappedSparseMatrix;
using Eigen::ColMajor;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::MatrixBase;
using Eigen::Triplet;

/*

  MatrixXd is Eigen's dense matrix class
  mySparseMatrix is a typedef of Eigen's sparse matrix class
  
  myMappedSparseMatrix is a mapped version of mySparseMatrix
  Map<MatrixXd> is a mapped version of MatrixXd

  Sparse3DArray is a sparse 3D array (with sparse matrices as slices)
  MappedSparse3DArray is a mapped Sparse3DArray

*/
typedef SparseMatrix<double,ColMajor,long> mySparseMatrix;
typedef MappedSparseMatrix<double,ColMajor,long> myMappedSparseMatrix;

/*
   Functions for converting between Matlab and Eigen matrices, for full and
   sparse matrices, and mapped and not-mapped. Definitions in matlab_eigen.cpp.
   Mapped indicates that no data is copied, pointers to the Matlab data are
   used. Un-mapped indicates that the data is copied. Mapped is definitely need
   for the sometimes huge motif feature 3D array.
*/

mySparseMatrix spm_matlab2eigen(const mxArray* spm_matlab); 
MatrixXd full_matlab2eigen(const mxArray* full_matlab);
const myMappedSparseMatrix spm_matlab2eigen_mapped(const mxArray* spm_matlab);
Map<MatrixXd> full_matlab2eigen_mapped(const mxArray* full_matlab);
mxArray* eigen2matlab_full(const MatrixXd& full_eigen);

/*
  Sparse3DArray and MappedSparse3DArray classes. These are used for the 3D
  motif features U. They are constructed from Matlab cell arrays of sparse
  double matrices.
*/

// non-mapped Sparse 3D Array class, actually copies data
class Sparse3DArray {
  public:
    std::vector<mySparseMatrix> x;
    int m,n,p;

    Sparse3DArray(const mxArray* matlab);
    Sparse3DArray();

    template <typename derived>
      MatrixXd operator*(const MatrixBase<derived>& D) const;

    MatrixXd spdmm_square_first(const MatrixXd& Y) const;

};

// mapped Sparse 3D Array class, just uses pointers to original Matlab data
class MappedSparse3DArray {
public:
    std::vector<myMappedSparseMatrix> x;
    int m,n,p;

    MappedSparse3DArray(const mxArray* matlab);
    MappedSparse3DArray();

    template <typename derived>
      MatrixXd operator*(const MatrixBase<derived>& D) const;
    
    VectorXd get_col(int i, int j) const;
    MatrixXd spdmm_square_first(const MatrixXd& Y) const;
};


inline Sparse3DArray::Sparse3DArray() {
}

// make sparse3DArray from a Matlab cell array of sparse double matrices. The
// data is copied.
inline Sparse3DArray::Sparse3DArray(const mxArray* matlab) {
  p = mxGetM(matlab);
  m = mxGetM(mxGetCell(matlab,0));
  n = mxGetN(mxGetCell(matlab,0));
  x.reserve(p);
  for (int i=0; i<p; i++) {
    // copies data
    x.push_back( spm_matlab2eigen(mxGetCell(matlab,i)));
  }

}

// Implements specialized multiplication between a Sparse 3D Array and a normal
// matrix. The dimensions should be: (m x n x p) * (n x p) ==> (m x p), and the
// computation is: M(i,k) = sum_j X(i,j,k) * D(j,k) <==> M = X*U. This is used
// to sum over genes (j index) when the motif features are a function of the
// gene and the experiment.
template <typename derived>
inline MatrixXd Sparse3DArray::operator*(const MatrixBase<derived>& D) const {
  MatrixXd retval = MatrixXd::Zero(m,p);
  for (int j=0; j<p; j++) {
    retval.col(j) = x[j]*D.col(j);
  }   
  return retval;
}

// X.spdmm_square_first(Y) = (X.^2)*Y, but without having to make a copy of the
// entire X.^2, which is up to 2GB for our data. Used in Newton Boosting equations.
inline MatrixXd Sparse3DArray::spdmm_square_first(const MatrixXd& Y) const {
  MatrixXd retval = MatrixXd::Zero(m,p);
  for (int j=0; j<p; j++) {
    retval.col(j) = (x[j].cwiseAbs2())*Y.col(j);
  }   
  return retval;
}

inline MappedSparse3DArray::MappedSparse3DArray() {
}

// make mappedSparse3DArray from a Matlab cell array of sparse double matrices.
// The data is not copied, pointers point to the origial Matlab data.
inline MappedSparse3DArray::MappedSparse3DArray(const mxArray* matlab) {
  p = mxGetM(matlab);
  m = mxGetM(mxGetCell(matlab,0));
  n = mxGetN(mxGetCell(matlab,0));
  x.reserve(p);
  for (int i=0; i<p; i++) {
    // does not copy data
    x.push_back( spm_matlab2eigen_mapped(mxGetCell(matlab,i)));
  }
}

// same as Sparse3DArray::operator* but for the mapped MappedSparse3DArray
template <typename derived>
inline MatrixXd MappedSparse3DArray::operator*(const MatrixBase<derived>& D) const {
  // (m x n x p) * (n x p) ==> (m x p)
  // M(i,k) = sSparseMatrix<double,ColMajor,long>um_j X(i,j,k) * D(j,k)A
  MatrixXd retval = MatrixXd::Zero(m,p);
  for (int j=0; j<p; j++) {
    retval.col(j) = x[j]*D.col(j);
  }   
  return retval;
}

inline VectorXd MappedSparse3DArray::get_col(int i, int j) const {
  VectorXd retval;
  retval = x[j].col(i);
  return retval;
}

// same as Sparse3DArray::spdmm_square_first but for the MappedSparse3DArray
inline MatrixXd MappedSparse3DArray::spdmm_square_first(const MatrixXd& Y) const {
  MatrixXd retval = MatrixXd::Zero(m,p);
  for (int j=0; j<p; j++) {
    retval.col(j) = (x[j].cwiseAbs2())*Y.col(j);
  }   
  return retval;
}



/*
  mult(X,Y,Z); overloaded functions for variants of matrix multiplication

  mySparseMatrix * MatrixXd ==> MatrixXd
  MatrixXd * mySparseMatrix ==> MatrixXd
  mySparseMatrix * mySparseMatrix ==> MatrixXd
  MappedSparse3DArray * MatrixXd ==> MatrixXd
  MappedSparse3DArray * mySparseMatrix ==> MatrixXd
  MappedSparse3DArray.^2 * mySparseMatrix ==> MatrixXd
  mySparseMatrix.^2 * mySparseMatrix ==> MatrixXd
*/   
void inline mult(const mySparseMatrix& X, const MatrixXd& Y, MatrixXd& Z) {
  Z = X*Y;
}

void inline mult(const MatrixXd& X, const mySparseMatrix& Y, MatrixXd& Z) {
  Z = X*Y;
}

void inline mult(const mySparseMatrix& X, const mySparseMatrix& Y, MatrixXd& Z) {
  mySparseMatrix Z_sparse = X*Y;
  Z = Z_sparse; // convert sparse to dense
}

void inline mult(const MappedSparse3DArray& X, const MatrixXd& Y, MatrixXd& Z) {
  //cout << "m: " << X.m << ", n: " << Y.rows() << ", p: " << X.p << endl;
  /*Z = MatrixXd::Zero(m,p);
  for (int p_idx=0; p_idx<p; p_idx++) {
    const long* jc = X.x[p_idx].outerIndexPtr();
    const long* ir = X.x[p_idx].innerIndexPtr();
    const double* vals = X.x[p_idx].valuePtr();
    for (int n_idx=0; n_idx<n; n_idx++) {
      for (int i = jc[n_idx]; i<jc[n_idx+1]; i++) {
        int m_idx = ir[i];
        Z(m_idx,p_idx) += vals[i]*Y(n_idx,p_idx);
      }
    }
  }*/
  Z = MatrixXd::Zero(X.m,X.p);
  for (int j=0; j<X.p; j++) {
    Z.col(j) = X.x[j]*Y.col(j);
  }
} 

void inline mult(const MappedSparse3DArray& X, const mySparseMatrix& Y, MatrixXd& Z) {
  int m = X.m;
  int p = X.p;
  //cout << "m: " << X.m << ", n: " << Y.rows() << ", p: " << X.p << endl;
  Z = MatrixXd::Zero(m,p);
  const long* Y_jc = Y.outerIndexPtr();
  const long* Y_ir = Y.innerIndexPtr();
  const double* Y_vals = Y.valuePtr();
  for (int p_idx=0; p_idx<p; p_idx++) {
    const long* X_jc = X.x[p_idx].outerIndexPtr();
    const long* X_ir = X.x[p_idx].innerIndexPtr();
    const double* X_vals = X.x[p_idx].valuePtr();
    for (int i=Y_jc[p_idx]; i<Y_jc[p_idx+1]; i++) {
      int n_idx = Y_ir[i];
      double Y_val = Y_vals[i];
      for (int j=X_jc[n_idx]; j<X_jc[n_idx+1]; j++) {
        int m_idx = X_ir[j];
        double X_val = X_vals[j];
        Z(m_idx,p_idx) += Y_val*X_val;
      }
    }
  }
}

void inline mult_square_first_arg(const MappedSparse3DArray& X, const mySparseMatrix& Y, MatrixXd& Z) {
  int m = X.m;
  int p = X.p;
  //cout << "m: " << X.m << ", n: " << Y.rows() << ", p: " << X.p << endl;
  Z = MatrixXd::Zero(m,p);
  const long* Y_jc = Y.outerIndexPtr();
  const long* Y_ir = Y.innerIndexPtr();
  const double* Y_vals = Y.valuePtr();
  for (int p_idx=0; p_idx<p; p_idx++) {
    const long* X_jc = X.x[p_idx].outerIndexPtr();
    const long* X_ir = X.x[p_idx].innerIndexPtr();
    const double* X_vals = X.x[p_idx].valuePtr();
    for (int i=Y_jc[p_idx]; i<Y_jc[p_idx+1]; i++) {
      int n_idx = Y_ir[i];
      double Y_val = Y_vals[i];
      for (int j=X_jc[n_idx]; j<X_jc[n_idx+1]; j++) {
        int m_idx = X_ir[j];
        double X_val = X_vals[j]*X_vals[j]; // square first arg value
        Z(m_idx,p_idx) += Y_val*X_val;
      }
    }
  }
}

void inline mult_square_first_arg(const mySparseMatrix& X, const mySparseMatrix& Y, MatrixXd& Z) {
  mySparseMatrix Z_sparse = X.cwiseAbs2()*Y;
  Z = Z_sparse; // sparse to dense
  //Z = (X.cwiseAbs2())*Y;
}

#endif // MATLAB_EIGEN_H_

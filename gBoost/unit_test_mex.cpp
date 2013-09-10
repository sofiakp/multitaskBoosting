#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "mex.h"
#include "matrix.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "matlab_eigen.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace Eigen;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  
  MappedSparseMatrix<double,ColMajor,long> foo = spm_matlab2eigen_mapped(prhs[0]);
  long *outer = foo.outerIndexPtr();
  long *inner = foo.innerIndexPtr();

    for(int i = 0; i < foo.cols(); ++i){
    cout << "Non-zero indices of column " << i << endl;
    for(int j = outer[i]; j < outer[i+1]; ++j){
      cout << inner[j] << endl;
    }
  }
  
  //MappedSparseMatrix<double,ColMajor,long> fcol = (foo.innerVector(0)).cwiseProduct(foo.innerVector(1)).eval();
  //cout << foo.innerVector(0) << endl;

}
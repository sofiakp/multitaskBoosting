#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "mex.h"
#include "matrix.h"

#include "matlab_eigen.h"


const myMappedSparseMatrix spm_matlab2eigen_mapped(const mxArray* spm_matlab) {
  int rows = mxGetM(spm_matlab);
  int cols = mxGetN(spm_matlab);
  mwIndex* ir = mxGetIr(spm_matlab);
  mwIndex* jc = mxGetJc(spm_matlab);
  double* data = mxGetPr(spm_matlab);
  int nnz = jc[cols];
  cout << nnz << endl;
  const myMappedSparseMatrix eigen_spm(rows,cols,nnz,(long*)jc,(long*)ir,data);
  return eigen_spm;
}

mySparseMatrix spm_matlab2eigen(const mxArray* spm_matlab) {
  int rows = mxGetM(spm_matlab);
  int cols = mxGetN(spm_matlab);
  mwIndex* ir = mxGetIr(spm_matlab);
  mwIndex* jc = mxGetJc(spm_matlab);
  double* data = mxGetPr(spm_matlab);
  int nnz = jc[cols];
  mySparseMatrix eigen_spm(rows,cols);
  std::vector<Triplet<double> > tripletList;
  tripletList.reserve(nnz); 
  int cur = 0;
  for (int col=0; col<cols; col++) {
    for (int k=jc[col]; k<jc[col+1]; k++) {
      int row = ir[k];
      double val = data[k];
      tripletList.push_back(Triplet<double>(row,col,val));
      cur += 1;
    }
  } 
  eigen_spm.setFromTriplets(tripletList.begin(),tripletList.end());
  return eigen_spm;
}

MatrixXd full_matlab2eigen(const mxArray* full_matlab) {
  int rows = mxGetM(full_matlab);
  int cols = mxGetN(full_matlab);
  double* pr = mxGetPr(full_matlab);
  MatrixXd full_eigen(rows,cols);
  std::copy(pr,pr+rows*cols,full_eigen.data());
  return full_eigen;
}

Map<MatrixXd> full_matlab2eigen_mapped(const mxArray* full_matlab) {
  int rows = mxGetM(full_matlab);
  int cols = mxGetN(full_matlab);
  double* pr = mxGetPr(full_matlab);
  Map<MatrixXd> full_eigen(pr,rows,cols);
  return full_eigen;
}

mxArray* eigen2matlab_full(const MatrixXd& full_eigen) {
  int rows = full_eigen.rows();
  int cols = full_eigen.cols();
  mxArray* full_matlab = mxCreateDoubleMatrix(rows,cols,mxREAL);
  double* matlab_pr = mxGetPr(full_matlab);
  std::copy(full_eigen.data(),full_eigen.data()+rows*cols,matlab_pr);
  return full_matlab;
}



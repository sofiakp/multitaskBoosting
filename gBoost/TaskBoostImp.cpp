#include <algorithm>
#include "mex.h"
#include "matrix.h"
#include <Eigen/Dense>
#include "Booster.h"
#include "matlab_utils.h"
#include "matlab_eigen.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace Eigen;
using namespace GBoost;

void mexFunctionImp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  
  if (nrhs != 4 || nlhs > 1) {
    mexErrMsgTxt("Usage: pred = task_boost_ex_imp(I, X, featIdx, filename)"); 
  }
  const mxArray* I_matlab = prhs[0];
  const mxArray* X_matlab = prhs[1];
  const mxArray* F_matlab = prhs[2];
  const mxArray* filename_matlab = prhs[3];

  int irows = mxGetM(I_matlab);
  int icols = mxGetN(I_matlab);
  int xrows = mxGetM(X_matlab);
  int xcols = mxGetN(X_matlab);
  int frows = mxGetM(F_matlab);
  int fcols = mxGetN(F_matlab);

  if (xrows == 0) {
    mexErrMsgTxt("X contains no examples");
  }
  if (xcols == 0) {
    mexErrMsgTxt("X has no features");
  }
  if (irows != xrows) {
    mexErrMsgTxt("X and I must have the same number of rows");
  }
  if (icols == 0) {
    mexErrMsgTxt("Number of tasks is 0");
  }
  if(!mxIsSparse(I_matlab)){
    mexErrMsgTxt("I must be sparse");
  }
  if(frows > 1 && fcols > 1){
    mexErrMsgTxt("featIdx must be a vector");
  }
  if(!mxIsChar(filename_matlab)){
    mexErrMsgTxt("filename must be a string");
  }
  
  const MappedSparseMatrix<double,ColMajor,long> I = spm_matlab2eigen_mapped(I_matlab);
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  vector<unsigned> featIdx(mxGetPr(F_matlab), mxGetPr(F_matlab) + max(frows, fcols));
  char* filename = mxArrayToString(filename_matlab);

  TaskTreeBooster<  MappedSparseMatrix<double,ColMajor,long>, Map<MatrixXd>, Map<VectorXd> > booster;
  booster.load(filename);
  plhs[0] = mxCreateDoubleMatrix(X.rows(), featIdx.size(), mxREAL); // This should be nfeat-by-ntasks
  Map<MatrixXd> imp(mxGetPr(plhs[0]), X.rows(), featIdx.size());
  booster.predict(I, X, featIdx, imp);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  mexFunctionImp(nlhs, plhs, nrhs, prhs);
}

#include <algorithm>
#include "mex.h"
#include "matrix.h"
#include <Eigen/Dense>
#include "Booster.h"
#include "matlab_utils.h"
#include "matlab_eigen.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace Eigen;
using namespace GBoost;

void check_train_arguments(const mxArray* prhs[]);

void mexFunctionTrain(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  
  if (nrhs != 9 || nlhs != 1) {
    mexErrMsgTxt("Usage:;"); 
  }
  check_train_arguments(prhs);
  
  const MappedSparseMatrix<double,ColMajor,long> I = spm_matlab2eigen_mapped(prhs[0]);
  const mxArray* X_matlab = prhs[1];
  const mxArray* R_matlab = prhs[2];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  Map<VectorXd> R(mxGetPr(R_matlab), mxGetM(R_matlab), mxGetN(R_matlab));
  
  unsigned int niter = (unsigned int)getDoubleScalar(prhs[3]);
  unsigned int maxDepth = (unsigned int)getDoubleScalar(prhs[4]);
  unsigned int minNodes = (unsigned int)getDoubleScalar(prhs[5]);
  double minErr = getDoubleScalar(prhs[6]);
  double fracFeat = getDoubleScalar(prhs[7]);
  double shrink = getDoubleScalar(prhs[8]);
  
  plhs[0] = mxCreateDoubleMatrix(niter, 1, mxREAL);
  double* err = mxGetPr(plhs[0]);
    
  //mxArray* err_matlab = plhs[0];
  //Map<VectorXd> err(mxGetPr(err_matlab), , mxGetN(err_matlab));
  
  VectorXd err_eigen(niter);
  TaskTreeBooster<  MappedSparseMatrix<double,ColMajor,long>, Map<MatrixXd>, Map<VectorXd> > booster;
  booster.learn(I, X, R, niter, maxDepth, minNodes, minErr, fracFeat, shrink, err_eigen);
  
  for(unsigned i = 0; i < err_eigen.size(); ++i) err[i] = err_eigen(i);
}

void check_train_arguments(const mxArray* prhs[]){
  const mxArray* I_matlab = prhs[0];
  const mxArray* X_matlab = prhs[1];
  const mxArray* R_matlab = prhs[2];
   
  int irows = mxGetM(I_matlab);
  int icols = mxGetN(I_matlab);
  int xrows = mxGetM(X_matlab);
  int xcols = mxGetN(X_matlab);
  int rrows = mxGetM(R_matlab);
  int rcols = mxGetN(R_matlab);
  
  if (rcols > 1) {
    mexErrMsgTxt("The response R must be a column vector");
  }
  if (xrows != rrows) {
    mexErrMsgTxt("R and X must have the same number of rows");
  }
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
  
  const mxArray* niter_matlab = prhs[3];
  const mxArray* maxDepth_matlab = prhs[4];
  const mxArray* minNodes_matlab = prhs[5];
  const mxArray* minErr_matlab = prhs[6];
  const mxArray* fracFeat_matlab = prhs[7];
  const mxArray* shrink_matlab = prhs[8];
  
  if (!isIntegerScalar(niter_matlab) || getDoubleScalar(niter_matlab) <= 0){
    mexErrMsgTxt("niter must be a positive integer");
  }
  if (!isIntegerScalar(maxDepth_matlab) || getDoubleScalar(maxDepth_matlab) < 0){
    mexErrMsgTxt("maxDepth must be a non-negative integer");
  }
  if (!isIntegerScalar(minNodes_matlab) || getDoubleScalar(minNodes_matlab) < 0){
    mexErrMsgTxt("minNodes must be a non-negative integer");
  }
  if (!isDoubleScalar(minErr_matlab) || getDoubleScalar(minErr_matlab) < 0){
    mexErrMsgTxt("minErr must be a non-negative real");
  }
  if (!isIntegerScalar(fracFeat_matlab) || getDoubleScalar(fracFeat_matlab) < 0 || getDoubleScalar(fracFeat_matlab) > 1){
    mexErrMsgTxt("fracFeat must be a real in [0 1]");
  }
  if (!isDoubleScalar(shrink_matlab) || getDoubleScalar(shrink_matlab) <= 0){
    mexErrMsgTxt("shrink must be a positive real");
  }
}

void mexFunctionTest(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  /*RegressionTree< Map<MatrixXd>, Map<VectorXd> > tree;
  tree.loadFromMatlab(prhs[0]);
  
  const mxArray* X_matlab = prhs[1];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  
  const mxArray* I_matlab = prhs[2];
  double* tmp_idxs = mxGetPr(I_matlab);
  vector<unsigned> idxs(mxGetM(I_matlab));
  for(unsigned i = 0; i < idxs.size(); ++i) idxs[i] = tmp_idxs[i];
  
  plhs[0] = mxCreateDoubleMatrix(idxs.size(),1,mxREAL);
  Map<VectorXd> pred(mxGetPr(plhs[0]), idxs.size(), 1);
  tree.predict(X, idxs, pred);*/
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
#ifdef TRAIN
  mexFunctionTrain(nlhs, plhs, nrhs, prhs);
#else
  mexFunctionTest(nlhs, plhs, nrhs, prhs);
#endif
}
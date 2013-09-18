#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "mex.h"
#include "matrix.h"
#include <Eigen/Dense>
#include "RegressionTree.h"
#include "matlab_utils.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace Eigen;
using namespace GBoost;

void check_train_arguments(const mxArray* prhs[]);

void mexFunctionTrain(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  
  if (nrhs != 8 || nlhs != 1) {
    mexErrMsgTxt("Usage:;"); 
  }
  check_train_arguments(prhs);
  
  const mxArray* X_matlab = prhs[0];
  const mxArray* R_matlab = prhs[1];
  const mxArray* W_matlab = prhs[2];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  Map<VectorXd> R(mxGetPr(R_matlab), mxGetM(R_matlab), mxGetN(R_matlab));
  Map<VectorXd> W(mxGetPr(W_matlab), mxGetM(W_matlab), mxGetN(W_matlab));
  
  unsigned int maxDepth = (unsigned int)getDoubleScalar(prhs[4]);
  unsigned int minNodes = (unsigned int)getDoubleScalar(prhs[5]);
  double minErr = getDoubleScalar(prhs[6]);
  double fracFeat = getDoubleScalar(prhs[7]);
  
  const mxArray* I_matlab = prhs[3];
  double* tmp_idxs = mxGetPr(I_matlab);
  //Map<VectorXi> idxs_tmp((int*)mxGetData(I_matlab), mxGetM(I_matlab), mxGetN(I_matlab));
  vector<unsigned> idxs(mxGetM(I_matlab));
  for(unsigned i = 0; i < idxs.size(); ++i) idxs[i] = tmp_idxs[i];
  
  RegressionTree< Map<MatrixXd>, Map<VectorXd> > tree;
  tree.learn(X, R, W, maxDepth, minNodes, minErr, idxs, fracFeat);
  tree.printInfo();

  plhs[0] = tree.saveToMatlab();

  // vector<double> imp(X.cols(), 0.0);
  // tree.varImportance(imp);
  // for(unsigned i = 0; i < imp.size(); ++i) cout << "Feat" << i << " imp " << imp[i] << endl;
}

void check_train_arguments(const mxArray* prhs[]){
  const mxArray* X_matlab = prhs[0];
  const mxArray* R_matlab = prhs[1];
  const mxArray* W_matlab = prhs[2];
  const mxArray* I_matlab = prhs[3];
   
  int xrows = mxGetM(X_matlab);
  int xcols = mxGetN(X_matlab);
  int rrows = mxGetM(R_matlab);
  int rcols = mxGetN(R_matlab);
  int wrows = mxGetM(W_matlab);
  int wcols = mxGetN(W_matlab);
  int irows = mxGetM(I_matlab);
  int icols = mxGetN(I_matlab);
  
  if (rcols > 1) {
    mexErrMsgTxt("The response R must be a column vector");
  }
  if (wcols > 1) {
    mexErrMsgTxt("The weights W must be a column vector");
  }
  if (rrows != wrows) {
    mexErrMsgTxt("R and W must have the same number of rows");
  }
  if (xrows == 0) {
    mexErrMsgTxt("X contains no examples");
  }
  if (xcols == 0) {
    mexErrMsgTxt("X has no features");
  }
  if (icols > 1) {
    mexErrMsgTxt("The indices must be a column vector");
  }
  if (irows == 0) {
    mexErrMsgTxt("No examples selected");
  }
  
  double* tmp_idxs = mxGetPr(I_matlab);
  for(int i = 0; i < irows; ++i){
    if(tmp_idxs[i] > xrows - 1 || tmp_idxs[i] < 0 || ceil(tmp_idxs[i]) != floor(tmp_idxs[i]))
      mexErrMsgTxt("The indices must be integers in [0 Xrows - 1]");
  }
  
  const mxArray* maxDepth_matlab = prhs[4];
  const mxArray* minNodes_matlab = prhs[5];
  const mxArray* minErr_matlab = prhs[6];
  const mxArray* fracFeat_matlab = prhs[7];
  
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
}

void mexFunctionTest(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  RegressionTree< Map<MatrixXd>, Map<VectorXd> > tree;
  tree.loadFromMatlab(prhs[0]);
  
  const mxArray* X_matlab = prhs[1];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  
  const mxArray* I_matlab = prhs[2];
  double* tmp_idxs = mxGetPr(I_matlab);
  vector<unsigned> idxs(mxGetM(I_matlab));
  for(unsigned i = 0; i < idxs.size(); ++i) idxs[i] = tmp_idxs[i];
  
  plhs[0] = mxCreateDoubleMatrix(idxs.size(),1,mxREAL);
  Map<VectorXd> pred(mxGetPr(plhs[0]), idxs.size(), 1);
  tree.predict(X, idxs, pred);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
#ifdef TRAIN
  mexFunctionTrain(nlhs, plhs, nrhs, prhs);
#else
  mexFunctionTest(nlhs, plhs, nrhs, prhs);
#endif
}

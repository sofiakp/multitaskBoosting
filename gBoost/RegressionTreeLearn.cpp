#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "mex.h"
#include "matrix.h"
#include <Eigen/Dense>
#include "RegressionTree.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace Eigen;
using namespace GBoost;

//void check_arguments(int nlhs, int nrhs, const mxArray* prhs[]);

void mexFunctionTrain(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  
  //check_arguments(nlhs,nrhs,prhs);
  
  cout << "Point 0" << endl;
  const mxArray* X_matlab = prhs[0];
  const mxArray* R_matlab = prhs[1];
  const mxArray* W_matlab = prhs[2];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  Map<VectorXd> R(mxGetPr(R_matlab), mxGetM(R_matlab), mxGetN(R_matlab));
  Map<VectorXd> W(mxGetPr(W_matlab), mxGetM(W_matlab), mxGetN(W_matlab));
  
  unsigned int maxDepth = (unsigned int)mxGetPr(prhs[3])[0];
  unsigned int minNodes = (unsigned int)mxGetPr(prhs[4])[0];
  double minErr = mxGetPr(prhs[5])[0];
  double fracFeat = mxGetPr(prhs[6])[0];
  
  cout << "Point 1" << endl;
  vector<unsigned int> idxs(X.rows());
  for(unsigned i = 0; i < X.rows(); ++i) idxs[i] = i;
  cout << "Point 2" << endl;
  
  RegressionTree< Map<MatrixXd>, Map<VectorXd> > tree;
  tree.learn(X, R, W, maxDepth, minNodes, minErr, idxs, fracFeat);
  tree.printInfo();
  
  plhs[0] = tree.saveToMatlab();
}

void mexFunctionTest(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

  RegressionTree< Map<MatrixXd>, Map<VectorXd> > tree;
  tree.loadFromMatlab(prhs[0]);
  
  const mxArray* X_matlab = prhs[1];
  Map<MatrixXd> X(mxGetPr(X_matlab), mxGetM(X_matlab), mxGetN(X_matlab));
  vector<unsigned int> idxs(X.rows());
  for(unsigned i = 0; i < X.rows(); ++i) idxs[i] = i;
  VectorXd pred(X.rows(), 1);
  tree.predict(X, idxs, pred);
  
  plhs[0] = mxCreateDoubleMatrix(pred.size(),1,mxREAL);
  double *predMat = mxGetPr(plhs[0]);
  for(unsigned i = 0; i < pred.size(); ++i) predMat[i] = pred(i);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
#ifdef TRAIN
  mexFunctionTrain(nlhs, plhs, nrhs, prhs);
#else
  mexFunctionTest(nlhs, plhs, nrhs, prhs);
#endif
}
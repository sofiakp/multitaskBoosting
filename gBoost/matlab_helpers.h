#ifndef MATLAB_HELPERS_H
#define MATLAB_HELPERS_H

#include "mex.h"
#include "matrix.h"


template<typename T>
mxClassID matlabClassID() { mexErrMsgTxt("Class ID not defined"); return mxUNKNOWN_CLASS;}

template<>
mxClassID matlabClassID<int>() { return mxINT32_CLASS; }

template<>
mxClassID matlabClassID<unsigned int>() { return mxUINT32_CLASS; }

template<>
mxClassID matlabClassID<unsigned char>() { return mxUINT8_CLASS; }

template<>
mxClassID matlabClassID<char>() { return mxCHAR_CLASS; }


template<>
mxClassID matlabClassID<double>() { return mxDOUBLE_CLASS; }

template<>
mxClassID matlabClassID<float>() { return mxSINGLE_CLASS; }

bool isDoubleFull(const mxArray* arr) {
  return mxIsDouble(arr) && !mxIsSparse(arr) && mxGetM(arr) != 0 && mxGetN(arr) != 0;
}

bool isDoubleSparse(const mxArray* arr) {
  return mxIsDouble(arr) && mxIsSparse(arr) && mxGetM(arr) != 0 && mxGetN(arr) != 0;
}

bool isDoubleFullColVec(const mxArray* arr) {
  return isDoubleFull(arr) && mxGetN(arr) == 1 && mxGetM(arr) != 0;
}

bool isDoubleScalar(const mxArray* arr) {
  return ( mxIsDouble(arr) && mxGetM(arr) == 1 && mxGetN(arr) == 1);
}

double getDoubleScalar(const mxArray* arr) {
  return mxGetPr(arr)[0];
}

bool getLogicalScalar(const mxArray* arr) {
  return (bool)(((mxLogical*)mxGetData(arr))[0]);
}


mxArray* full2sparse(const mxArray* full) {
  mwSize nrows = mxGetM(full);
  mwSize ncols = mxGetN(full);
  double* fulldata = mxGetPr(full);
  mwIndex nnz = 0;
  for (mwIndex col=0; col<ncols; col++) {
    for (mwIndex row=0; row<nrows; row++) {
      if (fulldata[row + nrows*col] != 0.0) {
        nnz += 1;
      }
    }
  }
  mxArray* sparse = mxCreateSparse(nrows,ncols,nnz,mxREAL);
  mwIndex* sparse_jc = mxGetJc(sparse);
  mwIndex* sparse_ir = mxGetIr(sparse);
  double* sparse_data = mxGetPr(sparse);
  mwIndex cur = 0;
  for (mwIndex col=0; col<ncols; col++) {
    sparse_jc[col] = cur;
    for (mwIndex row=0; row<nrows; row++) {
      if (fulldata[row + nrows*col] != 0.0) {
        sparse_ir[cur] = row;
        sparse_data[cur] = fulldata[row + nrows*col];
        cur += 1;
      }
    }
  }
  sparse_jc[ncols] = nnz;

  return sparse;
}

#endif // MATLAB_HELPERS_H_

#ifndef MATLAB_UTILS_H
#define MATLAB_UTILS_H

#include "mex.h"
#include "matrix.h"
#include <math.h>

template<typename T>
mxClassID matlabClassID() { mexErrMsgTxt("Class ID not defined"); return mxUNKNOWN_CLASS; }

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

bool isColVec(const mxArray* arr) {
  return mxGetN(arr) == 1 && mxGetM(arr) != 0;
}

bool isDoubleScalar(const mxArray* arr) {
  return ( mxIsDouble(arr) && mxGetM(arr) == 1 && mxGetN(arr) == 1);
}

bool isIntegerScalar(const mxArray* arr) {
  return isDoubleScalar(arr) && ceil(mxGetPr(arr)[0]) == floor(mxGetPr(arr)[0]);
}

double getDoubleScalar(const mxArray* arr) {
  return mxGetPr(arr)[0];
}

bool getLogicalScalar(const mxArray* arr) {
  return (bool)(((mxLogical*)mxGetData(arr))[0]);
}


#endif // MATLAB_UTILS_H

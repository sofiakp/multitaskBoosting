#!/bin/bash

NUM_SEARCH_THREADS=4
NUM_BOOST_THREADS=4

# *** SET THE LOCATION OF YOUR MATLAB BINARY DIRECTORY ***
## do which matlab, then ls -l whatever-you-get                                                                                                                                      
MATLABDIR=/usr/local/MATLAB/R2011b
#/afs/cs/package/matlab-r2011b/matlab/r2011b

#EIGEN_INCLUDE_DIR=/opt/eigen-eigen-43d9075b23ef/
EIGEN_INCLUDE_DIR=/home/sofiakp/projects/Anshul/matlab/medusa/src/processing4/eigen/

BOOST_INCLUDE_DIR=/home/sofiakp/cpplibs/boost_1_54_0
BOOST_SERIAL_DIR=/home/sofiakp/cpplibs/boost_1_54_0/stage/lib
BOOST_LINK_FLAGS=-L$(BOOST_SERIAL_DIR) -lboost_serialization

# MIGIHT HAVE TO SELECT A PARTICULAR g++ here, depending on your setup
CXX=g++
#g++44 for scail

# Add -O3 if not debugging
CXXFLAGS= -DQUIET -Wall -fPIC -I$(MATLABDIR)/extern/include -fopenmp -DNUM_SEARCH_THREADS=$(NUM_SEARCH_THREADS) -DNUM_BOOST_THREADS=$(NUM_BOOST_THREADS)

LDFLAGS= -fopenmp

MEX=$(MATLABDIR)/bin/mex

# -g for debugging
MEX_OPTION= -g -largeArrayDims CXX=$(CXX) LD=$(CXX) CXXFLAGS\#"\$$CXXFLAGS $(CXXFLAGS)" LDFLAGS\#"\$$LDFLAGS $(LDFLAGS)"

all: matlab_eigen unit_test_mex RegressionTreeLearn RegressionTreePredict TaskBoostLearn

matlab_eigen: matlab_eigen.cpp matlab_eigen.h
	$(MEX) $(MEX_OPTION) -I$(EIGEN_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(BOOST_LINK_FLAGS) -c matlab_eigen.cpp

unit_test_mex: unit_test_mex.cpp matlab_eigen.h matlab_eigen.cpp
	$(MEX) $(MEX_OPTION) -I$(EIGEN_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(BOOST_LINK_FLAGS) unit_test_mex.cpp matlab_eigen.o -o unit_test_mex

RegressionTreeLearn: RegressionTreeLearn.cpp RegressionTree.h matlab_utils.h
	$(MEX) $(MEX_OPTION) -I$(EIGEN_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(BOOST_LINK_FLAGS) -DTRAIN RegressionTreeLearn.cpp -o regression_tree_learn

RegressionTreePredict: RegressionTreeLearn.cpp RegressionTree.h matlab_utils.h
	$(MEX) $(MEX_OPTION) -I$(EIGEN_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(BOOST_LINK_FLAGS) RegressionTreeLearn.cpp -o regression_tree_predict

TaskBoostLearn: TaskBoostLearn.cpp Booster.h matlab_utils.h matlab_eigen.h matlab_eigen.cpp
	$(MEX) $(MEX_OPTION) -I$(EIGEN_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(BOOST_LINK_FLAGS) -DTRAIN TaskBoostLearn.cpp matlab_eigen.o -o task_boost_learn

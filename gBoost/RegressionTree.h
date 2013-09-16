#ifndef _GBOOST_REGTREE_H_
#define _GBOOST_REGTREE_H_

#include <vector>
#include <stack>
#include <numeric>
#include <math.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <limits>
#include "matlab_utils.h"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;
using namespace Eigen;

namespace GBoost {
  template<typename ValueType, typename IndexType = unsigned int > struct RegressionTreeNode{
    bool isLeaf; // TODO: remove this feature. leftNodeIdx == -1 means leaf
    ValueType value; // Predicted value for leaves, split threshold for interior nodes
    ValueType err; // Sum of squares (training) error for the examples of this node.

    int leftNodeIdx;
    int rightNodeIdx;
    
    IndexType featureIdx; // Index of the feature that defines the split, 0 for leaves.
    
    RegressionTreeNode() : isLeaf(true), value(0), err(0), leftNodeIdx(-1), rightNodeIdx(-1), featureIdx(0){}

    template<typename Archive>
    void serialize(Archive & ar, const unsigned int version){
      ar & isLeaf;
      ar & value;
      ar & err;
      ar & leftNodeIdx;
      ar & rightNodeIdx;
      ar & featureIdx;
    }
  };
  
  struct RegressionStackEntry {
    unsigned nodeIdx; // Index of node in the tree
    unsigned depth; // Useful to know when to stop growing the tree
    vector<unsigned> idxList; // List of example indices
    //VectorXi idxList;
    
    RegressionStackEntry(){}
    //RegressionStackEntry(unsigned listSize) : idxList(listSize) {}
    RegressionStackEntry(unsigned listSize) {idxList.reserve(listSize);}
  };
  
  // This a stump with a single split on a given variable
  template<typename ValueType> struct RegStumpInfo {
    ValueType threshold; // Threshold for the variable of the stump
    ValueType err1, err2; // Error at left and right child.
    ValueType y1, y2; // Values at left and right child respectively.
    unsigned int n1, n2; // Number of examples in left and right child.
    
    bool isPure;
    
    RegStumpInfo() {}
    
    void setValues(const bool &inIsPure, const ValueType &inThreshold, const ValueType &inErr1, const ValueType &inErr2, 
            const ValueType &inY1, const ValueType &inY2, const unsigned int &inN1, const unsigned int &inN2){
      isPure = inIsPure;
      threshold = inThreshold;
      err1 = inErr1;
      err2 = inErr2;
      y1 = inY1;
      y2 = inY2;
      n1 = inN1;
      n2 = inN2;
    }
    
    void printInfo() const {
      cout << "Pure: " << isPure << endl;
      cout << "Threshold: " << threshold << endl;
      cout << "Left value: " << y1 << " (n=" << n1 << "), right value: " << y2 << " (n=" << n2 << ")" << endl;
      cout << "Err: " << err1 << " " << err2 << endl;
    }
  };
  
  // used to sort indices
  template<typename MatType> class SortVectorByValue {
  private:
    const MatType &mat;
    
  public:
    SortVectorByValue(const MatType &inMat) : mat(inMat) {}
    
    inline bool operator()(unsigned l, unsigned r) const {
      return mat(l) < mat(r);
    }
  };
    
  /*
   * FeatureMatType is a type of Eigen::Matrix. This is the type of the input feature matrix. 
   * ResponseMatType is the type of Eigen::Vector. This is the type of the responses (and of the weights).
   */
  template< typename FeatureDerived, typename ResponseDerived > class RegressionTree {
    typedef typename FeatureDerived::Scalar ValueType; // Type of the features (eg. double)
    typedef typename FeatureDerived::Index IndexType; // Type of indices in the features (eg. unsigned)
    typedef typename ResponseDerived::Scalar ResponseValueType; // Type of the responses 
    
    typedef RegressionTreeNode<ValueType, IndexType> NodeType;
    typedef Matrix<ValueType, Dynamic, 1> ValueVectorType; // Type of a column of the feature matrix. typename internal::plain_col_type<FeatureDerived>::type
    
  private:
    vector<NodeType> nodes;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
      ar & nodes;
      // for(unsigned i = 0; i < nodes.size(); ++i) ar & nodes[i];
    }
    
  public:
    const vector<NodeType>& getNodes() const { return nodes; }
    
    unsigned size() const { return nodes.size(); }
    
    RegressionTree<FeatureDerived, ResponseDerived>(){}

    mxArray *saveToMatlab() const {
      const char *fieldNames[] = { "isLeaf", "value", "err", "leftNodeIdx", "rightNodeIdx", "featIdx"};
      
      mwSize dims[1]; dims[0] = nodes.size();
      
      // create array
      mxArray *ret = mxCreateStructArray(1, dims, 6, fieldNames);
      for (unsigned i=0; i < nodes.size(); ++i) {
        mxArray *isLeaf = mxCreateNumericMatrix(1, 1, mxUINT8_CLASS, mxREAL);
        ((unsigned char *)mxGetData(isLeaf))[0] = nodes[i].isLeaf?1:0;
        mxSetField( ret, i, "isLeaf", isLeaf );
        
        mxArray *value = mxCreateNumericMatrix(1,1, matlabClassID<ValueType>(), mxREAL);
        ((ValueType *)mxGetData(value))[0] = nodes[i].value;
        mxSetField( ret, i, "value", value );

        mxArray *err = mxCreateNumericMatrix(1,1, matlabClassID<ValueType>(), mxREAL);
        ((ValueType *)mxGetData(value))[0] = nodes[i].err;
        mxSetField( ret, i, "err", err);
        
        mxArray *leftNodeIdx = mxCreateNumericMatrix(1,1, matlabClassID<int>(), mxREAL);
        ((int *)mxGetData(leftNodeIdx))[0] = nodes[i].leftNodeIdx;
        mxSetField( ret, i, "leftNodeIdx", leftNodeIdx );
        
        mxArray *rightNodeIdx = mxCreateNumericMatrix(1,1, matlabClassID<int>(), mxREAL);
        ((int *)mxGetData(rightNodeIdx))[0] = nodes[i].rightNodeIdx;
        mxSetField( ret, i, "rightNodeIdx", rightNodeIdx );
        
        mxArray *featIdx = mxCreateNumericMatrix(1,1, matlabClassID<unsigned int>(), mxREAL);
        ((unsigned int *)mxGetData(featIdx))[0] = (unsigned int)nodes[i].featureIdx;
        mxSetField( ret, i, "featIdx", featIdx );
      }
      return ret;
    }
    
    bool loadFromMatlab( const mxArray *data ) {
      const unsigned numNodes = mxGetNumberOfElements(data);
      nodes.clear();
      nodes.resize(numNodes);
      
      for (unsigned i = 0; i < numNodes; ++i) {
        mxArray *isLeaf = mxGetField(data, i, "isLeaf");
        if (mxGetClassID(isLeaf) != mxUINT8_CLASS) {
          printf("isLeaf type mismatch");
          return false;
        }
        nodes[i].isLeaf = ((unsigned char *)mxGetData(isLeaf))[0] != 0;
        
        mxArray *value = mxGetField(data, i, "value");
        if (mxGetClassID(value) != matlabClassID<ValueType>()) {
          printf("Value type mismatch");
          return false;
        }
        nodes[i].value = ((ValueType *)mxGetData(value))[0];

        mxArray *err = mxGetField(data, i, "err");
        if (mxGetClassID(err) != matlabClassID<ValueType>()) {
          printf("Error type mismatch");
          return false;
        }
        nodes[i].err = ((ValueType *)mxGetData(err))[0];
        
        mxArray *leftNodeIdx = mxGetField(data, i, "leftNodeIdx");
        if (mxGetClassID(leftNodeIdx) != matlabClassID<int>()) {
          printf("leftNodeIdx type mismatch");
          return false;
        }
        nodes[i].leftNodeIdx = ((int *)mxGetData(leftNodeIdx))[0];
        
        mxArray *rightNodeIdx = mxGetField(data, i, "rightNodeIdx");
        if (mxGetClassID(rightNodeIdx) != matlabClassID<int>()) {
          printf("rightNodeIdx type mismatch");
          return false;
        }
        nodes[i].rightNodeIdx = ((int *)mxGetData(rightNodeIdx))[0];
        
        mxArray *featIdx = mxGetField(data, i, "featIdx");
        if (mxGetClassID(featIdx) != matlabClassID<unsigned int>()) {
          printf("featureIdx type mismatch");
          return false;
        }
        nodes[i].featureIdx = ((unsigned int *)mxGetData(featIdx))[0];
      }
      return true;
    }
    
    void printInfo() const {
      for (unsigned i = 0; i < nodes.size(); ++i) {
        if ( nodes[i].isLeaf ) {
          cout << i << ": " << nodes[i].value << endl;
        } else {
          cout << i << ": Feat" << nodes[i].featureIdx;
          cout << " < " << nodes[i].value << " then " << nodes[i].leftNodeIdx << " else " << nodes[i].rightNodeIdx << endl;
        }
      }
    }
    
    /*
     * Learn a stump from the data given in X, the responses R, and the response weights W.
     * idxs: indices of examples (row of X) that will be used for training, the rest will be ingored. This is assumed to have length
     * at least 1.
     * fidx: index of feature (column of X) on which the stumps single node will be split.
     * The stump is written in stumpInfo.
     */
     ResponseValueType learnSingleStump(const MatrixBase<FeatureDerived> &X, 
					const MatrixBase<ResponseDerived> &R, const MatrixBase<ResponseDerived> &W,
					vector<unsigned> &idxs, IndexType fidx, RegStumpInfo<ResponseValueType> &stumpInfo) {
      
      const unsigned N = idxs.size(); // number of examples that will be used for training.
      
      ValueVectorType Xcol = X.col(fidx);
      
      // The node is pure if there aren't big enough differences between the values of the feature
      bool isPure = (Xcol.array() - Xcol(idxs[0])).cwiseAbs().maxCoeff() < 10 * std::numeric_limits<ValueType>::epsilon();
      
      ResponseValueType sumW1, sumW2, y1, y2, sumWR1, sumWR2, sumWRSq1, sumWRSq2;
      sumW1 = sumW2 = y1 = y2 = sumWR1 = sumWR2 = sumWRSq1 = sumWRSq2 = 0;
      
      for (unsigned i = 0; i < N; ++i) {
        const ResponseValueType w = W(idxs[i]);
        const ResponseValueType r = R(idxs[i]);
        
        y2 += r;
        sumW2 += w;
        sumWR2 += w * r;
        sumWRSq2 += w * r * r;
      }
      y2 /= N;
      
      ResponseValueType err1 = 0;
      ResponseValueType err2 = y2 * y2 * sumW2 + sumWRSq2 - 2*y2*sumWR2;

      stumpInfo.setValues(true, 0, err1, err2, y1, y2, 0, N);
      
      // The node is pure if the average error is very small or all the values for the feature are (almost) the same.
      if (isPure || err2 < 10 * N * std::numeric_limits<ValueType>::epsilon()) return err2;
      
      vector<unsigned> sortedIdx(idxs); // copy the input vector
      // Sort the indices in sortedIdx by the values of Xcol at these indices.
      sort(sortedIdx.begin(), sortedIdx.end(), SortVectorByValue< ValueVectorType >(Xcol));
      
      for (unsigned i = 0; i < N - 1; ++i) {
        const unsigned qidx = sortedIdx[i];
        const ResponseValueType w = W(qidx);
        const ResponseValueType r = R(qidx);
        
        sumW1 += w;
        sumWR1 += w * r;
        sumWRSq1 += w * r * r;
        sumW2 -= w;
        sumWR2 -= w * r;
        sumWRSq2 -= w * r * r;
        
        y1 = ((y1 * i) + r) / (i + 1);
        y2 = ((y2 * (N - i) - r)) / (N - i - 1);
        
        if (Xcol(qidx) == Xcol(sortedIdx[i + 1])) continue; // Can't set the threshold there.
        
        err1 = y1 * y1 * sumW1 + sumWRSq1 - 2*y1*sumWR1;
        err2 = y2 * y2 * sumW2 + sumWRSq2 - 2*y2*sumWR2;
        
        if (stumpInfo.err1 + stumpInfo.err2 >= err1 + err2) {
          stumpInfo.setValues(false, (Xcol(sortedIdx[i]) + Xcol(sortedIdx[i + 1])) / 2, err1, err2, y1, y2, i + 1, N - i - 1);
        } //else{
          // Assuming the distribution is trully bimodal, an increase in the error means we should stop.
          //break;
        //}
      }
      return stumpInfo.err1 + stumpInfo.err2;
    }
    
    /*
     * Learn a regression tree from the data given in X, the responses R, and the response weights W.
     * maxDepth: maximum depth of interior nodes (leaf nodes with values, i.e. no splits, will be one level below that, 
     * so if maxDepth == 2, then the tree will have at most 7 interior nodes and 8 leafs.)
     * minNodeSize: minimum number of examples in a node in order to allow splits
     * minChildErr: minimum average error in a node to allow splits
     * idxs: indices of examples (row of X) that will be used for training, the rest will be ingored. 
     * This is assumed to have length at least 1.
     * fracFeat: fraction of features to test at each split (1 will test all features).
     */
    void learn(const MatrixBase<FeatureDerived> &X, const MatrixBase<ResponseDerived> &R, const MatrixBase<ResponseDerived> &W,
            unsigned maxDepth, unsigned minNodeSize, ValueType minChildErr, vector<unsigned> &idxs, double fracFeat){
      
      nodes.clear();
      nodes.reserve(min(int(pow(2, maxDepth + 2) - 1), 100)); // maximum number of nodes will be 2^(maxDepth + 2) - 1.
      
      IndexType nfeat = (IndexType)min((int)X.cols(), (int)ceil(X.cols() * fracFeat)); // number of features to test at each node split
      
      // nodes-to-be will be added in a stack. We'll pop a node, split it, add it to "nodes" and add its children to the stack.
      stack< RegressionStackEntry > nodeStack; 
      
      RegressionStackEntry all;
      all.idxList = idxs;
      all.nodeIdx = 0;
      all.depth = 0;
      nodeStack.push(all);
      
      NodeType root;      
      // Compute the total weighted loss at the root
      ResponseValueType sumW2, y2, sumWR2, sumWRSq2;
      sumW2 = y2 = sumWR2 = sumWRSq2 = 0;
      for (unsigned i = 0; i < idxs.size(); ++i) {
        const ResponseValueType w = W(idxs[i]);
        const ResponseValueType r = R(idxs[i]);
        
        y2 += r;
        sumW2 += w;
        sumWR2 += w * r;
        sumWRSq2 += w * r * r;
      }
      y2 /= idxs.size();
      root.err = y2 * y2 * sumW2 + sumWRSq2 - 2*y2*sumWR2;
      nodes.push_back(root); 

      Matrix<IndexType, Dynamic, 1> selFeat(nfeat); // Will store the indices of features to test
      vector<IndexType> tmp(X.cols());
      for(unsigned i = 0; i < X.cols(); ++i) tmp[i] = i;

      while(!nodeStack.empty()) {
        RegressionStackEntry top = nodeStack.top();
        nodeStack.pop();
          
        if(nfeat == X.cols())
          selFeat.setLinSpaced(nfeat, 0, nfeat - 1); // selFeat will be 0,1,2,...,nfeat-1.
        else{
          random_shuffle(tmp.begin(), tmp.end());
          tmp.resize(nfeat);
          for(unsigned i = 0; i < nfeat; ++i) selFeat(i) = tmp[i];
        }
        
        vector< RegStumpInfo< ValueType > > stumpResults(nfeat); // Store the results for all possible splits of this node
        Matrix< ResponseValueType, Dynamic, 1 > stumpErrs(nfeat);
        
        //#pragma omp parallel for num_threads(NUM_SEARCH_THREADS)
        for (unsigned f = 0; f < nfeat; ++f) {
          stumpErrs(f) = learnSingleStump(X, R, W, top.idxList, selFeat(f), stumpResults[f]);
        }
        
        IndexType minCol;
        stumpErrs.minCoeff(&minCol);
        RegStumpInfo<ResponseValueType> minStump = stumpResults[minCol];
        minCol = selFeat(minCol);
        // minStump.printInfo();	
     
        if (minStump.isPure) {
          nodes[top.nodeIdx].isLeaf = true;
          nodes[top.nodeIdx].value = minStump.y2;
          // No need to add children in the stack or the tree.
        }else{
          NodeType leftNode, rightNode;
	  leftNode.err = minStump.err1;
	  rightNode.err = minStump.err2;
          nodes[top.nodeIdx].isLeaf = false;
          nodes[top.nodeIdx].featureIdx = minCol;
          nodes[top.nodeIdx].value = minStump.threshold;
          nodes[top.nodeIdx].leftNodeIdx = nodes.size();
          nodes.push_back(leftNode);
          nodes[top.nodeIdx].rightNodeIdx = nodes.size();
          nodes.push_back(rightNode);

          bool leftIsLeaf = top.depth >= maxDepth || minStump.n1 < minNodeSize || minStump.err1 / minStump.n1 < minChildErr;
          bool rightIsLeaf = top.depth >= maxDepth || minStump.n2 < minNodeSize || minStump.err2 / minStump.n2 < minChildErr;
          
          RegressionStackEntry leftEntry(minStump.n1);
          RegressionStackEntry rightEntry(minStump.n2);
          
          // If both children are leaves, we don't need to find the indices of their examples
          if(!leftIsLeaf || !rightIsLeaf) {
            leftEntry.depth = rightEntry.depth = top.depth + 1;
            
            for(unsigned i = 0; i < top.idxList.size(); ++i){
              if (X(top.idxList[i], minCol) <= minStump.threshold){
                if (!leftIsLeaf) leftEntry.idxList.push_back(top.idxList[i]);
              } else {
                if (!rightIsLeaf) rightEntry.idxList.push_back(top.idxList[i]);
              }
            }
          }
          
          if(leftIsLeaf){
            nodes[nodes[top.nodeIdx].leftNodeIdx].isLeaf = true;
            nodes[nodes[top.nodeIdx].leftNodeIdx].value = minStump.y1;
          }else{
            leftEntry.nodeIdx = nodes[top.nodeIdx].leftNodeIdx;
            nodeStack.push(leftEntry);
          }
          
          if(rightIsLeaf){
            nodes[nodes[top.nodeIdx].rightNodeIdx].isLeaf = true;
            nodes[nodes[top.nodeIdx].rightNodeIdx].value = minStump.y2;
          }else{
            rightEntry.nodeIdx = nodes[top.nodeIdx].rightNodeIdx;
            nodeStack.push(rightEntry);
          }
          //cout << "Left list n=" << leftEntry.idxList.size() << ", right list n=" << rightEntry.idxList.size() << endl;
        }
      }
    }
    
    template < typename FeatureDerived2, typename ResponseDerived2 >
    void predict(const MatrixBase<FeatureDerived2> &X, vector<unsigned> &sampIdxs, 
		 MatrixBase<ResponseDerived2> &pred, unsigned nt = NUM_SEARCH_THREADS) const {
      // pred must either have one row per example, or as many rows as sampIdxs. In the former case, the rows not in sampIdx will be 0.
  
      const unsigned N = sampIdxs.size();
      pred.setZero();

      #pragma omp parallel for num_threads(nt)
      for (unsigned i = 0; i < N; ++i){
        unsigned curNode = 0;
        unsigned j = (pred.rows() == X.rows())? sampIdxs[i] : i; 
        while(true) {
          if(nodes[curNode].isLeaf){
            pred(j) = nodes[curNode].value;
            break;
          }
          
          if (X(sampIdxs[i], nodes[curNode].featureIdx) < nodes[curNode].value )
            curNode = nodes[curNode].leftNodeIdx;
          else
            curNode = nodes[curNode].rightNodeIdx;
        }
      }
    }

    void varImportance(vector<ValueType>& imp) const{
      unsigned nnodes = nodes.size();
      for(unsigned i = 0; i < nnodes; ++i){
     	if(!nodes[i].isLeaf){
	  unsigned f = nodes[i].featureIdx;
	  if(f >= imp.size()){
	    imp.resize(f + 1, 0.0);
	  }
     	  imp[f] += nodes[i].err - nodes[nodes[i].leftNodeIdx].err - nodes[nodes[i].rightNodeIdx].err;
     	}
      }
     }

    template <typename Derived>
    void varImportance(MatrixBase<Derived>& imp) const{
      unsigned nnodes = nodes.size();
      for(unsigned i = 0; i < nnodes; ++i){
     	if(!nodes[i].isLeaf){
	  unsigned f = nodes[i].featureIdx;
	  assert(f < imp.size()); // imp MUST be of the right size
     	  imp(f) += nodes[i].err - nodes[nodes[i].leftNodeIdx].err - nodes[nodes[i].rightNodeIdx].err;
     	}
      }
     }
  };
}

#endif //_GBOOST_REGTREE_H_

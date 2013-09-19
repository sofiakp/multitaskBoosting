#ifndef _TASK_TREE_BOOSTER_H_
#define _TASK_TREE_BOOSTER_H_

#include <sys/time.h>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "RegressionTree.h"
#include <assert.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>
#include "matlab_eigen.h"

using namespace Eigen;
using namespace std;

// namespace boost{
//   template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
//   inline void serialize(Archive & ar, Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
// 			const unsigned int version){
//     for(size_t i = 0; i < t.size(); ++i)
//       ar & t.data()[i];
//   }
// }

namespace GBoost {
  template <typename ValueType> struct TaskTreeBoosterParams {
    unsigned maxDepth;
    unsigned minNodeSize; 
    ValueType minChildErr;
    double fracFeat;
    double shrink;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version){
      ar & maxDepth;
      ar & minNodeSize;
      ar & minChildErr;
      ar & fracFeat;
      ar & shrink;
    }

    TaskTreeBoosterParams(){}

    TaskTreeBoosterParams(unsigned maxDepthIn, unsigned minNodeSizeIn, 
			  ValueType minChildErrIn, double fracFeatIn, double shrinkIn):
      maxDepth(maxDepthIn),minNodeSize(minNodeSizeIn),minChildErr(minChildErrIn),
      fracFeat(fracFeatIn),shrink(shrinkIn){}
    
    bool operator==(const TaskTreeBoosterParams &other) const {
     return maxDepth==other.maxDepth && minNodeSize == other.minNodeSize && minChildErr == other.minChildErr && 
       fracFeat == other.fracFeat && shrink == other.shrink;
    }
    
    bool operator!=(const TaskTreeBoosterParams &other) const {
      return !(*this == other);
    }

    void print(){
      cout << "maxDepth:" << maxDepth << ", minNodeSize:" << minNodeSize << ", minChildErr:" << minChildErr
	   << ", fracFeat:" << fracFeat << ", shrink:" << shrink << endl;
    }
  };

  template < typename TaskDerived, typename FeatureDerived, typename ResponseDerived > class TaskTreeBooster {
    typedef typename FeatureDerived::Scalar ValueType; // Type of the features (eg. double)
    typedef typename FeatureDerived::Index IndexType; // Type of indices in the features (eg. unsigned)
    typedef typename ResponseDerived::Scalar ResponseValueType; // Type of the responses
    typedef typename TaskDerived::Scalar TaskValueType;
    typedef typename TaskDerived::Index TaskIndexType;
    typedef typename SparseMatrix<TaskValueType>::InnerIterator TaskIteratorType;
    
    typedef Matrix<ValueType, Dynamic, 1> ValueVectorType;
    typedef Matrix<ResponseValueType, Dynamic, 1> ResponseValueVectorType;
    
    typedef RegressionTree< FeatureDerived, ResponseValueVectorType > LearnerType;
  private:
    vector < vector < unsigned > > taskIdx; // same information as taskInd, but list of indices of examples per task
    vector < vector < unsigned > > taskOvIdx; // for each task, list of indices of overlapping tasks
    vector<LearnerType> learners; // learner for each boosting round
    vector<double> alphas; // alphas for each boosting round
    vector<unsigned> bestTasks; // chosen task for each iteration
    unsigned ntasks;
    unsigned nfeat;
    ResponseValueVectorType trloss; ResponseValueVectorType tsloss;
    ResponseValueVectorType F;
    TaskTreeBoosterParams<ValueType> learnParams;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
      ar & ntasks;
      ar & taskIdx;
      ar & taskOvIdx;
      ar & learners;
      ar & alphas;
      ar & bestTasks;
      ar & nfeat;
      ar & trloss; 
      ar & tsloss;
      ar & F;
      ar & learnParams;
    }

    template < typename Derived>
    unsigned getTaskIdx(const SparseMatrixBase<TaskDerived> &tInd, vector < vector < unsigned > > &tIdx, MatrixBase<Derived> &taskVec){
      unsigned ntasks = tInd.outerSize();
      tIdx.clear();
      tIdx.reserve(ntasks);
      taskVec.setZero();

      // Iterate through the examples of each task.
      const TaskIndexType *outer = tInd.derived().outerIndexPtr();
      const TaskIndexType *inner = tInd.derived().innerIndexPtr();
      for(unsigned i = 0; i < ntasks; ++i){
        vector<unsigned> taskList(outer[i + 1] - outer[i], 0);
        for(TaskIndexType j = outer[i]; j < outer[i+1]; ++j){
          taskList[j - outer[i]] = (unsigned)inner[j];
	  taskVec(inner[j], 0) = 1;
        }
        tIdx.push_back(taskList);
       }
      /*
      for(unsigned i = 0; i < taskInd.outerSize(); ++i){
        taskIdx[i].clear();
        taskNonOvIdx[i].clear();
        for(typename SparseMatrix<TaskDerived>::InnerIterator it(taskInd, i); it; ++it){
          taskIdx[i].push_back(it.index());
        }
      }
      */
      return ntasks;
    }

    template <typename Derived, typename Derived2>
    void readTasks(const SparseMatrixBase<TaskDerived> &taskInd, MatrixBase<Derived> &tr, const MatrixBase<Derived2> &taskOv){
      
      ntasks = getTaskIdx(taskInd, taskIdx, tr);

      taskOvIdx.clear();
      taskOvIdx.reserve(ntasks);
      for(unsigned i = 0; i < ntasks; ++i){
       vector<unsigned> taskOvTmp;
        taskOvTmp.clear();
        taskOvTmp.reserve(ntasks);
        taskOvIdx.push_back(taskOvTmp);
      }

      for(unsigned i = 0; i < ntasks; ++i){
	for(unsigned j = 0; j < ntasks; ++j){
	  if(taskOv(i, j) > 0) taskOvIdx[i].push_back(j);
	}
      }

      // for(unsigned i = 0; i < ntasks; ++i){
      //  	taskOvIdx[i].push_back(i);
      //   for(unsigned j = i + 1; j < ntasks; ++j){
      // 	  if(abs(levels[i] - levels[j]) > 1) continue;
      //     vector<unsigned> inter(max(taskIdx[i].size(), taskIdx[j].size()));
      //     // taskIdx[i] is sorted. interIt will be a pointer to the last position of the intersection
      //     vector<unsigned>::iterator interIt = set_intersection(taskIdx[i].begin(), taskIdx[i].end(), 
      // 								taskIdx[j].begin(), taskIdx[j].end(), inter.begin());
      //     if(inter.begin() != interIt){ // Intersection is not empty
      //       taskOvIdx[i].push_back(j);
      //       taskOvIdx[j].push_back(i);
      //     }
      //   }
      // }
    }
    
    long elapsedTime(timeval& start, timeval& end){
      return (long)((end.tv_sec  - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0 + 0.5);
    }

  public:
    const vector<LearnerType>& getLearners() const{
      return learners;
    }
    const vector<double>& getAlphas() const{
      return alphas;
    }
    const vector<unsigned>& getBestTasks() const{
      return bestTasks;
    }
    const ResponseValueVectorType& getF() const{
      return F;
    }
    const ResponseValueVectorType& getTrLoss() const{
      return trloss;
    }
    const ResponseValueVectorType& getTsLoss() const{
      return tsloss;
    }

    TaskTreeBooster(){}
    
    void printInfo(){
      cout << "Learning parameters" << endl;
      learnParams.print();
      for(unsigned i = 0; i < bestTasks.size(); ++i){
	cout << "Iter " << i << ", task " << bestTasks[i] << ", loss (tr,ts) " << trloss[i] << ", " << tsloss[i] << endl;
	learners[i].printInfo();
      }
    }

    void store(const char* filename) const{
      std::ofstream ofs(filename);
      boost::archive::text_oarchive oa(ofs);
      oa << (*this);
      ofs.flush();
    }

    void load(const char* filename){
      std::ifstream ifs(filename);
      boost::archive::text_iarchive ia(ifs);
      ia >> (*this);
    }

    template < typename OtherDerived>
    void setPseudoResiduals(const MatrixBase<ResponseDerived> &Y, const MatrixBase<OtherDerived> &F, 
			    MatrixBase<OtherDerived> &W, MatrixBase<OtherDerived> &R){
        W.setOnes();
        R = Y - F; // Negative gradient devided by weights
    }
    
    template < typename Derived>
    ResponseValueType computeLoss(const MatrixBase<ResponseDerived> &Y, const MatrixBase<Derived> &F){
      return (Y - F).cwiseAbs2().sum();
    }

    template < typename Derived, typename OtherDerived>
    ResponseValueType computeLoss(const MatrixBase<ResponseDerived> &Y, const MatrixBase<Derived> &F, const SparseMatrixBase<OtherDerived> &ind){
      return (ind.cwiseProduct(Y - F)).cwiseAbs2().sum();
    }

    template < typename Derived2 >
    void learn(const SparseMatrixBase<TaskDerived> &taskInd, const SparseMatrixBase<TaskDerived> &testInd, const MatrixBase<Derived2> &taskOv,
	       const MatrixBase<FeatureDerived> &X, const MatrixBase<ResponseDerived> &Y, unsigned niter, unsigned maxDepth, unsigned minNodeSize, 
	       ValueType minChildErr, double fracFeat, double shrink, bool resume, const char* filename){

      unsigned startIter;
      unsigned nexamples = X.rows();
      timeval start, end;
      cout << "Initializing..." << endl;
      gettimeofday(&start, NULL);

      if(resume){
	startIter = bestTasks.size();
	if(niter <= startIter){
	  cerr << "In TaskTreeBooster::learn: Resume niter is smaller than current number of iterations. Nothing to do." << endl;
	  return;
	}
	assert(X.cols() == nfeat);
	assert(F.size() == X.rows());
	assert(learnParams == TaskTreeBoosterParams<ValueType>(maxDepth,minNodeSize,minChildErr,fracFeat,shrink));
      }else{
	startIter = 0;
	learnParams = TaskTreeBoosterParams<ValueType>(maxDepth,minNodeSize,minChildErr,fracFeat,shrink);
	nfeat = X.cols();
	learners.clear();
	alphas.clear();
	bestTasks.clear();
	F.resize(nexamples);
	F.setZero(); // Initial prediction is zero for each example
      }
      // These are very basic checks...
      assert(taskInd.outerSize() == testInd.outerSize());
      assert(taskInd.innerSize() == testInd.innerSize());
      assert(taskInd.innerSize() == X.rows());
      assert(taskOv.rows() == taskInd.outerSize());
      assert(taskOv.cols() == taskInd.outerSize());
      assert(X.rows() == Y.rows());

      VectorXd tr(nexamples);
      readTasks(taskInd, tr, taskOv);
      const SparseMatrix<double, ColMajor, long> tr_sp = tr.sparseView();
      learners.reserve(niter);
      alphas.reserve(niter);
      bestTasks.reserve(niter);
      trloss.resize(niter);
      tsloss.resize(niter);

      vector < vector < unsigned > > testIdx;
      VectorXd ts(nexamples);
      getTaskIdx(testInd, testIdx, ts);
      const SparseMatrix<double, ColMajor, long> ts_sp = ts.sparseView();
      unsigned ntr = tr_sp.nonZeros();
      unsigned nts = ts_sp.nonZeros();

      vector<LearnerType> taskLearners; // best learner for each task
      taskLearners.clear();
      taskLearners.reserve(ntasks);
      for(unsigned i = 0; i < ntasks; ++i){
        taskLearners.push_back(RegressionTree< FeatureDerived, ResponseValueVectorType >());
      }
     
      vector<double> taskAlphas(ntasks); // best alpha for each task
      ResponseValueVectorType taskErr(ntasks); // Reduction in loss for each task
      ResponseValueVectorType W(nexamples);
      ResponseValueVectorType R(nexamples);
      
      vector< unsigned > allTasks(ntasks); // This will be just a vector 0,1,...,ntasks-1
      for(unsigned i = 0; i < ntasks; ++i) allTasks[i] = i;

      gettimeofday(&end, NULL);
      cout << "Time elapsed: " << elapsedTime(start, end) << " ms" << endl;

      for(unsigned iter = startIter; iter < niter; ++iter){
	gettimeofday(&start, NULL);
      	cout << "Boosting iter " << iter << endl;

	setPseudoResiduals(Y, F, W, R);

	ResponseValueType oldErr = (iter > startIter)? tsloss(iter - 1): computeLoss(Y, F, tr_sp); // Error of last iteration
        vector< unsigned > todoTasks = (iter > startIter)? taskOvIdx[bestTasks[iter - 1]]: allTasks; // We only need to recompute the models for tasks that overlap the last task
        cout << "# Tasks " << todoTasks.size() << endl;
        
        // Iterate through the tasks whose optimal model has changed since the last iteration
        #pragma omp parallel for num_threads(NUM_BOOST_THREADS)
        for(unsigned t = 0; t < todoTasks.size(); ++t){
          // cout << "Learner " << t << endl;
          taskLearners[todoTasks[t]].learn(X, R, W, maxDepth, minNodeSize, minChildErr, taskIdx[todoTasks[t]], fracFeat);
          ResponseValueVectorType taskPred(nexamples); 
          taskLearners[todoTasks[t]].predict(X, taskIdx[todoTasks[t]], taskPred, 1); // taskPred will be 0 when taskIdx is 0
          taskAlphas[todoTasks[t]] = shrink;
	  taskErr(todoTasks[t]) = computeLoss(Y, F, taskInd.col(todoTasks[t])) - 
	    computeLoss(Y, F + taskAlphas[todoTasks[t]] * taskPred, taskInd.col(todoTasks[t])); 
        }

        //Find the task that gives the minimum error
        unsigned tmp = 0;
        taskErr.maxCoeff(&tmp); // Maximum reduction in loss  
        bestTasks.push_back(tmp);
        learners.push_back(taskLearners[bestTasks[iter]]);
	learners[iter].printInfo();
        alphas.push_back(taskAlphas[bestTasks[iter]]);
        
	vector<unsigned> indUnion(taskIdx[bestTasks[iter]].size() + testIdx[bestTasks[iter]].size());
	vector<unsigned>::iterator it = set_union(taskIdx[bestTasks[iter]].begin(), taskIdx[bestTasks[iter]].end(), 
						  testIdx[bestTasks[iter]].begin(), testIdx[bestTasks[iter]].end(), indUnion.begin());
	indUnion.resize(it - indUnion.begin());
	ResponseValueVectorType taskPred(nexamples);
	taskLearners[bestTasks[iter]].predict(X, indUnion, taskPred);
        F = F + alphas[iter] * taskPred; 
        trloss(iter) = oldErr - taskErr(bestTasks[iter]);
	tsloss(iter) = computeLoss(Y, F, ts_sp);
	cout << "Best task " << bestTasks[iter] << ", Avg loss " << trloss(iter) / ntr << " " << tsloss(iter) / nts << endl;
	
	store(filename);
	gettimeofday(&end, NULL);
	cout << "Time elapsed: " << elapsedTime(start, end) << " ms" << endl;
      }
    }
    
    void reverseBestTask(vector< vector < unsigned > > &revBestTasks){
      revBestTasks.reserve(ntasks);
      for(unsigned i = 0; i < bestTasks.size(); ++i)
        revBestTasks[bestTasks[i]].push_back(i);
    }
    
    template < typename ResponseDerived2 >
    void predict(const SparseMatrixBase<TaskDerived> &tInd, const MatrixBase<FeatureDerived> &X, 
		  MatrixBase<ResponseDerived2> &pred) {

      typedef typename ResponseDerived2::Scalar RType;

      assert(tInd.cols() == ntasks);
      vector < vector < unsigned > > tIdx;
      MatrixXd tmp(X.rows(), 1);
      getTaskIdx(tInd, tIdx, tmp);

      // #pragma omp parallel for num_threads(NUM_SEARCH_THREADS) TODO: HOW DO YOU MAKE THIS PARALLEL?
      for(unsigned i = 0; i < bestTasks.size(); ++i){
	Matrix<RType, Dynamic, 1> treePred(X.rows()); // Predictions of a single tree, for one iteration
	learners[i].predict(X, tIdx[bestTasks[i]], treePred);
	pred = pred + alphas[i] * treePred;
      }

      // vector < vector < unsigned > > revBestTasks; // revBestTasks[k] has indices of iterations where the k-th task was selected
      // reverseBestTask(revBestTasks);
      
      // for(unsigned k = 0; k < ntasks; ++k){
      //   vector<unsigned> tIdx; // Indices of examples that belong to the k-th task
      //   for(TaskIteratorType it(tInd, k); it; ++it){
      //     tIdx.push_back(it.index());
      //   }
      //   for(unsigned i = 0; i < revBestTasks[k].size(); ++i){
      //     unsigned j = revBestTasks[k][i];

      //   }
      // }
    }

    template < typename ResponseDerived2 >
    void varImportance(MatrixBase<ResponseDerived2>& imp){
      typedef typename ResponseDerived2::Scalar RType;
      
      assert(imp.rows() == nfeat);
      assert(imp.cols() == ntasks);
      imp.setZero();

      for(unsigned i = 0; i < bestTasks.size(); ++i){
	Matrix<RType, Dynamic, 1> c(nfeat);
	c.setZero();
	learners[i].varImportance(c); 
	imp.col(bestTasks[i]) = imp.col(bestTasks[i]) + c;
      }
    }
  };
}

#endif //_TASK_TREE_BOSTER_H

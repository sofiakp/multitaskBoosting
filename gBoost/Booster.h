#ifndef _TASK_TREE_BOOSTER_H_
#define _TASK_TREE_BOOSTER_H_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "RegressionTree.h"
//#include "LineSearch.h"
#include <assert.h>

using namespace Eigen;
using namespace std;

namespace GBoost {
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
    // TODO: this should be a pointer?
    //SparseMatrixBase<TaskDerived> taskInd; // sparse binary matrix nexamples x ntasks. Should be column major.
    vector < vector < unsigned > > taskIdx; // same information as taskInd, but list of indices of examples per task
    vector < vector < unsigned > > taskOvIdx; // for each task, list of indices of overlapping tasks
    vector<LearnerType> learners; // learner for each boosting round
    vector<double> alphas; // alphas for each boosting round
    vector<unsigned> bestTasks; // chosen task for each iteration
    unsigned ntasks;
    
  public:
    vector<LearnerType>& getLearners() const{
      return learners;
    }
    vector<double>& getAlphas() const{
      return alphas;
    }
    vector<unsigned>& getBestTasks() const{
      return bestTasks;
    }
    
    TaskTreeBooster(){}
    
    void readTasks(const SparseMatrixBase<TaskDerived> &taskInd){
      
      ntasks = taskInd.outerSize();
      taskIdx.clear();
      taskIdx.reserve(ntasks);
      taskOvIdx.clear();
      taskOvIdx.reserve(ntasks);
      
      // Iterate through the examples of each task.
      const TaskIndexType *outer = taskInd.derived().outerIndexPtr();
      const TaskIndexType *inner = taskInd.derived().innerIndexPtr();
      for(unsigned i = 0; i < ntasks; ++i){
        vector<unsigned> taskList(outer[i + 1] - outer[i], 0);
        for(TaskIndexType j = outer[i]; j < outer[i+1]; ++j){
          taskList[j - outer[i]] = (unsigned)inner[j];
        }
        taskIdx.push_back(taskList);
        vector<unsigned> taskOvTmp;
        taskOvTmp.clear();
        taskOvTmp.reserve(ntasks);
        taskOvIdx.push_back(taskOvTmp);
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
      for(unsigned i = 0; i < ntasks; ++i){
	taskOvIdx[i].push_back(i);
        for(unsigned j = i + 1; j < ntasks; ++j){
          vector<unsigned> inter(max(taskIdx[i].size(), taskIdx[j].size()));
          // taskIdx[i] is sorted. interIt will be a pointer to the last position of the intersection
          vector<unsigned>::iterator interIt = set_intersection(taskIdx[i].begin(), taskIdx[i].end(), taskIdx[j].begin(), taskIdx[j].end(), inter.begin());
          if(inter.begin() != interIt){ // Intersection is not empty
            taskOvIdx[i].push_back(j);
            taskOvIdx[j].push_back(i);
          }
        }
      }
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
      return ind.cwiseProduct(Y - F).cwiseAbs2().sum();
    }

    template < typename OutDerived >
    void learn(const SparseMatrixBase<TaskDerived> &taskInd, 
            const MatrixBase<FeatureDerived> &X, const MatrixBase<ResponseDerived> &Y, unsigned niter, unsigned maxDepth, unsigned minNodeSize, 
            ValueType minChildErr, double fracFeat, double shrink, MatrixBase<OutDerived> &err){
      
      readTasks(taskInd);
      learners.clear();
      learners.reserve(niter);
      alphas.clear();
      alphas.reserve(niter);
      bestTasks.clear();
      bestTasks.reserve(niter);
      
      unsigned nexamples = X.rows();
      vector<LearnerType> taskLearners; // best learner for each task
      taskLearners.clear();
      taskLearners.reserve(ntasks);
      for(unsigned i = 0; i < ntasks; ++i){
        taskLearners.push_back(RegressionTree< FeatureDerived, ResponseValueVectorType >());
      }
      
      vector<double> taskAlphas(ntasks); // best alpha for each task
      ResponseValueVectorType taskErr(ntasks); // Error for each task
      ResponseValueVectorType W(nexamples);
      ResponseValueVectorType R(nexamples);
      ResponseValueVectorType F(nexamples);
      F.setZero(); // Initial prediction is zero for each example
      
      vector< unsigned > allTasks(ntasks); // This will be just a vector 0,1,...,ntasks-1
      for(unsigned i = 0; i < ntasks; ++i) allTasks[i] = i;
      
      for(unsigned iter = 0; iter < niter; ++iter){
      
	setPseudoResiduals(Y, F, W, R);

	//ResponseValueType oldErr = (iter > 0)? err(iter - 1): computeLoss(Y, F); // Error of last iteration
        vector< unsigned > todoTasks = (iter > 0)? taskOvIdx[bestTasks[iter - 1]]: allTasks; // We only need to recompute the models for tasks that overlap the last task that we updated.
      
        cout << "# Tasks " << todoTasks.size() << endl;
        
        // Iterate through the tasks whose optimal model has changed since the last iteration
        //#pragma omp parallel for num_threads(NUM_BOOST_THREADS)
        for(unsigned t = 0; t < todoTasks.size(); ++t){
          cout << "Learner " << t << endl;
          taskLearners[todoTasks[t]].learn(X, R, W, maxDepth, minNodeSize, minChildErr, taskIdx[todoTasks[t]], fracFeat);
          cout << "Done learning " << t << endl;
          ResponseValueVectorType taskPred(nexamples); 
          taskLearners[todoTasks[t]].predict(X, taskIdx[todoTasks[t]], taskPred); // taskPred will be 0 when taskIdx is 0
          taskAlphas[todoTasks[t]] = shrink;
	  taskErr(todoTasks[t]) = computeLoss(Y, F + taskAlphas[todoTasks[t]] * taskPred); 
          //taskPartErr(todoTasks[t]) = .cwiseProduct(R).cwiseAbs2().sum() - taskInd.col(todoTasks[t]).cwiseProduct().cwiseAbs2().sum(); 
        }

        //Find the task that gives the minimum error
        unsigned tmp = 0;
        taskErr.minCoeff(&tmp);
        bestTasks.push_back(tmp);
        learners.push_back(taskLearners[bestTasks[iter]]);
        alphas.push_back(taskAlphas[bestTasks[iter]]);
        err(iter) = taskErr(bestTasks[iter]);
        ResponseValueVectorType taskPred(nexamples);
        taskLearners[bestTasks[iter]].predict(X, taskIdx[bestTasks[iter]], taskPred);
        F = F + alphas[iter] * taskPred;
      }
    }
    
    void reverseBestTask(vector< vector < unsigned > > &revBestTasks){
      revBestTasks.reserve(ntasks);
      for(unsigned i = 0; i < bestTasks.size(); ++i)
        revBestTasks[bestTasks[i]].push_back(i);
    }
    
    template < typename TaskDerived2, typename FeatureDerived2, typename ResponseDerived2 >
            void	predict(const SparseMatrixBase<TaskDerived2> &tInd, const MatrixBase<FeatureDerived2> &X, vector<unsigned> &sampIdxs,
            MatrixBase<ResponseDerived2> &pred) const {
      
      assert(tInd.cols() == ntasks);
      MatrixBase<ResponseDerived2> treePred(X.rows()); // Predictions of a single tree, for one iteration
      vector < vector < unsigned > > revBestTasks; // revBestTasks[k] has indices of iterations where the k-th task was selected
      reverseBestTask(revBestTasks);
      
      for(unsigned k = 0; k < ntasks; ++k){
        vector<unsigned> tIdx; // Indices of examples that belong to the k-th task
        for(TaskIteratorType it(tInd, k); it; ++it){
          tIdx.push_back(it.index());
        }
        for(unsigned i = 0; i < revBestTasks[k].size(); ++i){
          unsigned j = revBestTasks[k][i];
          learners[j].predict(X, tIdx, treePred);
          pred = pred + alphas[j] * tInd.col(k).cwiseProd(treePred);
        }
      }
    }
  };
}

#endif //_TASK_TREE_BOSTER_H

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv4/opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <chrono>


using namespace std;

/*CurveFittingVertex and CurveFittingEdge are from g2o's built in classes BaseVertex 
and BaseUnaryEdge */


// vertex - 3d vector
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vertex reset function 
    virtual void setToOriginImpl() override {
    _estimate << 0, 0, 0;
  }

    // override plus operation and use plain addition
    /*
    This deals with the incremental operation : x = x_prev + del x  
    We need to define how this update happens. In curve fitting, the parameters are in plain
    vector space, so the update is just simple addition. 
    */

    virtual void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector3d(update);
  }

    // dummy read and write function
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};


// edge -   1d error term - unary in nature for curve fitting
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // define the error term computation
    virtual void computeError() override {
    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
  } 


    // jacobian - calculate the jacobian of each edge relative to the vertex.
    virtual void linearizeOplus() override {
    const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
    const Eigen::Vector3d abc = v->estimate();
    double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    _jacobianOplusXi[0] = -_x * _x * y;
    _jacobianOplusXi[1] = -_x * y;
    _jacobianOplusXi[2] = -y;
  }
        
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}

public:
  double _x; //x data , y is in _measurement
};


int main(int argc, char **argv) {


    //data sampling part
  double ar = 1.0, br = 2.0, cr = 1.0;         //ground trith
  double ae = 2.0, be = -1.0, ce = 5.0;        //initial estimates
  int N = 100;                                 
  double w_sigma = 1.0;                        
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 

  vector<double> x_data, y_data;      
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }


    // choose the optimization method - in our case Gauss Newton

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; 

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;    
    optimizer.setAlgorithm(solver);   
    optimizer.setVerbose(true); 


    // add vertex    
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    //add edges
    for (int i = 0; i < N; i++) {
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);               
        edge->setMeasurement(y_data[i]);      
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); 
        optimizer.addEdge(edge);
    }

    // optimization process
    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // resu
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

  return 0;
}


    


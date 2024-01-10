#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
# include <eigen3/Eigen/Core>
# include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char ** argv){

    // ground truth values
    double ar = 1.0, br = 2.0, cr = 1.0;
    // initial estimation
    double ae = 2.0, be = -1.0, ce = 5.0;
    // num of data points
    int N = 100;
    // std deviation of the noise
    double w_sigma = 1.0;
    double inv_sigma = 1.0/w_sigma;
    cv::RNG rng; //Random number generator


    // data
    vector<double> x_data, y_data;
    for (int i =0 ; i<N; i++){
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma));


    }

    // Gauss-Newton iterations
    int iterations = 500;
    double cost = 0, lastCost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    for( int it= 0; it < iterations; it++){

        Matrix3d H = Matrix3d::Zero(); //Hessian matrix - second derivative
        Vector3d b = Vector3d::Zero(); //bias
        cost = 0;

        for(int i = 0; i<N; i++){
            // i th data
            double x_i = x_data[i], y_i = y_data[i];
            double error = y_i - exp(ae*x_i*x_i + be*x_i + ce);

            // Jacobian
            Vector3d J;
            J[0] = -x_i*x_i*exp(ae*x_i*x_i + be*x_i + ce); //del(e)/del(a)
            J[1] = -x_i*exp(ae*x_i*x_i + be*x_i + ce); //del(e)/del(b)
            J[2] = -exp(ae*x_i*x_i + be*x_i + ce); //del(e)/del(c)

            H += inv_sigma*inv_sigma*J*J.transpose();
            b += -inv_sigma*inv_sigma*error*J;

            cost += error*error;

        }

        // Solve Hx = b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0])){
            cout << "result is nan" <<endl;
            break;
        }

        if(it > 0 and cost >= lastCost){
            cout << "cost: " << cost  << ">= last cost: " << lastCost << "breaking" << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
"\t\testimated params: " << ae << "," << be << "," << ce << endl;



    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;

}
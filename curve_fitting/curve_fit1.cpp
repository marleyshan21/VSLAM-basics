#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/inference/Symbol.h>
#include <random>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>


// write a custom exp factor
class exp_curve_fitting_factor: public gtsam::NoiseModelFactor3<double, double, double>{

    double data_y_;
    double data_x_;


    public:

        exp_curve_fitting_factor(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, const double y_, const double x_, const gtsam::SharedNoiseModel& model):
            gtsam::NoiseModelFactor3<double, double, double>(model, key1, key2, key3), data_y_(y_), data_x_(x_)  {}

        // evaluate error
        gtsam::Vector evaluateError(const double& a, const double &b, const double &c, boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none, boost::optional<gtsam::Matrix&> H3 = boost::none) const override {

            // eqn: y = exp(ax^2 + bx + c)

            gtsam::Vector error;

            std::cout << "a, b, c: " << a << ", " << b << ", " << c << std::endl;
            std::cout << "data_x, data_y: " << data_x_ << ", " << data_y_ << std::endl;
            std::cout << "exp(a * data_x_ * data_x_ + b * data_x_ + c): " << exp(a * data_x_ * data_x_ + b * data_x_ + c) << std::endl;
            std::cout << "error: " << data_y_ - exp(a * data_x_ * data_x_ + b * data_x_ + c) << std::endl;

            error(0) = data_y_ - exp(a * data_x_ * data_x_ + b * data_x_ + c);


            // H1 is the jacobian of error wrt a
            if (H1) {
                (*H1)(0, 0) = -data_x_ * data_x_ * exp(a * data_x_ * data_x_ + b * data_x_ + c);
            }
            // H2 is the jacobian of error wrt b
            if (H2) {
                (*H2)(0, 0) = -data_x_ * exp(a * data_x_ * data_x_ + b * data_x_ + c);
            }
            // H3 is the jacobian of error wrt c
            if (H3) {
                (*H3)(0, 0) = -exp(a * data_x_ * data_x_ + b * data_x_ + c);
            }

            return error;
        }
};

// prepare another function to generate data
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> generate_data(){

    // generate data using the equation y = exp(ax^2 + bx + c)

    // generate a, b, c
    double a = 1.0;
    double b = 2.0;
    double c = 1.0;

    // generate 100 double values of x
    std::vector<double> x(100);
    for (int i = 0; i < 100; i++){
        x[i] = i / 100.0;
    }

    // generate y values using the equation
    std::vector<double> y(100);
    for (int i = 0; i < 100; i++){
        y[i] = exp(a * x[i] * x[i] + b * x[i] + c);
    }

    // add noise to the y values
    std::vector<double> y_noise(100);
    // create a gaussian noise model
    const double  mean = 0.0;
    const double std_dev = 1.0;
    std::default_random_engine generator;
    std::normal_distribution<double> noise_model(mean, std_dev);


    for (int i = 0; i < 100; i++){
        y_noise[i] = y[i] + noise_model(generator);
    }

    return std::make_tuple(x, y_noise, y);
}




// prepare the main code to fit a curve using this custom exp factor
void build_factor_graph(){


    // generate data
    std::vector<double> x, y_noise, y;
    std::tie(x, y_noise, y) = generate_data();

    // factor graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values InitialEstimate;

    // noise model
    auto noise_model = gtsam::noiseModel::Isotropic::Sigma(1, 1);

    // add factors to the graph
    for (size_t i = 0; i < x.size(); i++){
        // add a factor for each data point
        graph.emplace_shared<exp_curve_fitting_factor>(gtsam::Symbol('a', 0), gtsam::Symbol('b', 0), gtsam::Symbol('c', 0), y_noise[i], x[i], noise_model);
    }

    // add initial estimate
    InitialEstimate.insert(gtsam::Symbol('a', 0), 3.0);
    InitialEstimate.insert(gtsam::Symbol('b', 0), 3.0);
    InitialEstimate.insert(gtsam::Symbol('c', 0), 3.0);

    // optimize

    gtsam::LevenbergMarquardtParams params;
    params.setVerbosity("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, InitialEstimate, params);

    gtsam::Values result = optimizer.optimize();

    // print the result
    std::cout << "Optimization complete" << std::endl;
    std::cout << "Initial error: " << graph.error(InitialEstimate) << std::endl;
    std::cout << "Final error: " << graph.error(result) << std::endl;
    std::cout << "Final result: " << std::endl;
    result.print("Final result: ");

}





int main(){

        build_factor_graph();

        return 0;
}
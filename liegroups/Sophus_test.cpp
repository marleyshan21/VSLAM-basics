/*
 * This code is a test to understand the basics of Lie
 * groups and perturbations using Sophus library.
 */

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

/*
 * Reference:
 * https://github.com/gaoxiang12/slambook/blob/master/ch4/useSophus/useSophus.cpp
 */


int main(){

    // Rotation matric with 90 degrees along z-axis
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();

    // in Sophus
    Sophus::SO3d SO3_R(R); // SO(3) from rotation matrix
    std::cout << "SO(3) from matrix: " << std::endl << SO3_R.matrix() << std::endl;

    // Use logmap to get the Lie algebra
    Eigen::Vector3d so3 = SO3_R.log();
    std::cout << "so3 = " << std::endl << so3.transpose() << std::endl;

    // use hat to get the skew symmetric matrix of the vector
    std::cout << "so3 hat = " << std::endl << Sophus::SO3d::hat(so3) << std::endl;

    // use vee to get the vector from the skew symmetric matrix
    std::cout << "so3 hat vee = " << std::endl << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << std::endl;

    // update using the pertubation model

    Eigen::Vector3d update_so3(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    std::cout << "SO3 updated = " << std::endl << SO3_updated.matrix() << std::endl;

    std::cout << "********************************************" << std::endl;

    // SE(3) operations
    Eigen::Vector3d t(1,0,0); // translation vector of 1 along x-axis
    Sophus::SE3d SE3_Rt(R,t); // SE(3) from R,t

    // Lie algebra se(3) is a 6d vector
    // represent a 6d vector using eigen matrix using typedefs
    typedef Eigen::Matrix<double,6,1> Vector6d;

    Vector6d se3 = SE3_Rt.log();
    std::cout << "se3 = " << std::endl << se3.transpose() << std::endl;

    // update using the pertubation model
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0) = 1e-4d;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    std::cout << "SE3 updated = " << std::endl << SE3_updated.matrix() << std::endl;

    return 0;
}
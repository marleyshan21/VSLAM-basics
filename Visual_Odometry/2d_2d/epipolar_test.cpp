#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <chrono>

std::vector<cv::DMatch> remove_outliers(std::vector<cv::DMatch> matches){

    auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                       [](const cv::DMatch &m1, const cv::DMatch &m2){
                                           return m1.distance < m2.distance;
                                       });

    double min_dist = min_max.first->distance;

    std::vector<cv::DMatch> good_matches;
    for(auto & match : matches){
        if(match.distance <= std::max(2*min_dist, 30.0)){
            good_matches.push_back(match);
        }
    }
    return good_matches;
}


void feature_matching(cv::Mat img1, cv::Mat img2, std::vector<cv::KeyPoint> &keypoints1,
                      std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches){

    // create the orb feature detector, descriptor and matcher
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // detect the orb features
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // compute the orb descriptors
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    // match the descriptors
    std::vector<cv::DMatch> all_matches;
    matcher->match(descriptors1, descriptors2, all_matches);

    // remove the outliers
    std::vector<cv::DMatch> good_matches = remove_outliers(all_matches);


    // store the matches
    matches = good_matches;

    // draw the matches
//    cv::Mat img_matches;
//    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
//    cv::imshow("matches", img_matches);
//    cv::waitKey(0);

}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints1, std::vector<cv::KeyPoint> keypoints2,
                          std::vector<cv::DMatch> matches, cv::Mat &R, cv::Mat &t){

    // camera intrinsic parameters
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2f> points1, points2;
    for(size_t i = 0; i < matches.size(); i++){
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // compute the fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    std::cout << "fundamental matrix is " << std::endl << fundamental_matrix << std::endl;

    // compute the essential matrix
    cv::Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "essential matrix is " << std::endl << essential_matrix << std::endl;

    // compute the homography matrix
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homography matrix is " << std::endl << homography_matrix << std::endl;

    // recover the pose from the essential matrix
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);

    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t << std::endl;

}


int main(int argc, char **argv){

    if(argc != 3){
        std::cout << "usage: epipolar_test img1 img2" << std::endl;
        return 1;
    }

    // read the images
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    // assert that the images are not empty
    assert(img1.data != nullptr && img2.data != nullptr);

    // find feature matches
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    feature_matching(img1, img2, keypoints1, keypoints2, matches);

    // estimate R, t using the matches
    cv::Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);

    return 0;

}

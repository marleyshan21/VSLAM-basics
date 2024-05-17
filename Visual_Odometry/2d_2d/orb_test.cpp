/*
 * @brief: test the orb feature extraction
 * @reference: slambook - https://github.com/gaoxiang12/slambook-en
 *
 */

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <chrono>


std::vector<cv::DMatch> remove_outliers(std::vector<cv::DMatch> matches){

    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch &m1, const cv::DMatch &m2){
            return m1.distance < m2.distance;
        });

    double min_dist = min_max.first->distance;
//    double max_dist = min_max.second->distance;

    std::vector<cv::DMatch> good_matches;

    for(auto & match : matches){
        if(match.distance <= std::max(3*min_dist, 50.0)){
            good_matches.push_back(match);
        }
    }

    return good_matches;
}



int main(int argc, char **argv){

    // check if the input is correct
    if(argc != 3){
        std::cout << "usage: 2d_2d img1 img2" << std::endl;
        return 1;
    }

    // read the images
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    // asser thtat the images are not empty
    assert(img1.data != nullptr && img2.data != nullptr);

    // create the orb feature detector
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // detect the orb features
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // compute the orb descriptors
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract orb features cost time: " << time_used.count() << " seconds." << std::endl;

    // match the orb features
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match orb features cost time: " << time_used.count() << " seconds." << std::endl;

    // remove the outliers
    std::vector<cv::DMatch> good_matches = remove_outliers(matches);

    // draw the matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    cv::imshow("matches", img_matches);
    cv::waitKey(0);

    return 0;


}
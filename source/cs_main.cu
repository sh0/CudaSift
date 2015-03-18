/*
 * CudaSift library
 *
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

// Internal
#include "cs_library.h"
#include "opencv_sift.h"

// C++
#include <iostream>

// OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// Testing
int main(int argc, char** argv)
{
    // Argument checking
    if (argc < 2) {
        std::cout << "Usage: cudasift image.png" << std::endl;
        return 0;
    }

    // File loading
    std::string fn = argv[1];
    cv::Mat image_raw = cv::imread(fn, 0);
    if (image_raw.cols <= 0 || image_raw.rows <= 0) {
        std::cout << "Failed to load image!" << std::endl;
        return 0;
    }

    // Convert
    cv::Mat image_32f;
    image_raw.convertTo(image_32f, CV_32FC1);

    // Upload
    cv::cuda::GpuMat image_gpu;
    image_gpu.upload(image_32f);

    // Process (GPU)
    cudasift::s_sift sift;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints2;
    sift.detect(image_gpu, keypoints, keypoints2);

    // Process (CPU)
    /*
    #if 1
        cv::debug::SIFT_Impl sift2;
        std::vector<cv::KeyPoint> keypoints2;
        cv::Mat descriptors2;
        sift2.detect(image_raw, keypoints2, descriptors2);
    #else
        cv::Ptr<cv::xfeatures2d::SIFT> sift2 = cv::xfeatures2d::SIFT::create();
        std::vector<cv::KeyPoint> keypoints2;
        sift2->detect(image_raw, keypoints2);
    #endif
    */

    // Convert to color
    cv::Mat image_rgb;
    cv::cvtColor(image_raw, image_rgb, cv::COLOR_GRAY2BGR);

    // Show result
    cv::Mat result1 = image_rgb.clone();
    for (size_t i = 0; i < keypoints.size(); i++)
        cv::circle(result1, keypoints[i].pt, 3, cv::Scalar(0, 0, 255));

    cv::Mat result2 = image_rgb.clone();
    for (size_t i = 0; i < keypoints2.size(); i++)
        cv::circle(result2, keypoints2[i].pt, 3, cv::Scalar(0, 0, 255));

    cv::imshow("CudaSift (GPU)", result1);
    cv::imshow("CudaSift (CPU)", result2);
    cv::waitKey(-1);

    // Exit
    return 0;
}

/*
 * CudaSift library
 *
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

// Internal
#include "cs_library.h"
#include "cs_device.h"

namespace cudasift {

// Constructor
s_sift::s_sift()
{
    // Parameters
    m_num_octaves = 5;
    m_num_scales = 5;
    m_initial_blur = 0.0;
    m_threshold = 5.0;
    m_lowest_scale = 0.0;
    m_subsampling = 1.0;

    // Points and descriptors
    m_num_points = 0;
    m_max_points = 4096;
    m_sift = cv::cuda::GpuMat(CUDASIFT_POINT_SIZE, m_max_points, CV_32FC1);
    m_desc = cv::cuda::GpuMat(m_sift.cols, 128, CV_32FC1);
}

// Detector
void s_sift::detect(cv::cuda::GpuMat image, std::vector<cv::KeyPoint>& keypoints)
{
    // CUDA
    //cv::cuda::Stream stream = cv::cuda::Stream::Null();

    // Reset points and descriptors
    m_num_points = 0;

    // Extract
    extract_octaves(image);
    if (m_num_points <= 0)
        return;

    // Get orientations
    cpu_compute_orientations(image, m_sift, m_num_points, m_max_points);

    // Descriptors
    cpu_extract_sift_descriptors(image, m_sift, m_desc, m_num_points, m_max_points);

    // Download points
    cv::Mat point_mat;
    m_sift.download(point_mat);

    // Convert points
    const float* ptr_xpos = point_mat.ptr<float>(CUDASIFT_POINT_XPOS);
    const float* ptr_ypos = point_mat.ptr<float>(CUDASIFT_POINT_YPOS);
    const float* ptr_scale = point_mat.ptr<float>(CUDASIFT_POINT_SCALE);
    const float* ptr_orientation = point_mat.ptr<float>(CUDASIFT_POINT_ORIENTATION);
    const float* ptr_score = point_mat.ptr<float>(CUDASIFT_POINT_SCORE);
    for (size_t i = 0; i < m_num_points; i++) {
        cv::KeyPoint kp;
        kp.pt.x = ptr_xpos[i];
        kp.pt.y = ptr_ypos[i];
        kp.size = ptr_scale[i];
        kp.angle = ptr_orientation[i];
        kp.response = ptr_score[i];
        keypoints.push_back(kp);
    }
}

void s_sift::extract_octaves(cv::cuda::GpuMat image)
{
    double initial_blur = m_initial_blur;
    double subsampling = m_subsampling;
    for (size_t octave = 0; octave < m_num_octaves; octave++) {
        // Downscaling
        if (octave > 0) {
            cv::cuda::GpuMat subimage;
            cv::cuda::resize(image, subimage, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
            image = subimage;
        }

        // Extract scales
        initial_blur = sqrt(initial_blur * initial_blur + 0.5f * 0.5f) / 2.0f;
        subsampling *= 2.0;
        extract_scales(image, initial_blur, subsampling);
    }
}

void s_sift::extract_scales(cv::cuda::GpuMat image, double initial_blur, double subsampling)
{
    // Check
    if (m_lowest_scale >= subsampling * 2.0f)
        return;

    // Gaussian filtering
    double base_blur = pow(2.0, -1.0 / m_num_scales);
    double scale_difference = pow(2.0f, 1.0f / m_num_scales);

    std::vector<cv::cuda::GpuMat> blur_images;
    double scale = base_blur;
    for (size_t i = 0; i < m_num_scales + 3; i++) {
        // Create filter
        double sigma = sqrt((scale * scale) - (initial_blur * initial_blur));
        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(9, 9), sigma, sigma);

        // Filter
        cv::cuda::GpuMat blur_image;
        filter->apply(image, blur_image);
        blur_images.push_back(blur_image);

        // Next scale
        scale *= scale_difference;
    }

    // Subtract gaussian images (DoG)
    std::vector<cv::cuda::GpuMat> diff_images;
    for (size_t i = 0; i < blur_images.size() - 1; i++) {
        cv::cuda::GpuMat diff_image;
        cv::cuda::subtract(blur_images[i], blur_images[i + 1], diff_image, cv::noArray(), -1);
        diff_images.push_back(diff_image);
    }

    // Point finding
    scale = base_blur * scale_difference;
    for (size_t i = 0; i < diff_images.size() - 2; i++) {
        // Check scale
        if (scale >= m_lowest_scale / subsampling) {
            // Find points
            m_num_points = cpu_find_points(
                diff_images[i + 0], diff_images[i + 1], diff_images[i + 2], m_sift,
                m_threshold, m_num_points, m_max_points, 16.0f, scale, 1.0f / m_num_scales
            );
        }

        // Next scale
        scale *= scale_difference;
    }
}

};

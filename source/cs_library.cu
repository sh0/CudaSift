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
    m_max_points = 4096;
    m_sift = cv::cuda::GpuMat(CUDASIFT_POINT_SIZE, m_max_points, CV_32FC1);
    m_desc = cv::cuda::GpuMat(m_sift.cols, 128, CV_32FC1);
}

// Detector
void s_sift::detect(cv::cuda::GpuMat image, std::vector<cv::KeyPoint>& keypoints)
{
    // CUDA
    //cv::cuda::Stream stream = cv::cuda::Stream::Null();

    // Parameters
    double sigma = 1.6;
    int octave_num = cvRound(std::log(static_cast<double>(std::min(image.cols, image.rows))) / std::log(2.0) - 2.0) + 1;
    int layer_num = 3;

    // Build images
    cv::cuda::GpuMat image_initial = build_initial(image, sigma);
    std::vector<std::vector<cv::cuda::GpuMat>> image_pyr = build_pyramid(image_initial, octave_num, layer_num, sigma);
    std::vector<std::vector<cv::cuda::GpuMat>> image_dog = build_pyramid(image_pyr, sigma);

    // Find points
    unsigned int num_points = 0;
    for (size_t octave_id = 0; octave_id < image_dog.size(); octave_id++) {
        double scale = std::pow(2.0, octave_id);
        for (size_t layer_id = 0; layer_id < image_dog[octave_id].size() - 2; layer_id++) {
            num_points = cpu_find_points(
                image_dog[octave_id][layer_id + 0], image_dog[octave_id][layer_id + 1], image_dog[octave_id][layer_id + 2],
                m_sift, m_threshold, num_points, m_max_points, 16.0f, scale, 1.0f / layer_num
            );
        }
    }

    // Quit if no points were found
    if (num_points <= 0)
        return;

    // Get orientations
    cpu_compute_orientations(initial_image, m_sift, num_points, m_max_points);

    // Descriptors
    cpu_extract_sift_descriptors(initial_image, m_sift, m_desc, num_points, m_max_points);

    // Download points
    cv::Mat point_mat;
    m_sift.download(point_mat);

    // Convert points
    const float* ptr_xpos = point_mat.ptr<float>(CUDASIFT_POINT_XPOS);
    const float* ptr_ypos = point_mat.ptr<float>(CUDASIFT_POINT_YPOS);
    const float* ptr_scale = point_mat.ptr<float>(CUDASIFT_POINT_SCALE);
    const float* ptr_orientation = point_mat.ptr<float>(CUDASIFT_POINT_ORIENTATION);
    const float* ptr_score = point_mat.ptr<float>(CUDASIFT_POINT_SCORE);
    for (size_t i = 0; i < num_points; i++) {
        cv::KeyPoint kp;
        kp.pt.x = ptr_xpos[i];
        kp.pt.y = ptr_ypos[i];
        kp.size = ptr_scale[i];
        kp.angle = ptr_orientation[i];
        kp.response = ptr_score[i];
        keypoints.push_back(kp);
    }
}

cv::cuda::GpuMat s_sift::build_initial(cv::cuda::GpuMat image, double sigma)
{
    // Build filter
    static const float SIFT_INIT_SIGMA = 0.5f;
    float initial_sigma = sqrtf(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
    cv::Ptr<cv::cuda::Filter> initial_filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(), initial_sigma, initial_sigma);

    // Apply filter
    cv::cuda::GpuMat initial_image;
    initial_filter->apply(image, initial_image);
    return initial_image;
}

std::vector<std::vector<cv::cuda::GpuMat>> s_sift::build_pyramid(cv::cuda::GpuMat image, int octave_num, int layer_num, double sigma)
{
    // Layer blur sigmas
    std::vector<double> layer_sigma = { sigma };
    double k = std::pow(2.0, 1.0 / layer_num);
    for (int i = 1; i < layer_num + 3; i++)
    {
        double sig_prev = std::pow(k, static_cast<double>(i - 1)) * sigma;
        double sig_total = sig_prev * k;
        layer_sigma.push_back(std::sqrt(sig_total * sig_total - sig_prev * sig_prev));
    }

    // Octaves consisting of layers
    std::vector<std::vector<cv::cuda::GpuMat>> octave_pyr;
    for (int octave_id = 0; octave_id < octave_num; octave_id++) {
        // Set of images in a single octave - we refer to those images as layers
        std::vector<cv::cuda::GpuMat> layer_pyr;

        // First layer image
        if (octave_id == 0) {
            // First image
            layer_pyr.push_back(image);
        } else {
            // Downscale from previous octave
            cv::cuda::GpuMat octave_image;
            cv::cuda::resize(octave_pyr.back().back(), octave_image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_NEAREST);
            layer_pyr.push_back(octave_image);
        }

        // Rest of the layer images
        for (int layer_id = 1; layer_id < layer_num + 3; layer_id++) {
            // Create filter
            double sigma = sqrt((scale * scale) - (initial_blur * initial_blur));
            cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(), layer_sigma[layer_id], layer_sigma[layer_id]);

            // Apply filter
            cv::cuda::GpuMat layer_image;
            filter->apply(layer_pyr.back(), layer_image);
            layer_pyr.push_back(layer_image);
        }

        // Save the generated octave
        octave_pyr.push_back(layer_pyr);
    }

    // Success
    return octave_pyr;
}

std::vector<std::vector<cv::cuda::GpuMat>> s_sift::build_dog(std::vector<std::vector<cv::cuda::GpuMat>>& images)
{
    // Iterate over octaves
    std::vector<std::vector<cv::cuda::GpuMat>> octave_pyr;
    for (size_t octave_id = 0; octave_id < images.size(); octave_id++) {

        // Iterate over layers
        std::vector<cv::cuda::GpuMat> layer_pyr;
        for (size_t layer_id = 0; layer_id < images[octave_id].size() - 1; layer_id++) {
            cv::cuda::GpuMat image_dog;
            cv::cuda::subtract(images[octave_id][layer_id + 0], images[octave_id][layer_id + 1], image_dog, cv::noArray(), -1);
            layer_pyr.push_back(image_dog);
        }
        octave_pyr.push_back(layer_pyr);
    }

    // Success
    return octave_pyr;
}

/*
void s_sift::extract_octaves(cv::cuda::GpuMat image)
{
    double initial_blur = m_initial_blur;
    double subsampling = m_subsampling;
    for (size_t octave = 0; octave < m_num_octaves; octave++) {
        // Downscaling
        if (octave > 0) {
            cv::cuda::GpuMat subimage;
            cv::cuda::resize(image, subimage, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_NEAREST);
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
*/

};

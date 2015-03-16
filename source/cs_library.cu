/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

// Internal
#include "cs_device.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

void extract_sift(
    cv::cuda::GpuMat image, int num_octaves,
    double initial_blur, float thresh, float lowest_scale, float subsampling
) {
    // Parameters
    static const size_t num_scales = 5;

    // CUDA
    cv::cuda::Stream stream = cv::cuda::Stream::Null();

    // Octaves
    if (num_octaves > 1) {
        cv::cuda::GpuMat subimage;
        cv::cuda::resize(image, subimage, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR, stream);

        float total_blur = sqrt(initial_blur * initial_blur + 0.5f * 0.5f) / 2.0f;
        extract_sift(subimage, num_octaves - 1, total_blur, thresh, lowest_scale, subsampling * 2.0f);
    }

    // Scales
    if (lowest_scale < subsampling * 2.0f) {
        // Gaussian filtering
        std::vector<cv::cuda::GpuMat> blur_images;
        double scale_difference = pow(2.0f, 1.0f / num_scales);
        double scale = pow(2.0, -1.0 / num_scales);
        for (size_t i = 0; i < num_scales + 3; i++) {
            // Create filter
            double sigma = sqrt((scale * scale) - (initial_blur * initial_blur));
            cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(8, 8), sigma, sigma);

            // Filter
            cv::cuda::GpuMat blur_image;
            filter->apply(image, blur_image, stream);
            blur_images.push_back(blur_image);

            // Next scale
            scale *= scale_difference;
        }

        // Subtract gaussian images (DoG)
        std::vector<cv::cuda::GpuMat> diff_images;
        for (size_t i = 0; i < blur_images.size() - 1; i++) {
            cv::cuda::GpuMat diff_image;
            cv::cuda::subtract(blur_images[i], blur_images[i + 1], diff_image, cv::noArray(), -1, stream);
        }

        // Point finding
    }
}

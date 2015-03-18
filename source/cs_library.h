/*
 * CudaSift library
 *
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_LIBRARY
#define H_CS_LIBRARY

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace cudasift
{
    struct s_sift
    {
        public:
            // Constructor
            s_sift();

            // Detector
            void detect(cv::cuda::GpuMat image, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& keypoints2);

            // Descriptors
            cv::cuda::GpuMat descriptors() { return m_desc; }

        private:
            /*
            void extract_octaves(cv::cuda::GpuMat image);
            void extract_scales(cv::cuda::GpuMat image, double initial_blur, double subsampling);
            */
            cv::cuda::GpuMat build_initial(cv::cuda::GpuMat image, double sigma);
            std::vector< std::vector<cv::cuda::GpuMat> > build_pyramid(cv::cuda::GpuMat image, int octave_num, int layer_num, double sigma);
            std::vector< std::vector<cv::cuda::GpuMat> > build_dog(std::vector< std::vector<cv::cuda::GpuMat> >& images);

            // Points and descriptors
            unsigned int m_max_points;
            cv::cuda::GpuMat m_sift;
            cv::cuda::GpuMat m_desc;
    };
};

#endif

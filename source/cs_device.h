/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_DEVICE
#define H_CS_DEVICE

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#define WARP_SIZE      16
#define NUM_SCALES      5

#define MINMAX_W      126
#define MINMAX_S       64
#define MINMAX_H        8
#define NUMDESCBUFS     4

namespace cudasift
{
    enum {
        CUDASIFT_POINT_XPOS = 0,
        CUDASIFT_POINT_YPOS, // 1
        CUDASIFT_POINT_SCALE, // 2
        CUDASIFT_POINT_SHARPNESS, // 3
        CUDASIFT_POINT_EDGENESS,  // 4
        CUDASIFT_POINT_ORIENTATION, // 5
        CUDASIFT_POINT_SCORE, // 6
        CUDASIFT_POINT_SIZE // structure size
    };

    cudaError set_threshold(float a, float b);
    cudaError set_scales(float* scales, size_t size);
    cudaError set_factor(float factor);
    cudaError set_edge_limit(float edge_limit);
    cudaError set_max_points(int max_points);
    cudaError set_point_counter(unsigned int points);
    cudaError get_point_counter(unsigned int& points);

    void cpu_extract_sift_descriptors(
        cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, cv::cuda::GpuMat& desc,
        int numPts, int maxPts
    );
    __global__ void gpu_extract_sift_descriptors(float *g_Data, float *d_sift, float *d_desc, int maxPts);

    unsigned int cpu_find_points(
        cv::cuda::GpuMat& data1, cv::cuda::GpuMat& data2, cv::cuda::GpuMat& data3, cv::cuda::GpuMat& sift,
        float thresh, int numPts, int maxPts, float edgeLimit, float scale, float factor
    );
    __global__ void gpu_find_points(
        float *d_Data1, float *d_Data2, float *d_Data3, float *d_Sift,
        int width, int pitch, int height
    );

    void cpu_compute_orientations(cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, int numPts, int maxPts);
    __global__ void gpu_compute_orientations(float *g_Data, float *d_Sift, int maxPts, int w, int h);
};

#endif

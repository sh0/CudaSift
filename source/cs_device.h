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

#define SUBTRACT_W     32
#define SUBTRACT_H     16
#define SUBTRACTM_W    32
#define SUBTRACTM_H     1

#define SCALEDOWN_W   160
#define SCALEDOWN_H    16

#define MINMAX_W      126
#define MINMAX_S       64
#define MINMAX_H        8
#define NUMDESCBUFS     4

#define CONVROW_W     160
#define CONVCOL_W      32
#define CONVCOL_H      40
#define CONVCOL_S       8

namespace cudasift
{
    cudaError set_threshold(float a, float b);
    cudaError set_scales(float* scales, size_t size);
    cudaError set_factor(float factor);
    cudaError set_edge_limit(float edge_limit);
    cudaError set_max_points(int max_points);
    cudaError set_point_counter(unsigned int points);
    cudaError get_point_counter(unsigned int& points);
    cudaError set_kernel(float* kernel, size_t size);

    void cpu_scale_down(cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src, float variance);
    __global__ void gpu_scale_down(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch);

    void cpu_subtract(cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src_a, cv::cuda::GpuMat& src_b);
    __global__ void gpu_subtract(float *d_Result, float *d_Data1, float *d_Data2, int width, int pitch, int height);

    void cpu_extract_sift_descriptors(
        cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, cv::cuda::GpuMat& desc,
        int numPts, int maxPts
    );
    __global__ void gpu_extract_sift_descriptors(float *g_Data, float *d_sift, float *d_desc, int maxPts);

    unsigned int cpu_find_points(
        cv::cuda::GpuMat& data1, cv::cuda::GpuMat& data2, cv::cuda::GpuMat& data3, cv::cuda::GpuMat& sift,
        float thresh, int maxPts, float edgeLimit, float scale, float factor
    );
    __global__ void gpu_find_points(
        float *d_Data1, float *d_Data2, float *d_Data3, float *d_Sift,
        int width, int pitch, int height
    );

    void cpu_compute_orientations(cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, int numPts, int maxPts);
    __global__ void gpu_compute_orientations(float *g_Data, float *d_Sift, int maxPts, int w, int h);

    void cpu_subtract_multi(cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src);
    __global__ void gpu_subtract_multi(float *d_Result, float *d_Data, int width, int pitch, int height);

    unsigned int cpu_find_points_multi(cv::cuda::GpuMat& src, cv::cuda::GpuMat& sift, float thresh, int maxPts, float edgeLimit, float scale, float factor, float lowestScale);
    __global__ void gpu_find_points_multi(float *d_Data0, float *d_Sift, int width, int pitch, int height, int nScales);

    void cpu_lowpass(cv::cuda::GpuMat& dst, cv::cuda::GpuMat& src, float baseBlur, float diffScale, float initBlur);
    __global__ void gpu_lowpass_row(float *d_Result, float *d_Data, int width, int pitch, int height);
    __global__ void gpu_lowpass_col(float *d_Result, float *d_Data, int width, int pitch, int height);
};

#endif

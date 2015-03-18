/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

// Internal
#include "cs_device.h"

// C++
#include <cassert>

// CUDA error checking macro
static void cuda_safecall(cudaError ret, const char* file, const char* func, const int line)
{
    if (ret != cudaSuccess) {
        printf("[cudasift] CUDA error: %s! (%s:%s:%d)\n", cudaGetErrorString(ret), file, func, line);
    }
}
#define CUDA_SAFECALL(ret) cuda_safecall(ret, __FILE__, __func__, __LINE__)

inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
inline int iDivDown(int a, int b) { return a / b; }
inline int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }
inline int iAlignDown(int a, int b) { return a - a % b; }

namespace cudasift
{

// FindPoints, FindPointsMulti
__constant__ float d_Threshold[2];
__constant__ float d_Scales[8];
__constant__ float d_Factor;
__constant__ float d_EdgeLimit;
__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[1];

// ExtractSiftDescriptors
texture<float, 2, cudaReadModeElementType> tex;

// Set and get functions
cudaError set_threshold(float a, float b)
{
    float r[] = { a, b };
    return cudaMemcpyToSymbol(d_Threshold, &r, 2 * sizeof(float));
}

cudaError set_scales(float* scales, size_t size)
{
    assert(size <= 8);
    return cudaMemcpyToSymbol(d_Scales, scales, size * sizeof(float));
}

cudaError set_factor(float factor)
{
    return cudaMemcpyToSymbol(d_Factor, &factor, sizeof(float));
}

cudaError set_edge_limit(float edge_limit)
{
    return cudaMemcpyToSymbol(d_EdgeLimit, &edge_limit, sizeof(float));
}

cudaError set_max_points(int max_points)
{
    return cudaMemcpyToSymbol(d_MaxNumPoints, &max_points, sizeof(int));
}

cudaError set_point_counter(unsigned int points)
{
    return cudaMemcpyToSymbol(d_PointCounter, &points, sizeof(unsigned int));
}

cudaError get_point_counter(unsigned int& points)
{
    return cudaMemcpyFromSymbol(&points, d_PointCounter, sizeof(unsigned int));
}

#define SIFT_EXTREMA_W 32

struct s_extrema
{
    static const int m_border = 5;
    static const int m_block_w = 32;
    static const int m_block_h = 32;

    unsigned int* m_counter;
    int m_width;
    int m_height;
    cv::cuda::PtrStepf m_data[3];
    cv::cuda::PtrStepf m_sift;

    float m_threshold_contrast;

    __global__ void execute();
};

__global__ void s_extrema::execute()
{
    // Shared block memory
    __shared__ float shared_data[3][3][m_block_w];
    __shared__ float shared_min[2][3][m_block_w];
    __shared__ float shared_max[2][3][m_block_w];

    // Coordinates
    int x = m_block_w * blockIdx.x + threadIdx.x;
    int y_min = m_block_h * blockIdx.y + threadIdx.y - 1;
    int y_max = min(y_min + m_block_h + 1, m_height - m_border + 1);
    if (x >= m_width - m_border || y >= m_height - m_border)
        return;

    // Loop in Y direction
    for (int y = y_min; y < y_max; y++) {

    }
}

unsigned int cpu_find_points2(
    cv::cuda::GpuMat& data1, cv::cuda::GpuMat& data2, cv::cuda::GpuMat& data3,
    cv::cuda::GpuMat& sift, unsigned int num_points, unsigned int max_points,
    float threshold_contrast, float threshold_edge, float scale
) {
    // Structure
    s_extrema extrema;
    extrema.m_counter = NULL;
    extrema.m_data[0] = data1;
    extrema.m_data[1] = data2;
    extrema.m_data[2] = data3;
    extrema.m_sift = sift;

    extrema.m_threshold_contrast = threshold_contrast;

    // Counter
    CUDA_SAFECALL(cudaMalloc(&extrema.m_counter, sizeof(unsigned int)));
    CUDA_SAFECALL(cudaMemcpy(extrema.m_counter, &num_points, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Execute
    dim3 blocks(iDivUp(data1.cols, extrema.m_block_w), iDivUp(data1.rows, extrema.m_block_h));
    dim3 threads(extrema.m_block_w, 1);
    extrema.execute<<<blocks, threads>>>();
    CUDA_SAFECALL(cudaGetLastError());
    CUDA_SAFECALL(cudaThreadSynchronize());

    // Counter
    CUDA_SAFECALL(cudaMemcpy(&num_points, extrema.m_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_SAFECALL(cudaFree(extrema.m_counter));

    // Success
    return num_points;
}

unsigned int cpu_find_points(
    cv::cuda::GpuMat& data1, cv::cuda::GpuMat& data2, cv::cuda::GpuMat& data3, cv::cuda::GpuMat& sift,
    float thresh, int numPts, int maxPts, float edgeLimit, float scale, float factor
) {
    assert(data1.cols > 0 && data1.rows > 0 && data1.type() == CV_32FC1);
    assert(data2.cols > 0 && data2.rows > 0 && data2.type() == CV_32FC1);
    assert(data3.cols > 0 && data3.rows > 0 && data3.type() == CV_32FC1);

    cv::cuda::PtrStepSzf gpu_data1 = data1;
    cv::cuda::PtrStepSzf gpu_data2 = data2;
    cv::cuda::PtrStepSzf gpu_data3 = data3;
    cv::cuda::PtrStepSzf gpu_sift = sift;

    int w = gpu_data1.cols;
    int p = gpu_data1.step / sizeof(float);
    int h = gpu_data1.rows;

    CUDA_SAFECALL(set_threshold(thresh, -thresh));
    CUDA_SAFECALL(set_edge_limit(edgeLimit));
    CUDA_SAFECALL(set_scales(&scale, 1));
    CUDA_SAFECALL(set_factor(factor));
    CUDA_SAFECALL(set_max_points(maxPts));

    CUDA_SAFECALL(set_point_counter(numPts));

    dim3 blocks(iDivUp(w, MINMAX_W), iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2);
    gpu_find_points<<<blocks, threads>>>(gpu_data1.ptr(), gpu_data2.ptr(), gpu_data3.ptr(), gpu_sift.ptr(), w, p, h);
    CUDA_SAFECALL(cudaGetLastError());
    CUDA_SAFECALL(cudaThreadSynchronize());

    unsigned int total_points = 0;
    CUDA_SAFECALL(get_point_counter(total_points));
    return total_points;
}

__global__ void gpu_find_points(
    float* d_Data1, float* d_Data2, float* d_Data3, float* d_Sift,
    int width, int pitch, int height
) {
    #define MEMWID (MINMAX_W + 2)
    __shared__ float data1[3 * MEMWID], data2[3 * MEMWID], data3[3 * MEMWID];
    __shared__ float ymin1[MEMWID],   ymin2[MEMWID],   ymin3[MEMWID];
    __shared__ float ymax1[MEMWID],   ymax2[MEMWID],   ymax3[MEMWID];

    const int tx = threadIdx.x;
    const int minx = blockIdx.x * MINMAX_W;
    const int maxx = min(minx + MINMAX_W, width);
    const int xpos = minx + tx;

    int ptr0 = tx;
    int ptr1 = tx;
    int yq = 0;

    for (int y = 0; y < MINMAX_H + 2; y++) {

        int ypos = MINMAX_H * blockIdx.y + y - 1;
        int yptr = min(max(ypos, 0), height - 1) * pitch;
        int xposr = xpos - 1;
        int ptr2 = yq * MEMWID + tx;

        if (xposr < 0) {
            data1[ptr2] = 0;
            data2[ptr2] = 0;
            data3[ptr2] = 0;
        } else if (xposr >= width) {
            data1[ptr2] = 0;
            data2[ptr2] = 0;
            data3[ptr2] = 0;
        } else {
            data1[ptr2] = d_Data1[yptr + xposr];
            data2[ptr2] = d_Data2[yptr + xposr];
            data3[ptr2] = d_Data3[yptr + xposr];
        }

        //__syncthreads();
        if (y > 1) {
            float min1 = fminf(fminf(data1[ptr0], data1[ptr1]), data1[ptr2]);
            float min2 = fminf(fminf(data2[ptr0], data2[ptr1]), data2[ptr2]);
            float min3 = fminf(fminf(data3[ptr0], data3[ptr1]), data3[ptr2]);
            float max1 = fmaxf(fmaxf(data1[ptr0], data1[ptr1]), data1[ptr2]);
            float max2 = fmaxf(fmaxf(data2[ptr0], data2[ptr1]), data2[ptr2]);
            float max3 = fmaxf(fmaxf(data3[ptr0], data3[ptr1]), data3[ptr2]);
            ymin1[tx] = min1;
            ymin2[tx] = fminf(fminf(min1, min2), min3);
            ymin3[tx] = min3;
            ymax1[tx] = max1;
            ymax2[tx] = fmaxf(fmaxf(max1, max2), max3);
            ymax3[tx] = max3;
        }

        //__syncthreads();
        if (y > 1) {
            if (tx < MINMAX_W && xpos < maxx) {
                float minv = fminf(fminf(fminf(fminf(fminf(ymin2[tx], ymin2[tx + 2]), ymin1[tx + 1]), ymin3[tx + 1]), data2[ptr0 + 1]), data2[ptr2 + 1]);
                minv = fminf(minv, d_Threshold[1]);
                float maxv = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(ymax2[tx], ymax2[tx + 2]), ymax1[tx + 1]), ymax3[tx + 1]), data2[ptr0 + 1]), data2[ptr2 + 1]);
                maxv = fmaxf(maxv, d_Threshold[0]);
                float val = data2[ptr1 + 1];

                if (val < minv || val > maxv) {
                    float dxx = 2.0f * val - data2[ptr1 + 0] - data2[ptr1 + 2];
                    float dyy = 2.0f * val - data2[ptr0 + 1] - data2[ptr2 + 1];
                    float dxy = 0.25f * (data2[ptr2 + 2] + data2[ptr0 + 0] - data2[ptr0 + 2] - data2[ptr2 + 0]);
                    float tra = dxx + dyy;
                    float det = dxx * dyy - dxy * dxy;

                    if (tra * tra < d_EdgeLimit * det) {
                        float edge = __fdividef(tra * tra, det);
                        float dx = 0.5f * (data2[ptr1 + 2] - data2[ptr1 + 0]);
                        float dy = 0.5f * (data2[ptr2 + 1] - data2[ptr0 + 1]);
                        float ds = 0.5f * (data1[ptr1 + 1] - data3[ptr1 + 1]);
                        float dss = 2.0f * val - data3[ptr1 + 1] - data1[ptr1 + 1];
                        float dxs = 0.25f * (data3[ptr1 + 2] + data1[ptr1 + 0] - data1[ptr1 + 2] - data3[ptr1 + 0]);
                        float dys = 0.25f * (data3[ptr2 + 1] + data1[ptr0 + 1] - data3[ptr0 + 1] - data1[ptr2 + 1]);
                        float idxx = dyy * dss - dys * dys;
                        float idxy = dys * dxs - dxy * dss;
                        float idxs = dxy * dys - dyy * dxs;
                        float idyy = dxx * dss - dxs * dxs;
                        float idys = dxy * dxs - dxx * dys;
                        float idss = dxx * dyy - dxy * dxy;
                        float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
                        float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
                        float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
                        float pds = idet * (idxs * dx + idys * dy + idss * ds);

                        if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f) {
                            pdx = __fdividef(dx, dxx);
                            pdy = __fdividef(dy, dyy);
                            pds = __fdividef(ds, dss);
                        }

                        float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
                        int maxPts = d_MaxNumPoints;
                        unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
                        idx = (idx >= maxPts ? maxPts - 1 : idx);
                        d_Sift[idx + CUDASIFT_POINT_XPOS * maxPts] = xpos + pdx;
                        d_Sift[idx + CUDASIFT_POINT_YPOS * maxPts] = ypos - 1 + pdy;
                        d_Sift[idx + CUDASIFT_POINT_SCALE * maxPts] = d_Scales[0] * exp2f(pds * d_Factor);
                        d_Sift[idx + CUDASIFT_POINT_SHARPNESS * maxPts] = val + dval;
                        d_Sift[idx + CUDASIFT_POINT_EDGENESS * maxPts] = edge;
                        //printf("idx: %d %.1f %.1f %.2f\n", idx, d_Sift[idx + 0*maxPts], d_Sift[idx + 1*maxPts], edge);
                    }
                }
            }
        }

        __syncthreads();

        ptr0 = ptr1;
        ptr1 = ptr2;
        yq = (yq < 2 ? yq + 1 : 0);
    }
}

void cpu_compute_orientations(cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, int numPts, int maxPts)
{
    cv::cuda::PtrStepSzf gpu_image = image;
    cv::cuda::PtrStepSzf gpu_sift = sift;

    int p = gpu_image.step / sizeof(float);
    int h = gpu_image.rows;

    dim3 blocks(numPts);
    dim3 threads(32);
    gpu_compute_orientations<<<blocks, threads>>>(gpu_image.ptr(), gpu_sift.ptr(), maxPts, p, h);
    CUDA_SAFECALL(cudaGetLastError());
    CUDA_SAFECALL(cudaThreadSynchronize());
}

__global__ void gpu_compute_orientations(float* g_Data, float* d_Sift, int maxPts, int w, int h)
{
    __shared__ float data[16 * 15];
    __shared__ float hist[32 * 13];
    __shared__ float gauss[16];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    for (int i = 0; i < 13; i++)
        hist[i * 32 + tx] = 0.0f;

    __syncthreads();
    float i2sigma2 = -1.0f / (2.0f * 3.0f * 3.0f);

    if (tx < 15)
        gauss[tx] = exp(i2sigma2 * (tx - 7) * (tx - 7));

    int xp = (int)(d_Sift[bx + CUDASIFT_POINT_XPOS * maxPts] - 6.5f);
    int yp = (int)(d_Sift[bx + CUDASIFT_POINT_YPOS * maxPts] - 6.5f);
    int px = xp & 15;
    int x = tx - px;

    for (int y = 0; y < 15; y++) {
        int memPos = 16 * y + x;
        int xi = xp + x;
        int yi = yp + y;

        if (xi < 0)
            xi = 0;

        if (xi >= w)
            xi = w - 1;

        if (yi < 0)
            yi = 0;

        if (yi >= h)
            yi = h - 1;

        if (x >= 0 && x < 15)
            data[memPos] = g_Data[yi * w + xi];
    }

    __syncthreads();
    for (int y = 1; y < 14; y++) {
        int memPos = 16 * y + x;

        if (x >= 1 && x < 14) {
            float dy = data[memPos + 16] - data[memPos - 16];
            float dx = data[memPos + 1]  - data[memPos - 1];
            int bin = 16.0f * atan2f(dy, dx) / 3.1416f + 16.5f;

            if (bin == 32)
                bin = 0;

            float grad = sqrtf(dx * dx + dy * dy);
            hist[32 * (x - 1) + bin] += grad * gauss[x] * gauss[y];
        }
    }

    __syncthreads();
    for (int y = 0; y < 5; y++)
        hist[y * 32 + tx] += hist[(y + 8) * 32 + tx];

    __syncthreads();
    for (int y = 0; y < 4; y++)
        hist[y * 32 + tx] += hist[(y + 4) * 32 + tx];

    __syncthreads();
    for (int y = 0; y < 2; y++)
        hist[y * 32 + tx] += hist[(y + 2) * 32 + tx];

    __syncthreads();
    hist[tx] += hist[32 + tx];

    __syncthreads();
    if (tx == 0)
        hist[32] = 6 * hist[0] + 4 * (hist[1] + hist[31]) + (hist[2] + hist[30]);
    if (tx == 1)
        hist[33] = 6 * hist[1] + 4 * (hist[2] + hist[0]) + (hist[3] + hist[31]);
    if (tx >= 2 && tx <= 29)
        hist[tx + 32] = 6 * hist[tx] + 4 * (hist[tx + 1] + hist[tx - 1]) + (hist[tx + 2] + hist[tx - 2]);
    if (tx == 30)
        hist[62] = 6 * hist[30] + 4 * (hist[31] + hist[29]) + (hist[0] + hist[28]);
    if (tx == 31)
        hist[63] = 6 * hist[31] + 4 * (hist[0] + hist[30]) + (hist[1] + hist[29]);

    __syncthreads();
    float v = hist[32 + tx];
    hist[tx] = (v > hist[32 + ((tx + 1) & 31)] && v >= hist[32 + ((tx + 31) & 31)] ? v : 0.0f);

    __syncthreads();
    if (tx == 0) {
        float maxval1 = 0.0;
        float maxval2 = 0.0;
        int i1 = -1;
        int i2 = -1;

        for (int i = 0; i < 32; i++) {
            float v = hist[i];

            if (v > maxval1) {
                maxval2 = maxval1;
                maxval1 = v;
                i2 = i1;
                i1 = i;
            } else if (v > maxval2) {
                maxval2 = v;
                i2 = i;
            }
        }

        float val1 = hist[32 + ((i1 + 1) & 31)];
        float val2 = hist[32 + ((i1 + 31) & 31)];
        float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
        d_Sift[bx + CUDASIFT_POINT_ORIENTATION * maxPts] = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);

        if (maxval2 < 0.8f * maxval1)
            i2 = -1;

        if (i2 >= 0) {
            float val1 = hist[32 + ((i2 + 1) & 31)];
            float val2 = hist[32 + ((i2 + 31) & 31)];
            float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
            d_Sift[bx + CUDASIFT_POINT_SCORE * maxPts] = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
        } else {
            d_Sift[bx + CUDASIFT_POINT_SCORE * maxPts] = i2;
        }
    }
}

void cpu_extract_sift_descriptors(
    cv::cuda::GpuMat& image, cv::cuda::GpuMat& sift, cv::cuda::GpuMat& desc,
    int numPts, int maxPts
) {
    // Pointers
    cv::cuda::PtrStepSzf gpu_image = image;
    cv::cuda::PtrStepSzf gpu_sift = sift;
    cv::cuda::PtrStepSzf gpu_desc = desc;

    // Bind texture
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
    size_t offset = 0;
    CUDA_SAFECALL(cudaBindTexture2D(&offset, tex, gpu_image.ptr(), tex.channelDesc, gpu_image.cols, gpu_image.rows, gpu_image.step));

    // Calculate descriptors
    dim3 blocks(numPts);
    dim3 threads(16);
    gpu_extract_sift_descriptors<<<blocks, threads>>>(gpu_image.ptr(), gpu_sift.ptr(), gpu_desc.ptr(), maxPts);
    CUDA_SAFECALL(cudaGetLastError());
    CUDA_SAFECALL(cudaThreadSynchronize());

    // Unbind texture
    CUDA_SAFECALL(cudaUnbindTexture(tex));
}

__global__ void gpu_extract_sift_descriptors(float* g_Data, float* d_sift, float* d_desc, int maxPts)
{
    __shared__ float buffer[NUMDESCBUFS * 128];
    __shared__ float gauss[16];
    __shared__ float gradients[256];
    __shared__ float angles[256];

    const int tx = threadIdx.x; // 0 -> 16
    const int bx = blockIdx.x;  // 0 -> numPts

    gauss[tx] = exp(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

    __syncthreads();

    float theta = 2.0f * 3.1415f / 360.0f * d_sift[CUDASIFT_POINT_ORIENTATION * maxPts + bx];
    float sina = sinf(theta);           // cosa -sina
    float cosa = cosf(theta);           // sina  cosa
    float scale = 12.0f / 16.0f * d_sift[CUDASIFT_POINT_SCALE * maxPts + bx];
    float ssina = scale * sina;
    float scosa = scale * cosa;

    // Compute angles and gradients
    float xpos = d_sift[CUDASIFT_POINT_XPOS * maxPts + bx] + (tx - 7.5f) * scosa + 7.5f * ssina;
    float ypos = d_sift[CUDASIFT_POINT_YPOS * maxPts + bx] + (tx - 7.5f) * ssina - 7.5f * scosa;

    for (int i = 0; i < 128 * NUMDESCBUFS / 16; i++)
        buffer[16 * i + tx] = 0.0f;

    for (int y = 0; y < 16; y++) {
        float dx = tex2D(tex, xpos + cosa, ypos + sina) - tex2D(tex, xpos - cosa, ypos - sina);
        float dy = tex2D(tex, xpos - sina, ypos + cosa) - tex2D(tex, xpos + sina, ypos - cosa);
        gradients[16 * y + tx] = gauss[y] * gauss[tx] * sqrtf(dx * dx + dy * dy);
        angles[16 * y + tx] = 4.0f / 3.1415f * atan2f(dy, dx) + 4.0f;
        xpos -= ssina;
        ypos += scosa;
    }

    __syncthreads();
    if (tx < NUMDESCBUFS) {
        for (int txi = tx; txi < 16; txi += NUMDESCBUFS) {
            int hori = (txi + 2) / 4 - 1;
            float horf = (txi - 1.5f) / 4.0f - hori;
            float ihorf = 1.0f - horf;
            int veri = -1;
            float verf = 1.0f - 1.5f / 4.0f;

            for (int y = 0; y < 16; y++) {
                int i = 16 * y + txi;
                float grad = gradients[i];
                float angf = angles[i];
                int angi = angf;
                int angp = (angi < 7 ? angi + 1 : 0);
                angf -= angi;
                float iangf = 1.0f - angf;
                float iverf = 1.0f - verf;
                int hist = 8 * (4 * veri + hori);
                //printf("%d\n", hist);
                int p1 = tx + NUMDESCBUFS * (angi + hist);
                int p2 = tx + NUMDESCBUFS * (angp + hist);

                if (txi >= 2) {
                    float grad1 = ihorf * grad;
                    if (y >= 2) {
                        float grad2 = iverf * grad1;
                        buffer[p1 + 0] += iangf * grad2;
                        buffer[p2 + 0] +=  angf * grad2;
                    }

                    if (y <= 14) {
                        float grad2 = verf * grad1;
                        buffer[p1 + 32 * NUMDESCBUFS] += iangf * grad2;
                        buffer[p2 + 32 * NUMDESCBUFS] +=  angf * grad2;
                    }
                }

                if (txi <= 14) {
                    float grad1 = horf * grad;
                    if (y >= 2) {
                        float grad2 = iverf * grad1;
                        buffer[p1 + 8 * NUMDESCBUFS] += iangf * grad2;
                        buffer[p2 + 8 * NUMDESCBUFS] +=  angf * grad2;
                    }

                    if (y <= 14) {
                        float grad2 = verf * grad1;
                        buffer[p1 + 40 * NUMDESCBUFS] += iangf * grad2;
                        buffer[p2 + 40 * NUMDESCBUFS] +=  angf * grad2;
                    }
                }

                verf += 0.25f;
                if (verf > 1.0f) {
                    verf -= 1.0f;
                    veri ++;
                }
            }
        }
    }

    __syncthreads();
    const int t2 = (tx & 14) * 8;
    const int tx2 = (tx & 1);

    for (int i = 0; i < 16; i++)
        buffer[NUMDESCBUFS * (i + t2) + tx2] += buffer[NUMDESCBUFS * (i + t2) + tx2 + 2];

    __syncthreads();
    const int t1 = tx * 8;
    const int bptr = NUMDESCBUFS * tx + 2;
    buffer[bptr] = 0.0f;

    for (int i = 0; i < 8; i++) {
        int p = NUMDESCBUFS * (i + t1);
        buffer[p] += buffer[p + 1];
        buffer[bptr] += buffer[p] * buffer[p];
    }

    __syncthreads();
    if (tx < 8)
        buffer[bptr] += buffer[bptr + 8 * NUMDESCBUFS];

    __syncthreads();
    if (tx < 4)
        buffer[bptr] += buffer[bptr + 4 * NUMDESCBUFS];

    __syncthreads();
    if (tx < 2)
        buffer[bptr] += buffer[bptr + 2 * NUMDESCBUFS];

    __syncthreads();
    float isum = 1.0f / sqrt(buffer[2] + buffer[NUMDESCBUFS + 2]);

    buffer[bptr] = 0.0f;
    for (int i = 0; i < 8; i++) {
        int p = NUMDESCBUFS * (i + t1);
        buffer[p] = isum * buffer[p];

        if (buffer[p] > 0.2f)
            buffer[p] = 0.2f;

        buffer[bptr] += buffer[p] * buffer[p];
    }

    __syncthreads();
    if (tx < 8)
        buffer[bptr] += buffer[bptr + 8 * NUMDESCBUFS];

    __syncthreads();
    if (tx < 4)
        buffer[bptr] += buffer[bptr + 4 * NUMDESCBUFS];

    __syncthreads();
    if (tx < 2)
        buffer[bptr] += buffer[bptr + 2 * NUMDESCBUFS];

    __syncthreads();
    isum = 1.0f / sqrt(buffer[2] + buffer[NUMDESCBUFS + 2]);
    for (int i = 0; i < 8; i++) {
        int p = NUMDESCBUFS * (i + t1);
        d_desc[128 * bx + (i + t1)] = isum * buffer[p];
    }
}

};

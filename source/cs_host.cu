/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#include "cs_host.h"
#include "cs_device.h"

// CUDA error checking macro
static void cuda_safecall(cudaError ret, const char* file, const char* func, const int line)
{
    if (ret != cudaSuccess) {
        printf("[cudasift] CUDA error: %s! (%s:%s:%d)\n", cudaGetErrorString(ret), file, func, line);
    }
}
#define CUDA_SAFECALL(ret) cuda_safecall(ret, __FILE__, __func__, __LINE__)

namespace cudasift
{

inline int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
inline int iDivDown(int a, int b) { return a/b; }
inline int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
inline int iAlignDown(int a, int b) { return a - a%b; }

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initial_blur, float thresh, float lowest_scale, float subsampling)
{
    if (numOctaves > 1) {
        cv::cuda::GpuMat subImg(cv::Size(img.cols / 2, img.rows / 2), img.type());
        cpu_scale_down(subImg, img, 0.5f);

        float total_blur = sqrt(initial_blur * initial_blur + 0.5f * 0.5f) / 2.0f;
        ExtractSift(siftData, subImg, numOctaves - 1, total_blur, thresh, lowest_scale, subsampling * 2.0f);
    }

    if (lowest_scale < subsampling * 2.0f)
        ExtractSiftOctave(siftData, img, initial_blur, thresh, lowest_scale, subsampling);
}

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling)
{
    const int maxPts = iAlignUp(4096, 128);
    const int nb = NUM_SCALES + 3;
    const int nd = NUM_SCALES + 3;
    const double baseBlur = pow(2.0, -1.0/NUM_SCALES);
    int w = img.width;
    int h = img.height;
    CudaImage blurImg[nb];
    CudaImage diffImg[nd];
    CudaImage tempImg;
    CudaImage sift; // { xpos, ypos, scale, strength, edge, orient1, orient2 };
    CudaImage desc;

    float *memory = NULL;
    int p = iAlignUp(w, 128);
    int allocSize = (nb+nd+1)*p*h + maxPts*7 +  128*maxPts;
    CUDA_SAFECALL(cudaMalloc((void **)&memory, sizeof(float)*allocSize));
    for (int i=0;i<nb;i++)
        blurImg[i].Allocate(w, h, p, false, memory + i*p*h);
    for (int i=0;i<nb-1;i++)
        diffImg[i].Allocate(w, h, p, false, memory + (nb+i)*p*h);
    tempImg.Allocate(w, h, p, false, memory + (nb+nd)*p*h);

    sift.Allocate(maxPts, 7, maxPts, false, memory + (nb+nd+1)*p*h);
    desc.Allocate(128, maxPts, 128, false, memory + (nb+nd+1)*p*h + maxPts*7);

    float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
    cpu_lowpass(blurImg, img, diffImg, baseBlur, diffScale, initBlur);

    cpu_subtract_multi(diffImg, blurImg);

    double sigma = baseBlur*diffScale;
    unsigned int totPts = cpu_find_points_multi(diffImg, sift, thresh, maxPts, 16.0f, sigma, 1.0f/NUM_SCALES, lowestScale/subsampling);

    totPts = (totPts>=maxPts ? maxPts-1 : totPts);
    if (totPts>0) {
        cpu_compute_orientations(img, sift, totPts, maxPts);
        SecondOrientations(sift, &totPts, maxPts);
        cpu_extract_sift_descriptors(img, sift, desc, totPts, maxPts);
        AddSiftData(siftData, sift.d_data, desc.d_data, totPts, maxPts, subsampling);
    }
    CUDA_SAFECALL(cudaThreadSynchronize());
    CUDA_SAFECALL(cudaFree(memory));
}

void AddSiftData(SiftData &data, float *d_sift, float *d_desc, int numPts, int maxPts, float subsampling)
{
    int newNum = data.numPts + numPts;
    if (data.maxPts < newNum) {
        int newMaxNum = 2*data.maxPts;
        while (newNum>newMaxNum)
            newMaxNum *= 2;
        if (data.h_data!=NULL) {
            SiftPoint *h_data = (SiftPoint *)malloc(sizeof(SiftPoint)*newMaxNum);
            memcpy(h_data, data.h_data, sizeof(SiftPoint)*data.numPts);
            free(data.h_data);
            data.h_data = h_data;
        }
        if (data.d_data!=NULL) {
            SiftPoint *d_data = NULL;
            CUDA_SAFECALL(cudaMalloc((void**)&d_data, sizeof(SiftPoint)*newMaxNum));
            CUDA_SAFECALL(cudaMemcpy(d_data, data.d_data, sizeof(SiftPoint)*data.numPts, cudaMemcpyDeviceToDevice));
            CUDA_SAFECALL(cudaFree(data.d_data));
            data.d_data = d_data;
        }
        data.maxPts = newMaxNum;
    }
    int pitch = sizeof(SiftPoint);
    float *buffer = (float *)malloc(sizeof(float)*3*numPts);
    int bwidth = sizeof(float)*numPts;
    CUDA_SAFECALL(cudaMemcpy2D(buffer, bwidth, d_sift, sizeof(float)*maxPts, bwidth, 3, cudaMemcpyDeviceToHost));
    for (int i=0;i<3*numPts;i++)
        buffer[i] *= subsampling;
    CUDA_SAFECALL(cudaMemcpy2D(d_sift, sizeof(float)*maxPts, buffer, bwidth, bwidth, 3, cudaMemcpyHostToDevice));
    CUDA_SAFECALL(cudaThreadSynchronize());
    if (data.h_data!=NULL) {
        float *ptr = (float*)&data.h_data[data.numPts];
        for (int i=0;i<6;i++)
            CUDA_SAFECALL(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, numPts, cudaMemcpyDeviceToHost));
        CUDA_SAFECALL(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, sizeof(float)*128, numPts, cudaMemcpyDeviceToHost));
    }
    if (data.d_data!=NULL) {
        float *ptr = (float*)&data.d_data[data.numPts];
        for (int i=0;i<6;i++)
            CUDA_SAFECALL(cudaMemcpy2D(&ptr[i], pitch, &d_sift[i*maxPts], 4, 4, numPts, cudaMemcpyDeviceToDevice));
        CUDA_SAFECALL(cudaMemcpy2D(&ptr[16], pitch, d_desc, sizeof(float)*128, sizeof(float)*128, numPts, cudaMemcpyDeviceToDevice));
    }
    data.numPts = newNum;
    free(buffer);
}

void SecondOrientations(CudaImage &sift, int *initNumPts, int maxPts)
{
    int numPts = *initNumPts;
    int numPts2 = 2*numPts;
    float *d_sift = sift.d_data;
    int bw = sizeof(float)*numPts2;
    float *h_sift = (float *)malloc(7*bw);
    CUDA_SAFECALL(cudaMemcpy2D(h_sift, bw, d_sift, sizeof(float)*maxPts, sizeof(float)*numPts, 7, cudaMemcpyDeviceToHost));

    int num = numPts;
    for (int i=0;i<numPts;i++) {
        if (h_sift[6*numPts2+i]>=0.0f && num<maxPts) {
            for (int j=0;j<5;j++)
                h_sift[j*numPts2+num] = h_sift[j*numPts2+i];
            h_sift[5*numPts2+num] = h_sift[6*numPts2+i];
            h_sift[6*numPts2+num] = -1.0f;
            num ++;
        }
    }
    CUDA_SAFECALL(cudaMemcpy2D(&d_sift[numPts], sizeof(float)*maxPts, &h_sift[numPts], bw, sizeof(float)*(num-numPts), 7, cudaMemcpyHostToDevice));

    free(h_sift);
    *initNumPts = num;
}

};

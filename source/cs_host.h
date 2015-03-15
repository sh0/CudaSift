/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_HOST
#define H_CS_HOST

// Internal
#include "cs_data.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace cudasift
{
    void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling);
    void ExtractSiftOctave(SiftData &siftData, CudaImage &img, double initBlur, float thresh, float lowestScale, float subsampling);
    void AddSiftData(SiftData &data, float *d_sift, float *d_desc, int numPts, int maxPts, float subsampling);
    void SecondOrientations(CudaImage &sift, int *initNumPts, int maxPts);
};

#endif

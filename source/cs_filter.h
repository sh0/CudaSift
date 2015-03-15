/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_FILTER
#define H_CS_FILTER

#define CONVROW_W     160
#define CONVCOL_W      32
#define CONVCOL_H      40
#define CONVCOL_S       8

template<int RADIUS>
struct s_cudasift_filter
{
public:
    double SeparableFilter(CudaImage &dataA, CudaImage &dataB, CudaImage &temp, float *h_Kernel)
    {
        int width = dataA.width;
        int pitch = dataA.pitch;
        int height = dataA.height;

        float *d_DataA = dataA.d_data;
        float *d_DataB = dataB.d_data;

        float *d_Temp = temp.d_data;

        if (d_DataA==NULL || d_DataB==NULL || d_Temp==NULL) {
            printf("SeparableFilter: missing data\n");
            return 0.0;
        }

        const unsigned int kernelSize = (2 * RADIUS + 1) * sizeof(float);
        if (cudaMemcpyToSymbol(d_Kernel, h_Kernel, kernelSize) != cudaSuccess) {
            printf("CudaSift: SeparableFilter: cudaMemcpyToSymbol() failed!\n");
            return 0.0;
        }

        dim3 blockGridRows(
            iDivUp(width, CONVROW_W),
            height
        );
        dim3 threadBlockRows(CONVROW_W + 2 * RADIUS);
        ConvRowGPU<RADIUS><<<blockGridRows, threadBlockRows>>>(d_Temp, d_DataA, width, pitch, height);
        if (cudaGetLastError() != cudaSuccess || cudaThreadSynchronize() != cudaSuccess) {
            printf("CudaSift: SeparableFilter: ConvRowGPU() failed!\n");
            return 0.0;
        }

        dim3 blockGridColumns(
            iDivUp(width, CONVCOL_W),
            iDivUp(height, CONVCOL_H)
        );
        dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
        ConvColGPU<RADIUS><<<blockGridColumns, threadBlockColumns>>>(d_DataB, d_Temp, width, pitch, height);
        if (cudaGetLastError() != cudaSuccess || cudaThreadSynchronize() != cudaSuccess) {
            printf("CudaSift: SeparableFilter: ConvColGPU() failed!\n");
            return 0.0;
        }
    }

    double LowPass(CudaImage &dataB, CudaImage &dataA, CudaImage &temp, double var)
    {
        float kernel[2 * RADIUS+1];
        float kernelSum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            kernel[j + RADIUS] = (float) expf(-(double) j * j / 2.0 / var);
            kernelSum += kernel[j + RADIUS];
        }

        for (int j = -RADIUS; j <= RADIUS; j++)
            kernel[j + RADIUS] /= kernelSum;

        return SeparableFilter<RADIUS>(dataA, dataB, temp, kernel);
    }

private:
    __device__ __constant__ float d_Kernel[2 * RADIUS + 1];

    // Row convolution filter
    __global__ void ConvRowGPU(cv::cuda::PtrStepSz<float> d_Result, cv::cuda::PtrStepSz<float> d_Data)
    //float* d_Result, float* d_Data, int width, int pitch, int height
    {
        __shared__ float data[CONVROW_W + 2 * RADIUS];
        const int tx = threadIdx.x;
        const int minx = blockIdx.x * CONVROW_W;
        const int maxx = min(minx + CONVROW_W, d_Data.cols);
        float* yptr = d_Data.ptr(blockIdx.y);
        const int loadPos = minx + tx - RADIUS;
        const int writePos = minx + tx;

        if (loadPos < 0) {
            data[tx] = yptr[0];
        } else if (loadPos >= d_Data.cols) {
            data[tx] = yptr[d_Data.cols - 1];
        } else {
            data[tx] = yptr[loadPos];
        }

        __syncthreads();

        if (writePos < maxx && tx < CONVROW_W) {
            float sum = 0.0f;
            for (int i = 0; i <= (2 * RADIUS); i++)
                sum += data[tx + i] * d_Kernel[i];
            d_Result(blockIdx.y, writePos) = sum;
        }
    }

    // Column convolution filter
    __global__ void ConvColGPU(cv::cuda::PtrStepSz<float> d_Result, cv::cuda::PtrStepSz<float> d_Data)
    //float *d_Result, float *d_Data, int width, int pitch, int height
    {
        __shared__ float data[CONVCOL_W * (CONVCOL_H + 2 * RADIUS)];
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int miny = blockIdx.y * CONVCOL_H;
        const int maxy = min(miny + CONVCOL_H, d_Data.rows) - 1;
        const int totStart = miny - RADIUS;
        const int totEnd = maxy + RADIUS;
        const int colStart = blockIdx.x * CONVCOL_W + tx;
        const int colEnd = colStart + (d_Data.rows - 1) * pitch;
        const int smemStep = CONVCOL_W * CONVCOL_S;
        const int gmemStep = pitch * CONVCOL_S;

        if (colStart < d_Data.cols) {
            int smemPos = ty * CONVCOL_W + tx;
            int gmemPos = colStart + (totStart + ty) * pitch;
            for (int y = totStart + ty; y <= totEnd; y += blockDim.y) {
                if (y < 0) {
                    data[smemPos] = d_Data[colStart];
                } else if (y >= d_Data.rows) {
                    data[smemPos] = d_Data[colEnd];
                } else {
                    data[smemPos] = d_Data[gmemPos];
                }
                smemPos += smemStep;
                gmemPos += gmemStep;
            }
        }

        __syncthreads();

        if (colStart < d_Data.cols) {
            int smemPos = ty*CONVCOL_W + tx;
            int gmemPos = colStart + (miny + ty) * pitch;
            for (int y = miny + ty; y <= maxy; y += blockDim.y) {
                float sum = 0.0f;
                for (int i = 0; i <= 2 * RADIUS; i++)
                    sum += data[smemPos + i * CONVCOL_W] * d_Kernel[i];
                d_Result[gmemPos] = sum;
                smemPos += smemStep;
                gmemPos += gmemStep;
            }
        }
    }
};

#endif

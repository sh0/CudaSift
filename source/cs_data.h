/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_POINT
#define H_CS_POINT

// Internal
#include "cs_device.h"

namespace cudasift
{
    struct s_point
    {
        float xpos;
        float ypos;
        float scale;
        float sharpness;
        float edgeness;
        float orientation;
        float score;
        float ambiguity;

        int match;
        float match_xpos;
        float match_ypos;
        float match_error;
        float empty[4];

        float data[128];
    };

    struct s_key_points
    {
        public:
            // Constructor and destructor
            s_key_points(unsigned int max_points)
            {
                unsigned int alignment = 128 / sizeof(float);
                if (max_points % alignment)
                    max_points = max_points - (max_points % alignment) + alignment;

                m_device.num_points = 0;
                m_device.max_points = max_points;
                CUDA_SAFECALL(cudaMalloc(&m_device.data, max_points * CUDASIFT_POINT_SIZE * sizeof(float)));
            }

            ~s_key_points()
            {
                CUDA_SAFECALL(cudaFree(m_device.data));
            }

            // Upload and download
            void upload(std::vector<cv::Keypoint>& points)
            {
                assert(points.size() <= m_device.max_points)
                m_device.num_points = points.size();

                size_t stride = m_device.max_points * sizeof(float);
                float* data = new float[m_device.max_points * CUDASIFT_POINT_SIZE];
                memset(data, 0, m_device.max_points * CUDASIFT_POINT_SIZE * sizeof(float));
                for (size_t i = 0; i < points.size(); i++) {
                    data[CUDASIFT_POINT_XPOS * stride + i] = points[i].pt.x;
                    data[CUDASIFT_POINT_YPOS * stride + i] = points[i].pt.y;
                    data[CUDASIFT_POINT_SCALE * stride + i] = points[i].size;
                    data[CUDASIFT_POINT_ORIENTATION * stride + i] = points[i].angle;
                    data[CUDASIFT_POINT_SCORE * stride + i] = points[i].response;
                }
                CUDA_SAFECALL(cudaMemcpy(m_device.data, data, m_device.max_points * CUDASIFT_POINT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
                delete[] data;
            }

            std::vector<cv::Keypoint> download()
            {
                std::vector<s_point> points(m_device.num_points);
                if (m_device.num_points > 0) {
                    size_t stride = m_device.max_points * sizeof(float);
                    float* data = new float[m_device.max_points * CUDASIFT_POINT_SIZE];
                    CUDA_SAFECALL(cudaMemcpy(data, m_device.data, m_device.max_points * CUDASIFT_POINT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < m_device.num_points; i++) {
                        points[i].pt.x = data[CUDASIFT_POINT_XPOS * stride + i];
                        points[i].pt.y = data[CUDASIFT_POINT_YPOS * stride + i];
                        points[i].size = data[CUDASIFT_POINT_SCALE * stride + i];
                        points[i].angle = data[CUDASIFT_POINT_ORIENTATION * stride + i];
                        points[i].response = data[CUDASIFT_POINT_SCORE * stride + i];
                    }
                    delete[] data;
                }
                return points;
            }

            // Device
            struct s_device {
                unsigned int num_points;
                unsigned int max_points;
                s_point* data;
            };
            s_device device() { return m_device; }

        private:
            // Device
            s_device m_device;
    };
};

#endif

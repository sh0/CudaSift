/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef H_CS_POINT
#define H_CS_POINT

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

    struct s_data
    {
        public:
            // Constructor and destructor
            s_data(unsigned int max_points)
            {
                m_device.num_points = 0;
                m_device.max_points = max_points;
                if (cudaMalloc(&m_device.data, max_points * sizeof(s_point)) != cudaSuccess)
                    throw std::runtime_error("cudasift::s_data: cudaMalloc() failed!");
            }

            ~s_data()
            {
                if (cudaFree(m_device.data) != cudaSuccess)
                    throw std::runtime_error("cudasift::~s_data: cudaFree() failed!");
            }

            // Upload and download
            void upload(std::vector<s_point>& points)
            {
                if (points.size() > m_device.max_points)
                    throw std::runtime_error("cudasift::upload: Too many points for upload!");
                m_device.num_points = points.size();
                if (cudaMemcpy(m_device.data, points.data(), m_device.num_points * sizeof(s_point), cudaMemcpyHostToDevice) != cudaSuccess)
                    throw std::runtime_error("cudasift::upload: cudaMemcpy() failed!");
            }

            std::vector<s_point> download()
            {
                std::vector<s_point> points(m_device.num_points);
                if (m_device.num_points > 0) {
                    if (cudaMemcpy(points.data(), m_device.data, m_device.num_points * sizeof(s_point), cudaMemcpyDeviceToHost) != cudaSuccess)
                        throw std::runtime_error("cudasift::download: cudaMemcpy() failed!");
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

/*
 * CudaSift library
 *
 * Copyright (C) 2007-2015 Marten Bjorkman <celle@nada.kth.se>
 * Copyright (C) 2015 Siim Meerits <siim@yutani.ee>
 */

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

#endif

== CudaSift library ==

This is the third version of a SIFT (Scale Invariant Feature Transform) implementation using CUDA for GPUs from NVidia. The first version is from 2007 and GPUs have evolved since then. This version is considerably faster than the previous versions and has been optimized for GeForce GTX 580. It should be reasonable fast also on more recent GPUs.

On a GTX 580 GPU the code takes about 7 ms on a 640x480 pixel image and 13 ms on a 1280x960 pixel image. The 2 ms and 1 ms respective of that is for data transfers inbetween GPU and CPU. If you keep data on the GPU things are thus somewhat faster. There is also code for brute-force matching of features and homography computation that takes about 7 ms and 4 ms for two sets of around 1500 SIFT features each.

The code is free to use for non-commercial applications. If you use the code for research, please refer to the following upcoming paper.

M. Bj�rkman, N. Bergstr�m and D. Kragic, "Detecting, segmenting and tracking unknown objects using multi-label MRF inference", CVIU (accepted for publication)

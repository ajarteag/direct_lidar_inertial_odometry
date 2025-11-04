#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H

#include <cuda.h>
#include <cuda_runtime_api.h>

void cu_dot_kernel_launch(
    Eigen::Matrix4f *T,
    Eigen::Vector4f *pt,
    Eigen::Vector4f *out
);

#endif

#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
// #include "dlio/dlio.h"

/*
void cu_dot_kernel_launch(
    Eigen::Matrix4f *T,
    Eigen::Vector4f *pt,
    Eigen::Vector4f *out
);
*/

void deskew_cuda(
    const Eigen::Matrix4f &baselink2lidar_T,
    const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &frames,
    std::vector<PointType, Eigen::aligned_allocator<PointType>> &points,
    const std::vector<pcl::index_t> &unique_time_indices);

#endif

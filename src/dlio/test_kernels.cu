#include <Eigen/Core>
#include "cuda_kernel.h"

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

// Device code
__global__ void cu_dot(Eigen::Matrix4f *T, Eigen::Vector4f *pt, Eigen::Vector4f *out)
{
    *out = (*T) * (*pt);
    return;
}

// Host code
void cu_dot_kernel_launch(
    Eigen::Matrix4f *T,
    Eigen::Vector4f *pt,
    Eigen::Vector4f *out
)
{
    Eigen::Matrix4f *T_cuda;
    Eigen::Vector4f *pt_cuda, *out_cuda;

    // TODO investigate cudaMallocManaged for shared memory
    cudaMalloc(&T_cuda, sizeof(Eigen::Matrix4f));
    cudaMalloc(&pt_cuda, sizeof(Eigen::Vector4f));
    cudaMalloc(&out_cuda, sizeof(Eigen::Vector4f));

    // Copy memory
    cudaMemcpy(T_cuda, T, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(pt_cuda, pt, sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);

    cu_dot<<<1, 1>>>(T_cuda, pt_cuda, out_cuda);

    // Copy output
    cudaMemcpy(out, out_cuda, sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);

    cudaFree(T_cuda);
    cudaFree(pt_cuda);
    cudaFree(out_cuda);
}

#include "dlio/dlio.h"
#include "cuda_kernel.h"

#include <stdexcept>
#include <cublas_v2.h>

inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

void deskew_cuda(
    const Eigen::Matrix4f &baselink2lidar_T,
    const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &frames,
    std::vector<PointType, Eigen::aligned_allocator<PointType>> &points,
    const std::vector<pcl::index_t> &unique_time_indices)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    int num_frames = frames.size();
    int num_points = points.size();

    float *baselink2lidar_T_cuda;   // baselink to lidar transform
    float **baselink2lidar_Ts_cuda; // pointer for batched multiplication
    float **frames_cuda;            // pointer to frame transforms
    float **frame2baselink_T_cuda;  // pointer for frame results
    float **points_cuda;            // pointer to points
    float **Ts_cuda;                // pointer to point transforms
    float **result;                 // pointer to results

    // Allocate memory
    cudaMallocManaged(&baselink2lidar_T_cuda, sizeof(float) * 16);
    cudaMallocManaged(&baselink2lidar_Ts_cuda, sizeof(float *) * num_frames);
    cudaMallocManaged(&frames_cuda, sizeof(float *) * num_frames);
    cudaMallocManaged(&frame2baselink_T_cuda, sizeof(float *) * num_frames);

    cudaMallocManaged(&points_cuda, sizeof(float *) * num_points);
    cudaMallocManaged(&Ts_cuda, sizeof(float *) * num_points);
    cudaMallocManaged(&result, sizeof(float *) * num_points);

    memcpy(baselink2lidar_T_cuda, baselink2lidar_T.data(), sizeof(float) * 16);
    for (int i = 0; i < num_frames; i++)
    {
        baselink2lidar_Ts_cuda[i] = baselink2lidar_T_cuda;
        float *frame_cuda;
        cudaMallocManaged(&frame_cuda, sizeof(float) * 16);
        memcpy(frame_cuda, frames[i].data(), sizeof(float) * 16);
        frames_cuda[i] = frame_cuda;

        float *result_cuda;
        cudaMallocManaged(&result_cuda, sizeof(float) * 16);
        frame2baselink_T_cuda[i] = result_cuda;
    }

    for (int i = 0; i < num_points; i++)
    {
        float *point_cuda;
        cudaMallocManaged(&point_cuda, sizeof(float) * 4);
        memcpy(point_cuda, points[i].data, sizeof(float) * 4);
        point_cuda[3] = 1.0f; // homogeneous coordinate
        points_cuda[i] = point_cuda;

        float *result_cuda;
        cudaMallocManaged(&result_cuda, sizeof(float) * 4);
        result[i] = result_cuda;
    }

    for (int i = 0; i < num_frames; i++)
    {
        for (int k = unique_time_indices[i]; k < unique_time_indices[i + 1]; k++)
        {
            Ts_cuda[k] = frame2baselink_T_cuda[i];
        }
    }

    // Initialize cuBLAS Handle
    cublasHandle_t handle;
    checkCublasStatus(cublasCreate(&handle));

    // Transform from lidar to baselink
    checkCublasStatus(
        cublasSgemmBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            4,
            4,
            4,
            &alpha,
            frames_cuda,
            4,
            baselink2lidar_Ts_cuda,
            4,
            &beta,
            frame2baselink_T_cuda,
            4,
            num_frames));
    cudaDeviceSynchronize();

    // Transform point to world coordinates
    checkCublasStatus(
        cublasSgemvBatched(
            handle,
            CUBLAS_OP_N,
            4,
            4,
            &alpha,
            Ts_cuda,
            4,
            points_cuda,
            1,
            &beta,
            result,
            1,
            num_points));
    cudaDeviceSynchronize();

    // Copy back to CPU
    for (int i = 0; i < num_points; i++)
    {
        memcpy(points[i].data, result[i], sizeof(float) * 3);
    }

    // Destroy cuBLAS Handle
    checkCublasStatus(cublasDestroy(handle));

    // Free memory
    cudaFree(baselink2lidar_T_cuda);
    cudaFree(baselink2lidar_Ts_cuda);

    for (int i = 0; i < num_frames; i++)
    {
        cudaFree(frames_cuda[i]);
        cudaFree(frame2baselink_T_cuda[i]);
    }
    cudaFree(frames_cuda);
    cudaFree(frame2baselink_T_cuda);

    for (int i = 0; i < num_points; i++)
    {
        cudaFree(points_cuda[i]);
        cudaFree(result[i]);
    }
    cudaFree(points_cuda);
    cudaFree(Ts_cuda);
    cudaFree(result);
}

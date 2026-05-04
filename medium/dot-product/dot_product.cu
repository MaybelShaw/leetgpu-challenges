#include <cuda_runtime.h>

__global__ void dot_product_kernel(const float *A, const float *B, float *result, int N)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sdata[1024];

    float sum = 0.0f;
    if (idx < N)
    {
        sum = A[idx] * B[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        atomicAdd(result, sdata[0]);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float *A, const float *B, float *result, int N)
{
    cudaMemset(result, 0, sizeof(float));
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>

__global__ void reduction_kernel(const float *input, float *output, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        atomicAdd(output, input[idx]);
    }
}

__global__ void reduction_acc_kernel(const float *input, float *output, int N)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sdata[256];
    float sum = 0.0f;
    if (idx < N)
    {
        sum = input[idx];
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
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    cudaMemset(output, 0, sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduction_acc_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void fp16_dot_product_kernel(const half *A, const half *B, float *temp, int N)
{
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    if (idx < N)
    {
        float a = __half2float(A[idx]);
        float b = __half2float(B[idx]);
        sum = a * b;
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
        atomicAdd(temp, sdata[0]);
    }
}

__global__ void convert_result_kernel(const float *temp, half *result)
{
    result[0] = __float2half(temp[0]);
}

// A, B, result are device pointers
extern "C" void solve(const half *A, const half *B, half *result, int N)
{
    int threadsPerBlock = 1024;
    int blockPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *temp;
    cudaMalloc(&temp, sizeof(float));
    cudaMemset(temp, 0, sizeof(float));
    fp16_dot_product_kernel<<<blockPerGrid, threadsPerBlock>>>(A, B, temp, N);
    convert_result_kernel<<<1, 1>>>(temp, result);

    cudaFree(temp);
    cudaDeviceSynchronize();
}

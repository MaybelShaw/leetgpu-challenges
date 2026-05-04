#include <cuda_runtime.h>

__global__ void sum_sq_kernel(const float *input, float *partial_sums, int N)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < N; i += stride)
    {
        float x = input[i];
        sum += x * x;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1)
    {
        if (tid < step)
        {
            sdata[tid] += sdata[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void final_sum_kernel(const float *partial_sums, float *total_sum, int num_blocks)
{

    __shared__ float sdata[1024];

    int tid = threadIdx.x;

    float local_sum = 0.0f;
    if (tid < num_blocks)
    {
        local_sum = partial_sums[tid];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1)
    {
        if (tid < step)
        {
            sdata[tid] += sdata[tid + step];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        total_sum[0] = sdata[0];
    }
}

__global__ void rms_norm_kernel(const float *input, float gamma, float beta, float *output, int N, float eps, const float *total_sum)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float scale = rsqrtf(total_sum[0] / (float)N + eps);
    for (int i = idx; i < N; i += stride)
    {
        output[i] = input[i] * scale * gamma + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float gamma, float beta, float *output, int N, float eps)
{
    int threadsPerBlock = 1024;
    int maxBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxBlock)
    {
        blocksPerGrid = maxBlock;
    }

    float *partial_sums;
    float *total_sum;

    cudaMalloc(&partial_sums, blocksPerGrid * sizeof(float));
    cudaMalloc(&total_sum, sizeof(float));
    cudaMemset(partial_sums, 0, blocksPerGrid * sizeof(float));
    cudaMemset(total_sum, 0, sizeof(float));

    sum_sq_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_sums, N);
    final_sum_kernel<<<1, 1024>>>(partial_sums, total_sum, blocksPerGrid);
    rms_norm_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, eps, total_sum);

    cudaFree(partial_sums);
    cudaFree(total_sum);

    cudaDeviceSynchronize();
}
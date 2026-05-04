#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_kernel(const float *input, float *output, int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x)
    {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
    {
        output[i] = __expf(input[i] - max_val);
        local_sum += output[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum = sdata[0];

    for (int i = tid; i < N; i += blockDim.x)
    {
        output[i] /= sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    int threadsPerBlock = 256;

    softmax_kernel<<<1, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float *input, float *output, int halfN)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < halfN)
    {
        float x = input[idx];
        float gate = input[idx + halfN];
        output[idx] = x * gate / (1.0f + __expf(-gate));
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}

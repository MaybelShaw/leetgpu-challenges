#include <cuda_runtime.h>

__global__ void geglu_kernel(const float *input, float *output, int halfN)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < halfN)
    {
        float x = input[idx];
        float gate = input[idx + halfN];
        float gelu = 0.5f * gate * (1.0f + tanhf(0.7978845608f * (gate + 0.044715f * gate * gate * gate)));
        output[idx] = x * gelu;
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}

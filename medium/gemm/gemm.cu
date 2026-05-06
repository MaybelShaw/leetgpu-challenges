#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void gemm_kernel(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            float a = __half2float(A[row * K + i]);
            float b = __half2float(B[i * N + col]);
            sum += a * b;
        }
        float c = __half2float(C[row * N + col]);
        C[row * N + col] = __float2half(alpha * sum + beta * c);
    }
}

__global__ void gemm_tiled_kernel(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    for (int tile_k = 0; tile_k < K; tile_k += TILE)
    {
        int a_col = tile_k + tx;
        int b_row = tile_k + ty;
        if (row < M && a_col < K)
        {
            As[ty][tx] = __half2float(A[row * K + a_col]);
        }
        else
        {
            As[ty][tx] = 0.0f;
        }
        if (b_row < K && col < N)
        {
            Bs[ty][tx] = __half2float(B[b_row * N + col]);
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < TILE; i++)
        {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        float old_c = __half2float(C[row * N + col]);
        float result = alpha * sum + beta * old_c;
        C[row * N + col] = __float2half(result);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gemm_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K,
    stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 用 tl.dot 实现分块矩阵乘法
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    z = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        a_ptrs = a + rows[:, None] * stride_am + offs_n[None, :] * stride_an
        x = tl.load(a_ptrs, mask=(rows[:, None] < M) & (offs_n[None, :] < N), other=0.0)

        b_ptrs = b + offs_n[:, None] * stride_bn + cols[None, :] * stride_bk
        y = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & (cols[None, :] < K), other=0.0)

        z += tl.dot(x, y)

    c_ptrs = c + rows[:, None] * stride_cm + cols[None, :] * stride_ck
    tl.store(c_ptrs, z, mask=(rows[:, None] < M) & (cols[None, :] < K))


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K,
        stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

import torch
import triton
import triton.language as tl


# @triton.jit
# def gemm_kernel(
#     a: torch.Tensor,
#     b: torch.Tensor,
#     c: torch.Tensor,
#     M: int,
#     N: int,
#     K: int,
#     alpha: float,
#     beta: float,
# ):
#     row = tl.program_id(axis=0)
#     col = tl.program_id(axis=1)

#     z = tl.zeros((), dtype=tl.float32)
#     for i in range(K):
#         x = tl.load(a + row * K + i).to(tl.float32)
#         y = tl.load(b + i * N + col).to(tl.float32)
#         z += x * y

#     tl.store(c + row * N + col, z)

# # a, b, c are tensors on the GPU
# def solve(
#     a: torch.Tensor,
#     b: torch.Tensor,
#     c: torch.Tensor,
#     M: int,
#     N: int,
#     K: int,
#     alpha: float,
#     beta: float,
# ):
#     grid = (M, N)
#     gemm_kernel[grid](a, b, c, M, N, K, alpha, beta)


@triton.jit
def gemm_tiled_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a + offsets_m[:, None] * K + (k + offsets_k[None, :])
        b_ptrs = b + (k + offsets_k[:, None]) * N + offsets_n[None, :]

        a_mask = (offsets_m[:, None] < M) & (k + offsets_k[None, :] < K)
        b_mask = (k + offsets_k[:, None] < K) & (offsets_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile)

    c_ptrs = c + offsets_m[:, None] * N + offsets_n[None, :]
    c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)

    old_c = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(c_ptrs, alpha * acc + beta * old_c, mask=c_mask)


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_tiled_kernel[grid](a, b, c, M, N, K, alpha, beta, BLOCK_M, BLOCK_N, BLOCK_K)

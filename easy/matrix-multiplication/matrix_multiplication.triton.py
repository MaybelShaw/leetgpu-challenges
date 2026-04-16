import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
):
    row = tl.program_id(axis=0)
    col = tl.program_id(axis=1)

    z = tl.zeros((), dtype=tl.float32)
    for i in range(N):
        x = tl.load(a + row * stride_am + i * stride_an)
        y = tl.load(b + i * stride_bn + col * stride_bk)
        z += x * y

    tl.store(c + row * stride_cm + col * stride_ck, z)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )


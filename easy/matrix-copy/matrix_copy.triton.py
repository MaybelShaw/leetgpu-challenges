import torch
import triton
import triton.language as tl


@triton.jit
def copy_matrix_kernel(a_ptr, b_ptr, total, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    x = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    tl.store(b_ptr + offsets, x, mask=mask)


# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    total = N * N
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    copy_matrix_kernel[grid](a, b, total, BLOCK_SIZE=BLOCK_SIZE)

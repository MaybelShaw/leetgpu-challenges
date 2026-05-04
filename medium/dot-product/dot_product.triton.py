import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_kernel(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(a + offsets, mask=mask, other=0.0)
    y = tl.load(b + offsets, mask=mask, other=0.0)

    tl.atomic_add(result, tl.sum(x * y))


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    result.zero_()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    dot_product_kernel[grid](a, b, result, n, BLOCK_SIZE=BLOCK_SIZE)

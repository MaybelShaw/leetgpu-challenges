import torch
import triton
import triton.language as tl


@triton.jit
def reduction_kernel(input: torch.Tensor, output: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0)
    tl.atomic_add(output, tl.sum(x))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.zero_()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    reduction_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

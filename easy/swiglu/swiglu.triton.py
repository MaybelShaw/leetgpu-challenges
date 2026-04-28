import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    halfN = N // 2
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < halfN

    x = tl.load(input + offsets, mask=mask, other=0.0)
    gate = tl.load(input + halfN + offsets, mask=mask, other=0.0)
    y = x * gate / (1.0 + tl.exp(-gate))

    tl.store(output + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

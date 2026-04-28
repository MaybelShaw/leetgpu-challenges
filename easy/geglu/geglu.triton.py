import torch
import triton
import triton.language as tl


@triton.jit
def geglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    halfN = N // 2
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < halfN

    x = tl.load(input + offsets, mask=mask, other=0.0)
    gate = tl.load(input + halfN + offsets, mask=mask, other=0.0)

    gelu = 0.5 * gate * (1 + tl.tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate)))
    y = x * gelu

    tl.store(output + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    geglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)

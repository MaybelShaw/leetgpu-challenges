import torch
import triton
import triton.language as tl


@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    if pid < N / 2:
        x = tl.load(input + pid)
        y = tl.load(input + N - pid - 1)
        tl.store(input + pid, y)
        tl.store(input + N - pid - 1, x)


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](input, N, BLOCK_SIZE)

import torch
import triton
import triton.language as tl


@triton.jit
def sum_sq_kernel(input: torch.Tensor, partial_sums: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0)
    tl.store(partial_sums + pid, tl.sum(x * x))


@triton.jit
def final_sum_kernel(partial_sums: torch.Tensor, total_sum: torch.Tensor, num_blocks: int, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks

    x = tl.load(partial_sums + offsets, mask=mask, other=0.0)
    tl.store(total_sum, tl.sum(x))


@triton.jit
def rms_norm_kernel(
    input: torch.Tensor,
    gamma: float,
    beta: float,
    output: torch.Tensor,
    N: int,
    eps: float,
    total_sum: torch.Tensor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0)
    sum = tl.load(total_sum)
    scale = 1 / tl.sqrt(sum / N + eps)

    tl.store(output + offsets, x * gamma * scale + beta, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)

    partial_sums = torch.zeros((num_blocks,), device=input.device)
    total_sum = torch.zeros((1,), device=input.device)
    sum_sq_kernel[grid](input, partial_sums, N, BLOCK_SIZE)
    final_sum_kernel[(1,)](partial_sums, total_sum, num_blocks, BLOCK_SIZE)
    rms_norm_kernel[grid](input, gamma, beta, output, N, eps, total_sum, BLOCK_SIZE)

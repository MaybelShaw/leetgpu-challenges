import torch
import triton
import triton.language as tl


@triton.jit
def fp16_dot_product_kernel(A: torch.Tensor, B: torch.Tensor, temp: torch.Tensor, N: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(A + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.atomic_add(temp, tl.sum(a * b))


@triton.jit
def convert_result_kernel(temp: torch.Tensor, result: torch.Tensor):
    x = tl.load(temp)
    tl.store(result, x)


# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    temp = torch.zeros((1,), device=A.device, dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fp16_dot_product_kernel[grid](A, B, temp, N, BLOCK_SIZE=BLOCK_SIZE)
    convert_result_kernel[(1,)](temp, result)

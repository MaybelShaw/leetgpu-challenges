import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input + offsets, mask=mask, other=0.0)
    x = x - tl.max(x, axis=0)
    exp = tl.exp(x)
    sum = tl.sum(exp, axis=0)

    y = exp / sum

    tl.store(output + offsets, y, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = tl.next_power_of(N)
    softmax_kernel[(1,)](input, output, N, BLOCK_SIZE)

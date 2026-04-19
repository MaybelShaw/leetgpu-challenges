import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(input, kernel, output, input_size, kernel_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    if pid < input_size - kernel_size + 1:
        z = tl.zeros((), dtype=tl.float32)
        for i in range(kernel_size):
            x = tl.load(input + pid + i)
            y = tl.load(kernel + i)
            z += x * y
        tl.store(output + pid, z)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)

    conv1d_kernel[grid](input, kernel, output, input_size, kernel_size, BLOCK_SIZE)
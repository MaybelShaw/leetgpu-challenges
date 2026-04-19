import torch


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_size: int,
    kernel_size: int,
):
    windows = input.unfold(0, kernel_size, 1)
    result = (windows * kernel).sum(dim=1)
    output.copy_(result)

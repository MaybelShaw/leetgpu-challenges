import torch


# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    input.copy_(input.flip(0))

import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output[:N].copy_(input[:N] * (1 / (1 + torch.exp(-input[:N]))))

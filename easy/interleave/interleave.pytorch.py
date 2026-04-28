import torch


# A, B, output are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    output[:2 * N:2].copy_(A[:N])
    output[1:2 * N:2].copy_(B[:N])

import torch


# A, B, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    result.copy_(torch.dot(A[:N].float(), B[:N].float()).half())

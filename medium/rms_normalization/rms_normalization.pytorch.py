import torch


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    eps: float,
):
    sum = torch.sum(input * input)
    scale = 1 / torch.sqrt(sum / N + eps)
    output.copy_(input * gamma * scale + beta)

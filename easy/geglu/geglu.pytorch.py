import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    halfN = N // 2
    x = input[:halfN]
    gate = input[halfN:N]
    gelu = 0.5 * gate * (1 + torch.tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate)))
    output[:halfN].copy_(x * gelu)

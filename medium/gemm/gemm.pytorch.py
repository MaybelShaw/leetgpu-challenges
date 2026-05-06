import torch


# A, B, C are tensors on the GPU
def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    C_fp32 = C.to(torch.float32)
    result = alpha * (A_fp32 @ B_fp32 )+ beta * C_fp32
    C.copy_(result.to(C.dtype))

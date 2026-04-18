import torch


# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    torch.sub(255, image, out=image)

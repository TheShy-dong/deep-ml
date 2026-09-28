import torch

def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    # Your code here
    length=x.shape[-1]
    value=x[:,:length//2]
    gate=x[:,length//2:]*torch.sigmoid(x[:,length//2:])
    return gate*value
    pass
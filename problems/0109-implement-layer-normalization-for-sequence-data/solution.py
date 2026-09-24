import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    average=X.mean(-1,keepdim=True)
    fenmu=torch.sqrt((X-average).pow(2).mean(-1,keepdim=True)+epsilon)
    return (X-average)/fenmu*gamma+beta



    """
    Perform Layer Normalization.
    """
    # Your code here
    pass
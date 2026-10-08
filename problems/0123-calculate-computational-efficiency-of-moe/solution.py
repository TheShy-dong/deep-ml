import torch

def compute_efficiency(n_experts: int, k_active: int, d_in: int, d_out: int) -> torch.Tensor:
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs as a torch.Tensor
    """
    result=(n_experts-k_active)/n_experts*100
    return torch.tensor(result)
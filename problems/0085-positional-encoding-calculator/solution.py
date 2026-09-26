import torch
import math

def pos_encoding(position: int, d_model: int):
    """
    Compute positional encodings for Transformer models.

    Args:
        position: sequence length (number of positions)
        d_model: model dimensionality

    Returns:
        torch.Tensor of shape (position, d_model) with dtype float16,
        or -1 if position == 0 or d_model <= 0.
    """

    if position==0 or d_model<=0:
        return -1

    base=10000
    pe=torch.zeros(position,d_model)
    pos=torch.arange(position,dtype=torch.float32).unsqueeze(1)

    indices=torch.arange(0, d_model, 2,dtype=torch.float32)
    div_term=base**(-indices/d_model)

    pe[:,0::2]=torch.sin(div_term*pos)

    pe[:,1::2]=torch.cos(pos*div_term[:pe[:,1::2].shape[1]])

    return pe.to(torch.float16)










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

    if position == 0 or d_model <= 0:
        return -1

    base = 10000

    # 最终的位置编码矩阵
    pe = torch.zeros(position, d_model)

    # 每个 token 的位置：0, 1, 2, ..., position-1
    pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)

    # 偶数维度：0, 2, 4, ...
    indices = torch.arange(0, d_model, 2, dtype=torch.float32)

    # 10000^(-2i / d_model)
    div_term = base ** (-indices / d_model)

    # 偶数维使用 sin
    pe[:, 0::2] = torch.sin(pos * div_term)

    # 奇数维使用 cos
    # 如果 d_model 是奇数，需要少取一个 div_term
    pe[:, 1::2] = torch.cos(pos * div_term[:pe[:, 1::2].shape[1]])

    return pe.to(torch.float16)
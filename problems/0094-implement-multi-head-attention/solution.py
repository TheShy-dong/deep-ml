import torch
import torch.nn.functional as F
import math
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q=X@W_q
    K=X@W_k
    V=X@W_v
    return Q,K,V
    pass

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k=Q.shape[-1]
    scores=Q@K.T/math.sqrt(d_k)
    return F.softmax(scores, dim=-1)@V
    pass

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len=Q.shape[0]
    d_model=Q.shape[-1]
    head_dim=d_model//n_heads
    Q=Q.view(seq_len,n_heads,head_dim)
    K=K.view(seq_len,n_heads,head_dim)
    V=V.view(seq_len,n_heads,head_dim)
    Q=Q.transpose(0,1)#shape (n_heads,seq_len,head_dim)
    K=K.transpose(0,1)
    V=V.transpose(0,1)
    scores=Q@K.transpose(-1,-2)/math.sqrt(head_dim)
    scores=F.softmax(scores, dim=-1)@V#shape(n_heads,seq_len,head_dim)
    scores=scores.transpose(0,1)
    scores=scores.contiguous().view(seq_len,n_heads*head_dim)
    return scores


    pass
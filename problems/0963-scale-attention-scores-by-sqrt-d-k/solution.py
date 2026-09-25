import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    d=len(Q[0])
    scores=Q@K.T/np.sqrt(d)

    ##stable softmax
    scores=scores-np.max(scores,axis=-1,keepdims=True)
    exp_socres=np.exp(scores)
    weights=exp_socres/np.sum(exp_socres,axis=-1,keepdims=True)
    return weights.tolist()



    pass

import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    average=x.mean(-1,keepdim=True)
    sigma=torch.sqrt(eps+(x-average).pow(2).mean(-1,keepdim=True))
    return (x-average)/sigma*gamma+beta




    pass

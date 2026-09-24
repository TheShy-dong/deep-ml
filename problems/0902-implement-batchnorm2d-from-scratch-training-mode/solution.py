import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    average=x.mean(dim=(0,2,3),keepdim=True)
    fenmu=(x-average).pow(2).mean(dim=(0,2,3),keepdim=True)+eps
    return (x-average)/torch.sqrt(fenmu)*gamma.view(1,-1,1,1)+beta.view(1,-1,1,1)
    pass

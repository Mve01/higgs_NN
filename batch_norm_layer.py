import torch
import torch.nn as nn


class BatchNorm_running(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = 0.01
        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var", torch.ones(1, dim))

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0, unbiased = False) + self.eps 
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        ld_scalar = (self.gamma - 0.5 * torch.log(v)).sum()  
        log_det   = ld_scalar.expand(x.size(0))         
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0, unbiased = False) + self.eps
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        ld_scalar = (-self.gamma + 0.5 * torch.log(v)).sum()
        log_det   = ld_scalar.expand(x.size(0)) 
        return x_hat, log_det

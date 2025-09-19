from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from made import MADE


class MAFLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: List[int], reverse: bool):
        super(MAFLayer, self).__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True, seed=None)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())
        mu, raw_log_scale = torch.chunk(out, 2, dim=1)

        #applying a log scale limit to get rid of nan/inf values in samples
        B = 7.0  # ~exp(±3.5) scales; tweak if needed
        log_scale = B * torch.tanh(raw_log_scale / B)

        u = (x - mu) * torch.exp(0.5 * log_scale)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(log_scale, dim=1)
        return u, log_det

    def backward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for dim in range(self.dim):
            out = self.made(x)
            mu, raw_log_scale = torch.chunk(out, 2, dim=1)

            #applying a log scale limit to get rid of nan/inf values in samples
            B = 7.0  # ~exp(±3.5) scales; tweak if needed
            log_scale = B * torch.tanh(raw_log_scale / B)

            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(-0.5 * log_scale[:, dim])
            
        log_det = -0.5 * torch.sum(log_scale, dim=1)
        return x, log_det

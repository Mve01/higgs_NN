from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from maf_layer import MAFLayer
from batch_norm_layer import BatchNorm_running 


class MAF(nn.Module):
    def __init__(
        self, dim: int, n_layers: int, hidden_dims: List[int], device: str, use_reverse: bool = True
    ):
        """
        Args:
            dim: Dimension of input. E.g.: dim = 784 when using MNIST.
            n_layers: Number of layers in the MAF (= number of stacked MADEs).
            hidden_dims: List with sizes of the hidden layers in each MADE. 
            use_reverse: Whether to reverse the input vector in each MADE. 
            device: Device to use for computations. Default : "gpu"
        """
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse, device = self.device))
            self.layers.append(BatchNorm_running(dim).to(device))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=self.device)
        # Forward pass.
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum = log_det + log_det_sum 
        return x, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0], device=self.device)
        # Backward pass.
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum = log_det + log_det_sum
        return x, log_det_sum

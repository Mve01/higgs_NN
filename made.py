from typing import List, Optional
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import ReLU

class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, device: str, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.device = device
        self.mask = None
        self.weight = nn.Parameter(self.weight.to(device))
        if bias:
            self.bias = nn.Parameter(self.bias.to(device))

    def initialise_mask(self, mask: Tensor):
        """Internal method to initialise mask."""
        #self.register_buffer("mask", mask)
        self.mask = mask.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        x = x.to(self.device)
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        device: str,
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = 290713
    ) -> None:
        """Initalise MADE model.
    
        Args:
            n_in: Size of input.
            hidden_dims: List with sizes of the hidden layers.
            gaussian: Whether to use Gaussian MADE. Default: False.
            random_order: Whether to use random order. Default: False.
            seed: Random seed for numpy. Default: 290713.
        """
        super().__init__()
        # Set random seed.
        np.random.seed(seed)
        self.device = device
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        layers = []

        # List of layers sizes.
        dim_list = [self.n_in, *hidden_dims, self.n_out]
        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            layers.append(MaskedLinear(dim_list[i], dim_list[i + 1], device = self.device))
            layers.append(ReLU())
        # Hidden layer to output layer.
        layers.append(MaskedLinear(dim_list[-2], dim_list[-1], device = self.device))
        # Create model.
        self.model = nn.Sequential(*layers).to(device)
        # Get masks for the masked activations.
        self._create_masks()


    def forward(self, x: Tensor) -> Tensor:
        # with torch.no_grad():
        #     out = self.model(x)
        x = x.to(self.device)
        out = self.model(x)
        mu, raw_log_scale = torch.chunk(out, 2, dim=1) 
        return torch.cat([mu, raw_log_scale], dim=1)
    

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        # Define some constants for brevity.
        L = len(self.hidden_dims)
        D = self.n_in

        # Whether to use random or natural ordering of the inputs.
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # Set the connectivity number m for the hidden layers.
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = torch.zeros(len(m_next), len(m), device = self.device)
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int)).to(self.device)
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))

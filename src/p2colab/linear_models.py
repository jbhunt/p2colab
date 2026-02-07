import torch
import numpy as np
from torch import nn

def _make_rasied_cosine_basis_matrix(
    T,
    K,
    t_min=0.0,
    t_max=1.0,
    width_scale=1.0,
    device=None,
    dtype=torch.float32
    ):
    """
    """

    device = "cpu" if device is None else device
    t = torch.linspace(t_min, t_max, T, device=device, dtype=dtype)  # (T,)
    centers = torch.linspace(t_min, t_max, K, device=device, dtype=dtype)  # (K,)
    dc = centers[1] - centers[0]
    w = width_scale * dc
    x = (t[:, None] - centers[None, :]) / w
    Phi = torch.zeros((T, K), device=device, dtype=dtype)
    mask = (x.abs() <= 1.0)
    Phi[mask] = 0.5 * (1.0 + torch.cos(torch.pi * x[mask]))
    Phi = Phi / (Phi.norm(dim=0, keepdim=True) + 1e-8)

    return Phi
    
class RidgeWithTimeBasis(nn.Module):
    """
    """

    def __init__(self, input_size, T, C, K=7, width_scale=1.0):
        """
        """

        super().__init__()
        Phi = _make_rasied_cosine_basis_matrix(T, K, width_scale=width_scale)
        self.register_buffer("Phi", Phi)
        self.C = C
        self.K = K
        self.linear = nn.Linear(in_features=input_size + 1, out_features=K * C, bias=False, dtype=torch.float32)

        return
    
    def forward(self, X):
        """
        """

        N, _ = X.shape
        X_new = torch.cat([torch.ones(N, 1, device=X.device, dtype=torch.float32), X], dim=1)  # N x F + 1

        coef = self.linear(X_new)              # N x C * K
        coef = coef.view(N, self.C, self.K)    # N x C x K

        Y_hat = coef @ self.Phi.T              # N x C x T
        Y_hat = Y_hat.transpose(1, 2)          # N x T x C

        return Y_hat
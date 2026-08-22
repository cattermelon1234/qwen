import torch 
from torch import nn 
import torch.nn.functional as F 

class RMSNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: int = 1e-6
        ):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(dim)); # tells pytorch weight is learnable
            self.eps = eps

    def _norm(self, x: torch.Tensor):
        rms_inverse = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps) # dim = -1 because last dim is token dim
        return x * rms_inverse.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor
        ):
            normalized = self._norm(x)
            return normalized * (1.0 + self.weight)


class RoPE(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__();

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

class DynamicCache(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__();


class TextModel(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__();

if __name__ == "__main__":
    norm = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    print(out.shape)

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
    cos: torch.Tensor
    sin: torch.Tensor
    def __init__(
        self,
        config,
        head_dim,
        base=10_000,
        max_position_embeddings=2048
    ):
        super().__init__();
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2) / head_dim)
        )

        positions = torch.arange(0, max_position_embeddings, dtype=torch.float32)

        # [B, n_heads, max_seq_len, head_dim / 2]
        freqs = torch.outer(positions, inv_freq);

        self.register_buffer(
            "cos",
            freqs.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            freqs.sin(),
            persistent=False,
        )

    # x is input, cos/sin are precomputed rotation matrices for our given theta
    def apply_rotary_emb(self, x, cos, sin):
        # torch.chunk splits head dim into 2 chunks [a, b, c, d] -> x1 = [a, b], x2 = [c, d]
        x1, x2 = torch.chunk(x.float(), 2, dim=-1);
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin

        # RoPE doesn't require pairs to be strictly adjacent, we can set pairs to be (a, c), (b, d)
        # which makes our life significantly easier. Now we just have to concat y1 and y2
        return torch.cat((y1, y2), dim=-1).to(x.dtype);

    def forward(self, x, position_ids):
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return self.apply_rotary_emb(x, cos, sin)



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

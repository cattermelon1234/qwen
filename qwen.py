import torch 
from torch import nn 
import torch.nn.functional as F 
from config import QwenConfig

class RMSNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6
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
    ):
        super().__init__();

        if (config.head_dim % 2 != 0):
            raise ValueError("head_dim must be even for RoPE")

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, config.head_dim, 2) / config.head_dim)
        )

        positions = torch.arange(0, config.max_position_embeddings, dtype=torch.float32)

        # [max_position_embeddings, head_dim / 2]
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
        # x is [batch_size, n_heads, seq_len, head_dim]
        # pos_ids is [batch_size, seq_len]
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return self.apply_rotary_emb(x, cos, sin)


class SelfAttention(nn.Module):
    def __init__(
        self, config
        ): 
        super().__init__();
        self.hidden_size = config.hidden_size
        self.num_attention_heads = self.num_attention_heads # because attn heads != k_heads/q_heads in GQA
        self.head_dim = self.head_dim # head dim remains const even in GQA
        self.num_kv_heads = self.num_kv_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads # number of kv heads per q head

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.attention_heads * 2)

        # normally, hidden_size = head_dim * num_heads, but in GQA, we want to 
        # project k, v into a smaller dim to conserve space. our head_dim remains the same, 
        # but we have less heads. Thus, head_dim * num_kv_heads < hidden_size 
        self.k = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads)
        self.v = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads)

        # input to out proj is the concatenation of all attn heads (num_heads * head_dim -> hidden_size)
        self.proj_out = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states : torch.Tensor,
        pos_embedding,
        attention_mask: torch.Tensor | None = None,
        cache = None
    ):
        
        batch_size, seq_len, _ = hidden_states.shape
        q_proj = self.q_proj(hidden_states)
        q, gate = torch.chunk(q_proj, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)
        
        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim))).transpose(1, 2)
        k = self.k_norm(k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim))).transpose(1, 2)
        v = self.v_norm(v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim))).transpose(1, 2)

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
    config = QwenConfig() 
    batch_size = 4
    seq_len = 24
    norm = RMSNorm(64)
    out = norm(torch.randn(4, 20, 64))
    print(out.shape)

    rope = RoPE(config)
    query = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    print("query: ")
    print(query.shape)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rotated_query = rope(query, position_ids)
    print(rotated_query.shape)
    

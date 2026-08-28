from dataclasses import dataclass 

@dataclass
class QwenConfig:
  vocab_size: int = 151_936
  hidden_size: int = 2_048
  intermediate_size: int = 6_144
  num_hidden_layers: int = 28
  num_attention_heads: int = 16
  num_key_value_heads: int = 8

  max_position_embeddings: int = 40_960
  rope_theta: float = 1_000_000.0
  rms_norm_eps: float = 1e-6
  attention_bias: bool = False

  tie_word_embeddings: bool = True

  def __post_init__(self):
      if self.hidden_size % self.num_attention_heads != 0:
          raise ValueError(
              "hidden_size must be divisible by num_attention_heads"
          )
      if self.num_attention_heads % self.num_key_value_heads != 0:
          raise ValueError(
              "num_attention_heads must be divisible by num_key_value_heads"
          )

  @property
  def head_dim(self) -> int:
      return self.hidden_size // self.num_attention_heads

from dataclasses import dataclass 

@dataclass
class QwenConfig:
  vocab_size: int = 151_936
  hidden_size: int = 2_048
  intermediate_size: int = 5_632
  num_hidden_layers: int = 24
  num_attention_heads: int = 16
  num_key_value_heads: int = 2

  max_position_embeddings: int = 32_768
  rope_theta: float = 10_000.0
  rms_norm_eps: float = 1e-6

  tie_word_embeddings: bool = True

  def __post_init__(self):
      if self.hidden_size % self.num_attention_heads != 0:
          raise ValueError(
              "hidden_size must be divisible by num_attention_heads"
          )

  @property
  def head_dim(self) -> int:
      return self.hidden_size // self.num_attention_heads


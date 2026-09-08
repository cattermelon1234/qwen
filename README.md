# Qwen3 from scratch

A PyTorch implementation of dense Qwen3 inference. Start with the official
`Qwen/Qwen3-0.6B` checkpoint. Its head dimension is **128**, independently of
`hidden_size / num_attention_heads`.

## Setup

```sh
uv pip install --python .venv/bin/python -r requirements.txt
```

The reference version is pinned to Transformers 4.57.6.
PyTorch 2.13.0 was used during development.

## Generate

```sh
.venv/bin/python generate.py 'Explain gravity in one sentence.'
.venv/bin/python generate.py 'What is 2 + 2?' --device mps --dtype bfloat16
```

Generation uses the checkpoint chat template with thinking disabled and greedy decoding.
This is useful for deterministic comparisons; sampling is not implemented.
For batched prompts, use left padding. Right padding and context overflow are
rejected. Calls to `Qwen.forward` require explicit positions and an additive
4D attention mask.

## How loading works

`nn.Linear` initializes and registers a layer. Assigning it to `self.gate_proj`
gives its parameters names such as `gate_proj.weight`. It does not read weights
from disk. `pretrained.copy_reference_weights` renames Hugging Face state-dict
keys to match our module names, then `load_state_dict(strict=True)` copies the
tensors into those registered parameters. Missing/extra keys and incompatible
shapes fail instead of leaving random weights unnoticed.

For tied embeddings, `embed_tokens.weight` and `lm_head.weight` refer to the
same parameter. Input token IDs select rows; output hidden vectors multiply
by its transpose to produce vocabulary logits.

The loader reads the checkpoint configuration and supports dense Qwen3 with
standard RoPE, full attention, and SiLU. Other model families, scaled RoPE,
and sliding-window configurations are rejected. It temporarily loads a Hugging
Face reference model to provide straightforward, auditable weight conversion.

Sources: [checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B),
[configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json),
[reference implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py).

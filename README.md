# Qwen3 from scratch

A PyTorch implementation of dense Qwen3 inference. Start with the official
`Qwen/Qwen3-0.6B` checkpoint. Its head dimension is **128**, independently of
`hidden_size / num_attention_heads`.

## Setup

```sh
uv pip install --python .venv/bin/python -r requirements.txt
```

The reference version is pinned to Transformers 4.57.6 because the tests use
its Qwen3 component and cache APIs. PyTorch 2.13.0 was used during development.

## Run tests (no checkpoint download)

```sh
.venv/bin/python -m pytest -q
```

Tests create tiny random Hugging Face models and copy exactly the same weights
into this implementation. They check strict loading, shared embeddings,
float32/BF16 RMSNorm and RoPE, individual components on identical inputs,
every transformer block, final logits, causality, left padding, cached single
and multi-token decoding, every layer's K/V cache, greedy generation, EOS,
finished rows, and zero new tokens. The tiny attention width intentionally
exceeds the residual width.

## Verify the real checkpoint

```sh
.venv/bin/python verify_checkpoint.py
```

This downloads the original safetensors checkpoint and tokenizer into
`.cache/huggingface` and runs both implementations on CPU in float32. Allow
several GB of RAM for two models and loading overhead. Downloads are reused.
The script prints the resolved checkpoint commit; pass `--revision COMMIT`
to reproduce that exact version. It reports maximum and mean absolute errors
for intermediate outputs and logits, verifies cached decoding and K/V tensors,
and compares 12 greedy generated tokens. Reference parity assertions use `rtol=1e-4, atol=1e-5`; integer token
comparisons are exact. Comparing different prefill/decode matrix shapes uses
`atol=1e-4` for logits and `2e-4` for K/V: the real checkpoint showed up to
`8.77e-5` logit drift and `1.13e-4` V drift, while the corresponding cached
reference computations matched. Padding positions are excluded
from activation comparisons because their query outputs are unspecified.

If a test fails, investigate the first divergent activation. Do not simply
increase tolerances. `compare_layers` removes its diagnostic hooks even on
failure. `test_same_input_components` isolates components from errors propagated
by earlier layers.

## Generate

```sh
.venv/bin/python generate.py 'Explain gravity in one sentence.'
.venv/bin/python generate.py 'What is 2 + 2?' --device mps --dtype bfloat16
```

CPU float32 is the verification baseline. MPS/BF16 is an optional inference
mode and is not covered by the full-model float32 parity assertions. Generation
uses the checkpoint chat template with thinking disabled and greedy decoding.
This is useful for deterministic comparisons; sampling is not implemented.
For batched prompts, use left padding. Right padding and context overflow are
rejected. Calls to `Qwen.forward` require explicit positions and an additive
4D attention mask; `verification.forward` constructs them from a 2D padding mask.

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

## Verified result

On CPU float32, checkpoint commit
`c1899de289a04d12100db370d81485cdf75e47ca` passed all 28 layers, final logits,
cached decoding/K/V, and 12-token greedy continuation checks. All reported
layer and logit errors against the reference were zero for the test prompts.
The local suite passed 14 tests. These checks cover the selected inputs, not
every possible prompt or device.

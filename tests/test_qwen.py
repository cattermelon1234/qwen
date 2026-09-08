from dataclasses import asdict
from unittest.mock import patch
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb
from config import QwenConfig
from qwen import Qwen, RMSNorm, RoPE
from pretrained import copy_reference_weights
from verification import forward, compare_layers, compare_cache, assert_close


@pytest.fixture
def pair():
    torch.manual_seed(42)
    config = QwenConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    hf = Qwen3Config(**asdict(config))
    hf._attn_implementation = 'eager'
    reference = Qwen3ForCausalLM(hf).eval()
    ours = Qwen(config).eval()
    copy_reference_weights(ours, reference)
    return ours, reference


def test_config_and_strict_loading(pair):
    ours, reference = pair
    assert QwenConfig().head_dim == 128
    assert ours.layers[0].attention.q_proj.weight.shape == (64, 32)
    assert ours.lm_head.weight is ours.embed_tokens.weight
    assert QwenConfig.from_hf(reference.config) == ours.config
    weights = ours.state_dict()
    del weights['norm.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        ours.load_state_dict(weights, strict=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_norm_and_rope(pair, dtype):
    ours, reference = pair
    x = torch.randn(2, 4, 7, 16).to(dtype)
    a, b = RMSNorm(16).to(dtype), Qwen3RMSNorm(16).to(dtype)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    positions = torch.tensor([[0, 1, 2, 9, 17, 32, 63]]).expand(2, -1)
    rope = Qwen3RotaryEmbedding(reference.config)
    cos, sin = rope(x, positions)
    expected, _ = apply_rotary_pos_emb(x, x, cos, sin)
    actual = RoPE(ours.config).to(dtype)(x, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('padded', [False, True])
def test_layers_and_logits(pair, padded):
    ours, reference = pair
    ids = torch.randint(3, 97, (2, 7))
    padding = torch.ones_like(ids)
    if padded:
        padding[0, :2] = 0
        ids[0, :2] = 0
    compare_layers(ours, reference, ids, padding)


@pytest.mark.parametrize('chunks', [[4, 1, 1, 1], [2, 3, 2]])
@pytest.mark.parametrize('padded', [False, True])
def test_cache(pair, chunks, padded):
    ours, reference = pair
    ids = torch.randint(3, 97, (2, 7))
    padding = torch.ones_like(ids)
    if padded:
        padding[0, 0] = 0
        ids[0, 0] = 0
    compare_cache(ours, reference, ids, chunks, padding)


@torch.inference_mode()
def test_causality_and_padding(pair):
    ours, _ = pair
    ids = torch.randint(3, 97, (2, 7))
    original = forward(ours, ids)
    changed = ids.clone()
    changed[:, 4:] = torch.randint(3, 97, (2, 3))
    assert_close(forward(ours, changed)[:, :4], original[:, :4], 'causality')
    padded = torch.cat([torch.zeros((2, 3), dtype=torch.long), ids], dim=1)
    mask = (padded != 0).long()
    assert_close(forward(ours, padded, mask)[:, 3:], original, 'left padding')


def test_greedy_generation(pair):
    ours, reference = pair
    ids = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]])
    mask = ids.ne(0).long()
    actual = ours.generate(ids, 6, mask)
    expected = reference.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False)
    assert torch.equal(actual, expected)
    for i in range(2):
        single = ours.generate(ids[i:i+1, mask[i].bool()], 6)
        # Generated continuations can stop earlier for an individual row.
        continuation = single[:, mask[i].sum():]
        assert torch.equal(actual[i:i+1, 4:4+continuation.shape[1]], continuation)
    assert torch.equal(ours.generate(ids, 0, mask), ids)


def test_finished_rows(pair):
    ours, _ = pair
    # Row 0 ends immediately; row 1 emits token 3 and then EOS.
    scripted = [torch.tensor([2, 3]), torch.tensor([9, 2])]
    def fake_forward(input_ids, **kwargs):
        logits = torch.full((*input_ids.shape, 97), -100.0)
        logits[torch.arange(2), -1, scripted.pop(0)] = 100
        return logits
    with patch.object(ours, 'forward', side_effect=fake_forward):
        output = ours.generate(torch.tensor([[4], [5]]), 5)
    assert output.tolist() == [[4, 2, 0], [5, 3, 2]]


def test_generation_validation(pair):
    ours, _ = pair
    with pytest.raises(ValueError, match='left padding'):
        ours.generate(torch.tensor([[3, 0]]), 1, torch.tensor([[1, 0]]))
    with pytest.raises(ValueError, match='context'):
        ours.generate(torch.tensor([[3]]), 66)


def test_same_input_components(pair):
    ours, reference = pair
    x = torch.randn(2, 5, 32)
    for a, b in zip(ours.layers, reference.model.layers):
        assert_close(a.input_layernorm(x), b.input_layernorm(x), 'isolated input norm')
        assert_close(a.post_attention_layernorm(x), b.post_attention_layernorm(x), 'isolated post norm')
        assert_close(a.mlp(x), b.mlp(x), 'isolated MLP')
        positions = torch.arange(5)[None, :].expand(2, -1)
        from qwen import make_attention_mask
        mask = make_attention_mask(5, 0, torch.ones(2, 5), torch.float32, 'cpu')
        emb = reference.model.rotary_emb(x, positions)
        expected = b.self_attn(x, position_embeddings=emb, attention_mask=mask)[0]
        assert_close(a.attention(x, positions, mask), expected, 'isolated attention')

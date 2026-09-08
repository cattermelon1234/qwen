"""Shared numerical checks for tiny models and the real checkpoint."""
import torch
from qwen import DynamicCache, make_attention_mask


def forward(model, ids, padding=None, cache=None):
    past = cache.seq_len if cache is not None else 0
    if padding is None:
        padding = torch.ones((ids.shape[0], past + ids.shape[1]), device=ids.device, dtype=torch.long)
    positions = padding.long().cumsum(-1) - 1
    positions.masked_fill_(~padding.bool(), 0)
    mask = make_attention_mask(ids.shape[1], past, padding,
                               model.embed_tokens.weight.dtype, ids.device)
    return model(ids, positions[:, past:], mask, cache)


def assert_close(actual, expected, label, report=False, atol=1e-5):
    error = (actual.float() - expected.float()).abs()
    stats = f'{label}: max={error.max().item():.3g}, mean={error.mean().item():.3g}'
    if report:
        print(stats, flush=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=atol, msg=stats)


@torch.inference_mode()
def compare_layers(ours, reference, ids, padding=None, report=False):
    if padding is None:
        padding = torch.ones_like(ids)
    positions = padding.long().cumsum(-1) - 1
    positions.masked_fill_(~padding.bool(), 0)
    expected, handles = {}, []
    # Capture reference activations, then compare ours immediately in forward order.
    # Includes submodules to locate a discrepancy within a block.
    modules = [('embedding', ours.embed_tokens, reference.model.embed_tokens)]
    for i, (a, b) in enumerate(zip(ours.layers, reference.model.layers)):
        modules.extend([
            (f'layer {i} input norm', a.input_layernorm, b.input_layernorm),
            (f'layer {i} attention', a.attention, b.self_attn),
            (f'layer {i} post-attention norm', a.post_attention_layernorm, b.post_attention_layernorm),
            (f'layer {i} MLP', a.mlp, b.mlp),
            (f'layer {i} output', a, b),
        ])
    modules.extend([('final norm', ours.norm, reference.model.norm)])
    def capture(label):
        def hook(module, inputs, output):
            expected[label] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
    def check(label):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            assert_close(value[padding.bool()], expected.pop(label)[padding.bool()], label, report)
        return hook
    try:
        for label, a, b in modules:
            handles.append(b.register_forward_hook(capture(label)))
        ref_logits = reference(ids, attention_mask=padding, position_ids=positions,
                               use_cache=False).logits
        for label, a, b in modules:
            handles.append(a.register_forward_hook(check(label)))
        logits = forward(ours, ids, padding)
        assert_close(logits[padding.bool()], ref_logits[padding.bool()], 'logits', report)
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def compare_cache(ours, reference, ids, chunks, padding=None, report=False):
    if padding is None:
        padding = torch.ones_like(ids)
    assert sum(chunks) == ids.shape[1]
    full_cache = DynamicCache(ours.config, ids.shape[0], ids.device, ours.embed_tokens.weight.dtype)
    full = forward(ours, ids, padding, full_cache)
    cache = DynamicCache(ours.config, ids.shape[0], ids.device, ours.embed_tokens.weight.dtype)
    ref_cache = None
    start = 0
    for size in chunks:
        end = start + size
        mask = padding[:, :end]
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(~mask.bool(), 0)
        result = reference(ids[:, start:end], attention_mask=mask,
                           position_ids=positions[:, start:end], past_key_values=ref_cache,
                           use_cache=True)
        ref_cache = result.past_key_values
        actual = forward(ours, ids[:, start:end], mask, cache)
        real = padding[:, start:end].bool()
        assert_close(actual[real], result.logits[real], f'cache vs reference {start}:{end}', report)
        # Different GEMM shapes in prefill vs decode change FP32 rounding.
        # First require tighter parity against the reference's SAME chunks.
        assert_close(actual[real], full[:, start:end][real], f'cache vs full {start}:{end}', report, atol=1e-4)
        assert cache.seq_len == end
        for i in range(ours.config.num_hidden_layers):
            # Padded query states are unspecified; compare real cached positions.
            keep = mask.bool()[:, None, :, None].expand_as(cache.k_cache[i])
            for label, actual_kv, full_kv, ref_kv in (
                ('K', cache.k_cache[i], full_cache.k_cache[i], ref_cache[i][0]),
                ('V', cache.v_cache[i], full_cache.v_cache[i], ref_cache[i][1]),
            ):
                assert_close(actual_kv[keep], ref_kv[keep], f'layer {i} {label} vs reference')
                assert_close(actual_kv[keep], full_kv[:, :, :end][keep], f'layer {i} {label} vs full', atol=2e-4)
        start = end

"""Download and check real weights: python verify_checkpoint.py"""
import argparse
import torch
from transformers import GenerationConfig
from pretrained import DEFAULT_MODEL, load_pretrained
from verification import compare_layers, compare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default='main', help='Use a commit hash for reproducible downloads')
    args = parser.parse_args()
    torch.set_num_threads(4)
    ours, tokenizer, reference = load_pretrained(args.model, revision=args.revision, keep_reference=True)
    print(f'Checkpoint revision: {reference.config._commit_hash}', flush=True)
    prompts = ['What is 2 + 2?', 'Explain gravity in one short sentence.']
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
             tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
    inputs = tokenizer(texts, padding=True, return_tensors='pt', add_special_tokens=False)
    ids, padding = inputs.input_ids, inputs.attention_mask
    compare_layers(ours, reference, ids, padding, report=True)
    compare_cache(ours, reference, ids, [ids.shape[1] - 3, 1, 2], padding, report=True)
    actual = ours.generate(ids, 12, padding)
    generation = GenerationConfig(
        max_new_tokens=12, do_sample=False,
        eos_token_id=ours.config.eos_token_id, pad_token_id=ours.config.pad_token_id,
    )
    expected = reference.generate(ids, attention_mask=padding, generation_config=generation,
                                  use_model_defaults=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for i in range(len(prompts)):
        print(tokenizer.decode(actual[i, ids.shape[1]:], skip_special_tokens=True))
    print('PASS: layer/logit parity, cached logits and K/V, greedy generation')


if __name__ == '__main__':
    main()

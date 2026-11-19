"""Small, auditable SFT warm-up experiment; no pretrained weights required."""
import copy
import hashlib
import json
import platform
import random
import re
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


def prompt(number):
    return 'is_even ' + ' '.join(str(number)) + ' answer yes or no'


def target(number):
    return 'yes' if number % 2 == 0 else 'no'


def prepare(seed=14014):
    torch.manual_seed(seed)
    tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'user', 'assistant',
              'is_even', 'answer', 'yes', 'no', 'or'] + list('0123456789')
    raw = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]',
        unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    # Group by numeric identity; every split is balanced. No number is reused from lesson 12.
    rng = random.Random(seed)
    splits = {name: [] for name in ['train', 'validation', 'test']}
    for parity in [0, 1]:
        values = [n for n in range(100, 1000) if n % 2 == parity]
        rng.shuffle(values)
        for name, subset in zip(splits, [values[:120], values[120:160], values[160:200]]):
            splits[name].extend(subset)
    for values in splits.values():
        values.sort()
    config = Qwen2Config(vocab_size=len(tokens), hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=2, eos_token_id=3,
        tie_word_embeddings=True, attn_implementation='sdpa')
    model = Qwen2ForCausalLM(config)
    return tokenizer, model, splits


def encode_prompt(tokenizer, number):
    return tokenizer.apply_chat_template([dict(role='user', content=prompt(number))],
        tokenize=True, add_generation_prompt=True)


def batch(tokenizer, numbers, device):
    inputs, labels = [], []
    for n in numbers:
        prefix = encode_prompt(tokenizer, n)
        answer = tokenizer.encode(target(n), add_special_tokens=False) + [tokenizer.eos_token_id]
        inputs.append(prefix + answer)
        labels.append([-100] * len(prefix) + answer)
    # All three-digit prompts have equal length. Padding would need attention and label masks.
    assert len({len(row) for row in inputs}) == 1
    ids = torch.tensor(inputs, device=device)
    assert not ids.eq(tokenizer.unk_token_id).any(), 'Prompt contains an unknown token'
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=torch.tensor(labels, device=device))


@torch.inference_mode()
def evaluate(model, tokenizer, numbers, stage, split):
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([encode_prompt(tokenizer, n) for n in numbers], device=device)
    generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
        max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    rows = []
    for n, suffix in zip(numbers, generated[:, ids.shape[1]:].tolist()):
        answer = tokenizer.decode(suffix, skip_special_tokens=True).strip().lower()
        rows.append(dict(number=n, stage=stage, split=split, target=target(n), answer=answer,
            valid=answer in {'yes', 'no'}, correct=answer == target(n),
            eos=tokenizer.eos_token_id in suffix, token_ids=json.dumps(suffix)))
    return rows


def warmup(tokenizer, initial, splits, steps=200, seed=14014, callback=None):
    model = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(seed)
    history = []
    for step in range(1, steps + 1):
        model.train()
        sample = rng.sample(splits['train'], 32)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(tokenizer, sample, next(model.parameters()).device)).loss
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0:
            rows = evaluate(model, tokenizer, splits['validation'], 'sft', 'validation')
            point = dict(step=step, loss=loss.item(), grad_norm=float(grad),
                valid_rate=sum(r['valid'] for r in rows)/len(rows),
                accuracy=sum(r['correct'] for r in rows)/len(rows))
            history.append(point)
            if callback:
                callback(history)
    return model, history


def export(root, tokenizer, model, initial, splits, history, seed=14014):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, obj in [('model', model), ('initial', initial)]:
        obj.save_pretrained(root/name)
        tokenizer.save_pretrained(root/name)
    for name, numbers in splits.items():
        rows = [dict(data_source='decimal_parity', prompt=[dict(role='user', content=prompt(n))],
                     reward_model=dict(style='rule', ground_truth=target(n)),
                     extra_info=dict(number=n, split=name)) for n in numbers]
        pd.DataFrame(rows).to_parquet(root/f'{name}.parquet', index=False)
    data = json.dumps(splits, sort_keys=True)
    (root/'splits.json').write_text(data)
    manifest = dict(seed=seed, steps=200, learning_rate=1e-3, batch_size=32,
        split_sha256=hashlib.sha256(data.encode()).hexdigest(), torch=torch.__version__,
        python=platform.python_version(), device=str(next(model.parameters()).device),
        scope='Three-digit parity; fixed-step SFT; no GRPO result implied.')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(history).to_csv(root/'history.csv', index=False)
    return manifest


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return float(solution_str.strip().lower() == ground_truth)


def evaluate_grpo(run, output):
    """Read only checkpoints created locally by this course (pickle is trusted input)."""
    run, output = Path(run), Path(output)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(run.parent/'model')
    model = Qwen2ForCausalLM.from_pretrained(run.parent/'model').cuda()
    splits = json.loads((run.parent/'splits.json').read_text())
    rows = evaluate(model, tokenizer, splits['test'], 'sft', 'test')
    state = torch.load(run/'checkpoints/global_step_10/actor/model_world_size_1_rank_0.pt',
                       map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    rows += evaluate(model, tokenizer, splits['test'], 'grpo', 'test')
    frame = pd.DataFrame(rows)
    output.mkdir(exist_ok=True, parents=True)
    frame.to_csv(output/'grpo_predictions.csv', index=False)
    log = re.sub(r'\x1b\[[0-9;]*m', '', (run/'train.log').read_text())
    metrics = []
    for line in log.splitlines():
        if 'training/global_step:' in line:
            metrics.append({k: float(v) for k, v in re.findall(
                r'([\w/@.-]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', line)})
    steps = sorted({int(row['training/global_step']) for row in metrics})
    assert steps == list(range(1, 11)), f'Expected ten completed steps, got {steps}'
    pd.DataFrame(metrics).to_csv(output/'grpo_metrics.csv', index=False)
    report = dict(steps=steps, nonzero_gradient_steps=sum(row.get('actor/grad_norm', 0)>0 for row in metrics),
                  scope='Fixed SFT checkpoint vs ten additional GRPO steps; one seed; no continued-SFT control.',
                  test_accuracy=frame.groupby('stage').correct.mean().to_dict())
    (output/'grpo_report.json').write_text(json.dumps(report, indent=2))
    return frame.groupby('stage')[['valid', 'correct', 'eos']].mean()

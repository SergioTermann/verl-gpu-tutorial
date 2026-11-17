"""Pretrained Qwen LoRA experiment with explicit labels and artifact checks."""
import gc
import hashlib
from importlib.metadata import version
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
SYSTEM = ('Route the support ticket. Billing or payment -> R0; account access or login -> R1; '
          'parcel or delivery -> R2. Reply with exactly route=R0, route=R1, or route=R2. '
          'Do not add an explanation.')
TEMPLATES = {
    'R0': ['My card was charged twice for one order.', 'I need a copy of my invoice.',
           'The refund has not reached my bank.', 'The amount on my bill is wrong.',
           'Can I change my payment method?', 'A payment failed although my card works.',
           'I was billed after cancelling the subscription.', 'Please explain this extra fee.',
           'The receipt shows a different price from checkout.', 'I want to dispute a debit on my statement.'],
    'R1': ['I cannot sign in to my account.', 'I forgot my password.',
           'The password reset link does not work.', 'My account is locked.',
           'The login verification code never arrives.', 'My authenticator code is rejected.',
           'The sign-in page keeps sending me back.', 'I lost access to the email used for login.',
           'The service says my credentials are invalid.', 'I cannot complete two-factor authentication.'],
    'R2': ['My parcel has not arrived.', 'The tracking page has stopped updating.',
           'Please change the delivery address.', 'My package was delivered to the wrong house.',
           'The courier missed the scheduled delivery.', 'The package is marked delivered but missing.',
           'Can I choose another delivery date?', 'My shipment is stuck at the distribution center.',
           'The carrier has sent my parcel back to the sender.', 'Where can I collect the shipment after a missed drop-off?'],
}


def dataset():
    # Split by template family BEFORE creating ticket-ID variants.
    splits = {name: [] for name in ['train', 'validation', 'test']}
    for label, templates in TEMPLATES.items():
        for index, text in enumerate(templates):
            split = 'train' if index < 6 else 'validation' if index < 8 else 'test'
            for variant in range(4):
                ticket_id = f'{index*4+variant+100}'  # Same IDs occur across labels; not a label shortcut.
                splits[split].append(dict(id=f'{label}-{index}-{variant}', group=f'{label}-{index}',
                    text=f'Ticket {ticket_id}: {text}', target=f'route={label}', split=split))
    groups = {k: {r['group'] for r in v} for k, v in splits.items()}
    assert not(groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    return splits


def messages(row):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=row['text'])]


def resolve_model():
    local = os.environ.get('VERL_BASE_MODEL')
    conventional = Path.home()/'.cache/verl-course/Qwen2.5-0.5B-Instruct'
    if local:
        path = Path(local).expanduser()
        assert path.is_dir(), f'Model directory not found: {path}'
        return path
    if (conventional/'model.safetensors').exists():
        return conventional
    return Path(snapshot_download(MODEL_ID, revision=REVISION,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'README.md', 'LICENSE']))


def load_base(path):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).cuda()


def encode(tokenizer, row, max_length=160):
    prefix = tokenizer.apply_chat_template(messages(row), tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages(row)+[dict(role='assistant', content=row['target'])],
                                         tokenize=True, add_generation_prompt=False)
    assert full[:len(prefix)] == prefix, 'Chat template prefix differs; do not silently approximate masks'
    # Keep assistant text and the first EOS, excluding template newline after EOS.
    end = full.index(tokenizer.eos_token_id, len(prefix))+1
    full = full[:end]
    assert len(full) <= max_length, 'Input too long: review truncation policy instead of dropping the answer'
    labels = [-100]*len(prefix)+full[len(prefix):]
    assert any(x != -100 for x in labels) and labels[-1] == tokenizer.eos_token_id
    return full, labels


def collate(tokenizer, rows):
    pairs = [encode(tokenizer, r) for r in rows]
    length = max(len(ids) for ids, _ in pairs)
    return dict(input_ids=torch.tensor([a+[tokenizer.pad_token_id]*(length-len(a)) for a,b in pairs],device='cuda'),
        attention_mask=torch.tensor([[1]*len(a)+[0]*(length-len(a)) for a,b in pairs],device='cuda'),
        labels=torch.tensor([b+[-100]*(length-len(b)) for a,b in pairs],device='cuda'))


def sampled_base_hash(model):
    """Hash three frozen tensors; this is explicitly a sample, not a full-model checksum."""
    chosen = [(name, value) for name, value in model.named_parameters() if 'lora_' not in name][:3]
    digest = hashlib.sha256()
    for name, value in chosen:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, tokenizer, rows, stage, batch_size=4):
    model.eval()
    records = []
    original_padding = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            strings = [tokenizer.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True) for r in batch]
            encoded = tokenizer(strings, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
            result = model.generate(**encoded, max_new_tokens=12, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            for row, tokens in zip(batch, result[:, encoded.input_ids.shape[1]:]):
                answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                records.append(dict(**row, stage=stage, answer=answer,
                    valid=answer in {'route=R0','route=R1','route=R2'}, correct=answer==row['target'],
                    eos=bool((tokens==tokenizer.eos_token_id).any())))
    finally:
        tokenizer.padding_side = original_padding
    return records


def train(run, callback=None, seed=17017, steps=60):
    run = Path(run); run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); random.seed(seed)
    path = resolve_model()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = dataset()
    for rows in splits.values():
        for row in rows: encode(tokenizer, row)
    model = get_peft_model(load_base(path), LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
        target_modules=['q_proj','v_proj'], bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
    base_hash = sampled_base_hash(model)
    start_state = {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    def optimizer(): return torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01)
    # Diagnostic branch: deliberately fit one batch, then reset every adapter tensor and optimizer.
    opt = optimizer(); diagnostic = []
    model.train()
    fixed = collate(tokenizer, [splits['train'][0], splits['train'][24], splits['train'][48]])
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        loss = model(**fixed).loss; loss.backward(); opt.step()
        diagnostic.append(loss.item())
    assert diagnostic[-1] < diagnostic[0], 'Single-batch loss did not decrease; stop before scaling up'
    set_peft_model_state_dict(model, start_state)
    assert all(torch.equal(v.cpu(), start_state[k]) for k,v in get_peft_model_state_dict(model).items())
    del opt, fixed
    opt = optimizer(); rng = random.Random(seed); history = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for step in range(1, steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        rows = rng.sample(splits['train'], 8)
        microbatches = [collate(tokenizer, rows[i:i+2]) for i in range(0, 8, 2)]
        total_tokens = sum(int((b['labels'][:,1:]!=-100).sum()) for b in microbatches)
        mean_loss = 0.
        for b in microbatches:
            count = int((b['labels'][:,1:]!=-100).sum())
            loss = model(**b).loss
            (loss*count/total_tokens).backward()
            mean_loss += loss.item()*count/total_tokens
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        assert torch.isfinite(norm), 'Nonfinite gradient'
        opt.step()
        if step == 1 or step % 10 == 0:
            validation = evaluate(model, tokenizer, splits['validation'], 'lora')
            point = dict(step=step, loss=mean_loss, grad_norm=float(norm),
                         validation_accuracy=sum(r['correct'] for r in validation)/len(validation))
            history.append(point)
            if callback: callback(history)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    assert sampled_base_hash(model) == base_hash, 'Sampled base tensors changed'
    adapter_change = max((v.detach().cpu()-start_state[k]).abs().max().item() for k,v in get_peft_model_state_dict(model).items())
    assert adapter_change > 0
    # Only now view the final holdout, using exactly the same frozen base weights.
    with model.disable_adapter(): predictions = evaluate(model, tokenizer, splits['test'], 'base')
    predictions += evaluate(model, tokenizer, splits['test'], 'lora')
    model.save_pretrained(run/'adapter', safe_serialization=True)
    tokenizer.save_pretrained(run/'adapter')
    (run/'dataset.json').write_text(json.dumps(splits, indent=2))
    data_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    manifest = dict(model_id=MODEL_ID, revision=REVISION, seed=seed, steps=steps,
        source_note='Expected upstream revision; file hashes identify actual local artifacts, including overrides.',
        base_weights_sha256={p.name:file_hash(p) for p in sorted(path.glob('*.safetensors'))},
        tokenizer_sha256=file_hash(path/'tokenizer.json'),
        lora_rank=8, lora_alpha=16, targets=['q_proj','v_proj'], learning_rate=3e-4,
        microbatch=2, gradient_accumulation=4, effective_batch=8, max_length=160,
        trainable_parameters=sum(p.numel() for p in trainable), total_parameters=sum(p.numel() for p in model.parameters()),
        base_sample_hash_unchanged=True, adapter_max_change=adapter_change,
        data_sha256=data_hash, elapsed_seconds_including_validation=elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        torch=torch.__version__, transformers=version('transformers'), peft=version('peft'),
        gpu=torch.cuda.get_device_name(0),
        diagnostic_first_loss=diagnostic[0], diagnostic_last_loss=diagnostic[-1],
        diagnostic_reset=True, training_dtype='BF16 base; FP32 LoRA parameters',
        scope='Synthetic ticket routing; holdout by template family; one seed; no full-finetune control.')
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2))
    pd.DataFrame(history).to_csv(run/'history.csv', index=False)
    pd.DataFrame(predictions).to_csv(run/'predictions.csv', index=False)
    # Evaluate a fixed validation probe before unloading. No test-based acceptance threshold.
    probe = collate(tokenizer, splits['validation'][:3])
    model.eval()
    with torch.inference_mode(): expected = model(**{k:v for k,v in probe.items() if k!='labels'}).logits.detach().cpu()
    del opt, model, trainable, microbatches, b, loss
    gc.collect(); torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(path), run/'adapter').eval()
    with torch.inference_mode(): actual = reloaded(**{k:v for k,v in probe.items() if k!='labels'}).logits.detach().cpu()
    reload_diff = float((expected-actual).abs().max())
    assert torch.equal(expected, actual), f'Adapter reload changed probe logits: {reload_diff}'
    merged = reloaded.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode(): combined = merged(**{k:v for k,v in probe.items() if k!='labels'}).logits.detach().cpu()
    bf16_relative_error = float((combined.float()-actual.float()).norm()/actual.float().norm().clamp_min(1e-12))
    assert torch.isfinite(combined).all(), 'Nonfinite BF16 merged logits'
    # Diagnose merge algebra in FP32, rather than loosening the BF16 discrepancy threshold.
    del merged, reloaded, actual, combined
    gc.collect(); torch.cuda.empty_cache()
    precise = PeftModel.from_pretrained(load_base(path), run/'adapter').float().eval()
    with torch.inference_mode(): reference32 = precise(**{k:v for k,v in probe.items() if k!='labels'}).logits.detach().cpu()
    merged = precise.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode(): combined32 = merged(**{k:v for k,v in probe.items() if k!='labels'}).logits.detach().cpu()
    relative_error = float((combined32-reference32).norm()/reference32.norm().clamp_min(1e-12))
    assert torch.isfinite(combined32).all() and relative_error < 1e-4, 'FP32 merge failed the fixed numerical check'
    merged_predictions = evaluate(merged, tokenizer, splits['test'], 'merged')
    pd.DataFrame(merged_predictions).to_csv(run/'merged_predictions.csv', index=False)
    # This is approximate in BF16; report answer disagreements instead of hiding them.
    lora_answers = {r['id']:r['answer'] for r in predictions if r['stage']=='lora'}
    disagreements = sum(r['answer']!=lora_answers[r['id']] for r in merged_predictions)
    merged.save_pretrained(run/'merged', safe_serialization=True); tokenizer.save_pretrained(run/'merged')
    checks = dict(reload_max_absolute_difference=reload_diff, fp32_merge_relative_l2=relative_error,
                  bf16_direct_merge_relative_l2=bf16_relative_error,
                  bf16_direct_merge_passed_002=bf16_relative_error < .02,
                  merge_answer_disagreements=disagreements, probe_split='validation',
                  fp32_merge_tolerance=1e-4, exported_dtype='float32', test_examples=len(splits['test']))
    (run/'artifact_checks.json').write_text(json.dumps(checks, indent=2))
    # Model licensing stays with the model artifacts; weights are never committed to this repo.
    for name in ['LICENSE', 'README.md']:
        if (path/name).exists():
            (run/'merged'/('BASE_MODEL_'+name)).write_bytes((path/name).read_bytes())
    del merged,precise,probe,expected,reference32,combined32
    gc.collect(); torch.cuda.empty_cache()
    return manifest, checks

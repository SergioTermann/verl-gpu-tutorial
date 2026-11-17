import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

root = Path(os.environ['VERL_ADV12_ROOT'])
model_dir = root / 'model'
model_dir.mkdir(parents=True, exist_ok=True)
random.seed(12012)
torch.manual_seed(12012)

tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'user', 'assistant',
          'is_even', 'answer', 'yes', 'no'] + [str(i) for i in range(10)]
tokens += [f'extra{i}' for i in range(32 - len(tokens))]
raw = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]',
                                    bos_token='[BOS]', eos_token='[EOS]')
tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
tokenizer.save_pretrained(model_dir)
config = Qwen2Config(vocab_size=len(tokens), hidden_size=128, intermediate_size=256,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=64, pad_token_id=0, bos_token_id=2,
                     eos_token_id=3, tie_word_embeddings=True)
model = Qwen2ForCausalLM(config).to(torch.bfloat16)
model.save_pretrained(model_dir)

def row(number):
    prompt = f'is_even {number // 10} {number % 10} answer yes or no'
    target = 'yes' if number % 2 == 0 else 'no'
    return dict(data_source='decimal_parity',
                prompt=[dict(role='user', content=prompt)],
                reward_model=dict(style='rule', ground_truth=target),
                extra_info=dict(number=number, split=''))

all_numbers = list(range(100))
rng = random.Random(12012)
for attempt in range(1000):
    rng.shuffle(all_numbers)
    train_numbers, val_numbers, test_numbers = all_numbers[:70], all_numbers[70:80], all_numbers[80:]
    if {n % 10 for n in train_numbers} == set(range(10)):
        break
else:
    raise RuntimeError('Could not create a split covering every ones digit')

def save_split(name, numbers):
    rows = [row(n) for n in numbers]
    for item in rows:
        item['extra_info']['split'] = name
    pd.DataFrame(rows).to_parquet(root / f'{name}.parquet', index=False)
    return rows

train_rows = save_split('train', train_numbers)
val_rows = save_split('val', val_numbers)
test_rows = save_split('test', test_numbers)
assert not (set(train_numbers) & set(val_numbers) or set(train_numbers) & set(test_numbers)
            or set(val_numbers) & set(test_numbers))
(root / 'splits.json').write_text(json.dumps(dict(train=train_numbers, validation=val_numbers,
                                                  test=test_numbers), indent=2))
(root / 'test.json').write_text(json.dumps(test_rows, ensure_ascii=False, indent=2))
print(f'Prepared {sum(p.numel() for p in model.parameters()):,}-parameter Qwen2; '
      f'train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)}')

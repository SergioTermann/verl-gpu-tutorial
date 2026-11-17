from pathlib import Path
import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

root = Path('/opt/verl-smoke')
model_dir = root / 'model'
model_dir.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'user', 'assistant', 'count', 'one', 'two', 'three']
tokens += [f'w{i}' for i in range(256 - len(tokens))]
raw = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]', bos_token='[BOS]', eos_token='[EOS]')
tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
tokenizer.save_pretrained(model_dir)
config = Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                    pad_token_id=0, bos_token_id=2, eos_token_id=3, tie_word_embeddings=True)
model = Qwen2ForCausalLM(config).to(torch.bfloat16)
model.save_pretrained(model_dir)
rows = [dict(data_source='local_smoke', prompt=[dict(role='user', content=f'count one two three w{i}')],
             ability='smoke_test', reward_model=dict(style='rule', ground_truth='unused'),
             extra_info=dict(index=i, split='train')) for i in range(4)]
pd.DataFrame(rows).to_parquet(root / 'train.parquet')
pd.DataFrame(rows[:2]).to_parquet(root / 'val.parquet')
print(f'Prepared random Qwen2: {sum(p.numel() for p in model.parameters()):,} parameters, 4 synthetic prompts')

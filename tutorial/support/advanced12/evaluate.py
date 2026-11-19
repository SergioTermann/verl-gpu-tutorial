"""Evaluate the fixed final checkpoint, retaining unsuccessful learning evidence."""
import json
import re
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

def evaluate(root):
    root=Path(root)
    tokenizer=AutoTokenizer.from_pretrained(root/'model')
    model=Qwen2ForCausalLM.from_pretrained(root/'model').cuda().eval()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    state=torch.load(root/'checkpoints/global_step_10/actor/model_world_size_1_rank_0.pt',map_location='cpu',weights_only=False)
    split=json.loads((root/'splits.json').read_text())
    assert not(set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test']))
    rows=[]
    for stage in ['before','after']:
        if stage=='after':model.load_state_dict(state,strict=True)
        for name,key in [('train','train'),('validation','validation'),('test','test')]:
            for number in split[key]:
                prompt=f'is_even {number//10} {number%10} answer yes or no'
                text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(text,return_tensors='pt',return_token_type_ids=False).to('cuda')
                with torch.inference_mode():
                    output=model.generate(**inputs,max_new_tokens=2,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
                answer=tokenizer.decode(output[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip().lower()
                target='yes' if number%2==0 else 'no'
                rows.append(dict(stage=stage,split=name,number=number,target=target,answer=answer,valid=answer in {'yes','no'},correct=answer==target))
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'predictions.csv',index=False)
    summary=frame.groupby(['stage','split']).agg(count=('correct','size'),accuracy=('correct','mean'),valid_rate=('valid','mean')).reset_index()
    summary.to_csv(root/'evaluation.csv',index=False)
    log=re.sub(r'\x1b\[[0-9;]*m','',(root/'train.log').read_text())
    metrics=[]
    for line in log.splitlines():
        if 'training/global_step:' in line:
            row={k:float(v) for k,v in re.findall(r'([\w/@.-]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',line)}
            metrics.append(row)
    assert len(metrics)==10
    pd.DataFrame(metrics).to_csv(root/'metrics.csv',index=False)
    differences=[(state[k].float()-v.float()).abs().max().item() for k,v in initial.items()]
    report=dict(root=str(root),steps=len(metrics),max_weight_change=max(differences),
        nonzero_gradient_steps=sum(r.get('actor/grad_norm',0)>0 for r in metrics),
        memory_pressure='killed due to memory pressure' in log,
        majority_test_accuracy=max(sum(n%2==0 for n in split['test']),sum(n%2==1 for n in split['test']))/len(split['test']),
        note='No test-based tuning. Random cold start with strict sparse reward; no learning improvement claimed.')
    (root/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    print(json.dumps(evaluate(Path(__file__).with_name('latest_run.txt').read_text().strip()),indent=2))

"""Verify actual FSDP update and saved training state, not just process exit."""
import json
import re
from pathlib import Path
import torch
from transformers import Qwen2ForCausalLM

def verify(root):
    root=Path(root)
    checkpoint=root/'checkpoints/global_step_1/actor'
    weights=checkpoint/'model_world_size_1_rank_0.pt'
    trained=torch.load(weights,map_location='cpu',weights_only=False)
    initial=Qwen2ForCausalLM.from_pretrained(root/'model').state_dict()
    assert set(initial)==set(trained), 'Checkpoint keys differ from initial model'
    changes={}
    for key,before in initial.items():
        after=trained[key].detach().cpu()
        assert torch.isfinite(after).all(),key
        changes[key]=(after.float()-before.float()).abs().max().item()
    changed=sum(v>0 for v in changes.values())
    assert changed>0,'No model weights changed'
    optim_path=checkpoint/'optim_world_size_1_rank_0.pt'
    optimizer=torch.load(optim_path,map_location='cpu',weights_only=False)
    assert optimizer['state'],'Optimizer state is empty'
    assert (checkpoint/'extra_state_world_size_1_rank_0.pt').exists()
    assert (root/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()=='1'
    log=re.sub(r'\x1b\[[0-9;]*m','',(root/'train.log').read_text())
    lines=[line for line in log.splitlines() if 'training/global_step:1' in line]
    assert lines,'Missing completed step metrics'
    metrics={key:float(value) for key,value in re.findall(r'([A-Za-z_][\w/]*):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',lines[-1])}
    assert metrics['actor/grad_norm']>0
    result=dict(run_root=str(root),changed_tensors=changed,total_tensors=len(changes),
        max_abs_weight_change=max(changes.values()),metrics=metrics,
        checkpoint_files=[str(p.relative_to(root)) for p in checkpoint.iterdir() if p.is_file()],
        ray_memory_pressure='killed due to memory pressure' in log,
        disclosure='Synthetic byte-sum reward. Pipeline verification only; no task-accuracy claim.')
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    root=Path(__file__).with_name('latest_run.txt').read_text().strip()
    print(json.dumps(verify(root),indent=2))

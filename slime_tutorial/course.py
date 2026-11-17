"""Original course helpers. Upstream slime is fetched separately, not vendored."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REVISION = '79aa41a884ab0bf65222c94418f6357d3c31192b'
REPOSITORY = 'https://github.com/THUDM/slime.git'
FILES = ['train.py', 'slime/utils/types.py', 'slime/rollout/rm_hub/__init__.py',
         'docs/en/get_started/quick_start.md', 'docs/en/get_started/customization.md',
         'docs/en/developer_guide/debug.md']


def source():
    root = Path(os.environ.get('SLIME_SOURCE', str(Path.home()/'.cache/slime-course/source')))
    if not (root/'.git').exists():
        raise FileNotFoundError(f'Clone the pinned upstream source first; expected {root}. See README.md.')
    actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert actual == REVISION, f'Expected {REVISION}, got {actual}; do not silently mix API versions'
    # Windows checkouts may report line-ending-only changes inside WSL. Reject
    # substantive edits while allowing that transport artifact.
    changed = subprocess.run(['git','-C',str(root),'diff','--ignore-space-at-eol','--quiet'])
    assert changed.returncode == 0, 'Upstream checkout has substantive local changes'
    return root


def load_sample():
    root=source()
    sys.path.insert(0,str(root))
    import slime.utils.types as types
    assert Path(types.__file__).resolve()==(root/'slime/utils/types.py').resolve(), 'Another slime checkout was imported; restart kernel'
    return types.Sample


async def strict_reward(args, sample, **kwargs):
    """Single-sample --custom-rm-path interface: strict yes/no after normalization."""
    if sample.label not in {'yes','no'}:
        raise ValueError('Invalid label: dataset error must not silently become zero reward')
    answer=sample.response.strip().lower()
    return float(answer in {'yes','no'} and answer==sample.label)


async def batch_reward(args, samples, **kwargs):
    """Separate --group-rm entry point; preserves input ordering."""
    return [await strict_reward(args,s,**kwargs) for s in samples]


def audit(sample):
    n=sample.response_length
    assert 0 < n <= len(sample.tokens), 'Response length must fit the token sequence'
    assert sample.loss_mask is not None and len(sample.loss_mask)==n, 'Loss mask must match response length'
    assert all(x in {0,1} for x in sample.loss_mask), 'Mask must contain zero/one'
    assert sample.rollout_log_probs is not None and len(sample.rollout_log_probs)==n, 'Log probabilities must align with response positions'
    return dict(prompt_tokens=len(sample.tokens)-n,response_tokens=n,loss_tokens=sum(sample.loss_mask))


def source_report():
    root=source()
    return dict(repository=REPOSITORY,revision=REVISION,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES})


def function_signature(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return dict(name=name,line=node.lineno,arguments=[a.arg for a in node.args.args],asynchronous=isinstance(node,ast.AsyncFunctionDef))


def save_report(name,report):
    out=Path('results');out.mkdir(exist_ok=True)
    (out/name).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

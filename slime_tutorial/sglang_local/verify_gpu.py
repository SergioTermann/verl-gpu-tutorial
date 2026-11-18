"""GPU smoke test; requires the compatible environment described in README.md."""
import argparse
import asyncio
import importlib.metadata as metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('sglang-verification.json'))
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    import sglang as sgl

    assert args.model.is_dir(), 'Provide a local model directory'
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    engine = None
    loop = None
    try:
        engine = sgl.Engine(model_path=str(args.model.resolve()), dtype='bfloat16',
            mem_fraction_static=.4, context_length=256, disable_cuda_graph=True,
            attention_backend='torch_native', sampling_backend='pytorch', max_running_requests=4)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = engine.generate('The capital of France is', {'temperature': 0, 'max_new_tokens': 16})
        assert result['text'].strip(), 'Empty generation'
        report = dict(sglang=metadata.version('sglang'), sgl_kernel=metadata.version('sgl-kernel'),
            torch=torch.__version__, gpu=torch.cuda.get_device_name(0), response=result['text'],
            scope='GPU generation smoke test, not an answer-quality or slime-training benchmark.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if engine is not None:
            engine.shutdown()
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)


if __name__ == '__main__':
    main()

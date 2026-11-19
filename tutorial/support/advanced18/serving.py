"""Loopback-only, owned vLLM server and an auditable streaming benchmark."""
import concurrent.futures
from importlib.metadata import version
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class LocalServer:
    def __init__(self, model, log_dir, port=18018):
        self.model = Path(model).resolve()
        self.log_dir = Path(log_dir)
        self.port = port
        self.url = f'http://127.0.0.1:{port}'
        self.process = None
        self.log = None

    def start(self, timeout=240):
        assert self.model.is_dir(), 'Merged model is missing; complete lesson 17 first'
        # Refuse an occupied port, rather than connecting to or terminating an unrelated service.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.port))
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log = (self.log_dir/'server.log').open('w')
        command = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(self.model), '--served-model-name', 'course-router',
            '--host', '127.0.0.1', '--port', str(self.port), '--max-model-len', '256',
            '--dtype', 'bfloat16', '--generation-config', 'vllm',
            '--gpu-memory-utilization', '.35', '--max-num-seqs', '16',
            '--enforce-eager', '--no-enable-prefix-caching', '--disable-log-requests']
        env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false')
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
            session = requests.Session(); session.trust_env = False
            deadline = time.monotonic()+timeout
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'vLLM exited with {self.process.returncode}; read {self.log_dir}/server.log')
                try:
                    if session.get(self.url+'/health', timeout=1).status_code==200:
                        return self
                except requests.RequestException:
                    pass
                time.sleep(1)
            raise TimeoutError(f'Server did not become healthy in {timeout}s; inspect server.log')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=20)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
            finally:
                self.process = None
        if self.log is not None:
            self.log.close(); self.log = None


def stream_one(url, messages, row):
    """First nonempty text chunk is measured; SSE chunk count is NOT token count."""
    started = time.perf_counter(); first = None; last = None
    pieces=[]; timestamps=[]; usage=None; finish=None
    session=requests.Session();session.trust_env=False
    try:
        with session.post(url+'/v1/chat/completions', json=dict(model='course-router',messages=messages,
            temperature=0,max_tokens=12,stream=True,stream_options={'include_usage':True}),
            stream=True,timeout=(10,60)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if not line or not line.startswith('data: '): continue
                payload=line[6:]
                if payload=='[DONE]': break
                item=json.loads(payload)
                if item.get('usage') is not None:usage=item['usage']
                for choice in item.get('choices',[]):
                    content=choice.get('delta',{}).get('content')
                    if content:
                        now=time.perf_counter()
                        first=now if first is None else first
                        last=now; timestamps.append(now);pieces.append(content)
                    if choice.get('finish_reason') is not None: finish=choice['finish_reason']
        ended=time.perf_counter();answer=''.join(pieces).strip()
        assert first is not None and usage is not None and finish is not None, 'Incomplete streaming response'
        tokens=int(usage['completion_tokens'])
        return dict(id=row['id'],group=row['group'],target=row['target'],answer=answer,
            valid=answer in {'route=R0','route=R1','route=R2'},correct=answer==row['target'],
            success=True,error='',finish_reason=finish,completion_tokens=tokens,
            prompt_tokens=int(usage['prompt_tokens']),ttft_ms=(first-started)*1000,
            e2e_ms=(ended-started)*1000,text_chunks=len(pieces),
            mean_chunk_gap_ms=float(np.mean(np.diff(timestamps))*1000) if len(timestamps)>1 else None,
            tpot_estimate_ms=(last-first)*1000/(tokens-1) if tokens>1 else None)
    except Exception as exc:
        return dict(id=row['id'],group=row['group'],target=row['target'],answer=''.join(pieces),
            valid=False,correct=False,success=False,error=f'{type(exc).__name__}: {exc}',
            e2e_ms=(time.perf_counter()-started)*1000)
    finally:
        session.close()


def benchmark(server, rows, make_messages, output, repeats=3):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    warmup=stream_one(server.url,make_messages(rows[0]),rows[0])
    assert warmup['success'], f'Warmup failed: {warmup}'
    records=[];windows=[]
    # Rotate order across repetitions to reduce a fixed warm-up/order confound.
    for repeat in range(repeats):
        order=[1,2,4]
        order=order[repeat%3:]+order[:repeat%3]
        for concurrency in order:
            started=time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures=[pool.submit(stream_one,server.url,make_messages(row),row) for row in rows]
                batch=[f.result() for f in futures]
            duration=time.perf_counter()-started
            for record in batch: record.update(repeat=repeat,concurrency=concurrency)
            records.extend(batch)
            windows.append(dict(repeat=repeat,concurrency=concurrency,requests=len(batch),
                elapsed_s=duration,successful=sum(r['success'] for r in batch),
                output_tokens=sum(r.get('completion_tokens',0) for r in batch),
                output_tokens_per_second=sum(r.get('completion_tokens',0) for r in batch)/duration))
    frame=pd.DataFrame(records);frame.to_csv(output/'requests.csv',index=False)
    pd.DataFrame(windows).to_csv(output/'windows.csv',index=False)
    summary=[]
    for concurrency, group in frame.groupby('concurrency'):
        good=group[group.success]
        matching=[w for w in windows if w['concurrency']==concurrency]
        summary.append(dict(concurrency=int(concurrency),requests=len(group),
            errors=int((~group.success).sum()),accuracy_including_errors=float(group.correct.mean()),
            ttft_p50_ms=float(good.ttft_ms.quantile(.5)) if len(good) else None,
            ttft_p95_ms=float(good.ttft_ms.quantile(.95)) if len(good) else None,
            e2e_p50_ms=float(good.e2e_ms.quantile(.5)) if len(good) else None,
            e2e_p95_ms=float(good.e2e_ms.quantile(.95)) if len(good) else None,
            output_tokens_per_second=sum(w['output_tokens'] for w in matching)/sum(w['elapsed_s'] for w in matching)))
    result=pd.DataFrame(summary);result.to_csv(output/'summary.csv',index=False)
    (output/'protocol.json').write_text(json.dumps(dict(repeats=repeats,concurrencies=[1,2,4],
        warmup_requests=1,prefix_cache=False,temperature=0,max_tokens=12,host='127.0.0.1',
        dtype='bfloat16',exported_artifact_dtype='float32',vllm=version('vllm'),
        model_max_length=256,gpu_memory_utilization=.35,scope='Local short-output closed-loop HTTP benchmark.',
        latency_definition='Client HTTP request start to first nonempty text chunk / response completion; executor waiting excluded.',
        tpot_definition='Estimate from first/last text chunk and reported completion_tokens; chunks may contain multiple tokens and EOS may be counted.'),indent=2))
    return result

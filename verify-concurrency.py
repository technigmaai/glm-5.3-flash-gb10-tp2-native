#!/usr/bin/env python3
"""Exercise the configured request limit and record actual scheduler occupancy."""
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.request

BASE = os.environ.get('VERIFY_BASE_URL', 'http://127.0.0.1:8000')
METRICS = os.environ.get('VERIFY_METRICS_URL', 'http://127.0.0.1:8000/metrics')
PROMPTS = [
    'Write a complete red-black tree in Python with insert and rebalancing. Code only.',
    'Explain in detail how a hash map handles collisions, with examples.',
    'Write a Python LRU cache implementation with a linked list. Code only.',
    'Explain how database transactions and isolation levels work, with examples.',
]
ALIASES = os.environ.get('VERIFY_MODEL_ALIASES', 'glm53 nvidia/GLM-5.3-Flash-NVFP4').split()
CONCURRENCY = int(os.environ.get('VERIFY_CONCURRENCY', '4'))
barrier = threading.Barrier(CONCURRENCY)
finished = threading.Event()
peak_running = 0
metric_errors = []


def metrics():
    with urllib.request.urlopen(METRICS, timeout=5) as response:
        raw = response.read().decode()
    def value(name):
        return sum(float(n) for n in re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+([0-9.e+]+)$', raw, re.M))
    return value('num_requests_running'), value('num_preemptions_total')


def sample():
    global peak_running
    while not finished.is_set():
        try:
            running, _ = metrics()
            peak_running = max(peak_running, running)
        except Exception as error:
            metric_errors.append(str(error))
        finished.wait(0.25)


def request(i):
    body = json.dumps({
        'model': ALIASES[i % len(ALIASES)], 'max_tokens': 256, 'temperature': 0, 'stream': True,
        'chat_template_kwargs': {'enable_thinking': False},
        'stream_options': {'include_usage': True},
        'messages': [{'role': 'user', 'content': PROMPTS[i % len(PROMPTS)]}],
    }).encode()
    barrier.wait(timeout=30)
    started = time.monotonic()
    usage = None
    chunks = 0
    ended = False
    with urllib.request.urlopen(urllib.request.Request(BASE + '/v1/chat/completions', data=body,
                                headers={'Content-Type': 'application/json'}), timeout=600) as response:
        for line in response:
            if line.strip() == b'data: [DONE]':
                ended = True
            if not line.startswith(b'data: {'):
                continue
            event = json.loads(line[6:])
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('choices'):
                chunks += 1
            if event.get('usage'):
                usage = event['usage']
    assert ended and chunks and usage and usage['completion_tokens'] > 0, (i, ended, chunks, usage)
    return {'model': ALIASES[i % len(ALIASES)], 'elapsed_seconds': round(time.monotonic()-started, 2), 'usage': usage}


_, preemptions_before = metrics()
monitor = threading.Thread(target=sample)
monitor.start()
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(request, range(CONCURRENCY)))
finally:
    finished.set()
    monitor.join()
_, preemptions_after = metrics()
report = {'peak_running_requests': peak_running, 'preemptions': preemptions_after-preemptions_before,
          'requests': results, 'metrics_errors': metric_errors}
print(json.dumps(report, indent=2), flush=True)
assert peak_running == CONCURRENCY, f'Expected {CONCURRENCY} simultaneous requests, observed {peak_running}'
assert preemptions_after == preemptions_before, 'Scheduler preempted during the four-stream test'

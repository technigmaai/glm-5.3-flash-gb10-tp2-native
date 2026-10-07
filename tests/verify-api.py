#!/usr/bin/env python3
import concurrent.futures
import json
import os
import time
import urllib.request

ALIASES = os.environ.get('VERIFY_MODEL_ALIASES', 'glm53 nvidia/GLM-5.3-Flash-NVFP4').split()
BASE = os.environ.get('VERIFY_BASE_URL', 'http://127.0.0.1:8000')
with urllib.request.urlopen(BASE + '/v1/models', timeout=10) as response:
    models = {m['id'] for m in json.load(response)['data']}
assert set(ALIASES) <= models, f'Missing aliases: {set(ALIASES) - models}'


def chat(alias):
    request = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps({
        'model': alias, 'messages': [{'role': 'user', 'content': 'What is 17 multiplied by 23? Answer with the number.'}],
        'temperature': 0, 'max_tokens': 512, 'reasoning_effort': 'low',
    }).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    assert '391' in body['choices'][0]['message']['content'], (alias, body)
    assert body['choices'][0]['finish_reason'] == 'stop', (alias, body)
    return {'alias': alias, 'usage': body['usage'], 'passed': True}


with concurrent.futures.ThreadPoolExecutor(max_workers=len(ALIASES)) as pool:
    results = list(pool.map(chat, ALIASES))
print(json.dumps({'checked_at': time.time(), 'concurrent_alias_checks': results}, indent=2))

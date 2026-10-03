#!/usr/bin/env python3
"""Lightweight probe; worker health follows its spawned native process."""
import os
from pathlib import Path
import urllib.request

if os.environ['ROLE'] == 'head':
    with urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('API_PORT', '8000') + '/health', timeout=3) as response:
        assert response.status == 200
    assert Path('/tmp/glm53-stage').read_text().splitlines()[-1] == 'serving'
else:
    workers = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / 'cmdline').read_bytes()
            if b'VLLM::Worker_TP' in command:
                workers.append(proc.name)
        except (FileNotFoundError, PermissionError):
            pass
    assert workers, 'No native TP worker process is present'

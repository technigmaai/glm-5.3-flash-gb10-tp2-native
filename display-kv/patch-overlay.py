#!/usr/bin/env python3
"""Port the allocator at its only backing-buffer allocation; leave upstream untouched."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'experimental/fixes/worker_utils.py'
EXPECTED = '44cd0ab144cdbaa896041e5f8af56f510faab537d4d95afb5f6d1257e45ad154'
NEEDLE = '    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)\n'
text = SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED, 'Upstream allocator changed; re-audit this port'
assert text.count(NEEDLE) == 1
text = text.replace(NEEDLE, '    from glm53_display_kv import allocate_display_backed_kv\n'
                    '    buf = allocate_display_backed_kv(buf_size, dtype=torch.int8, device=device)\n')
ast.parse(text)
(HERE / 'worker_utils.py').write_text(text)
print('Ported the exact c748079 fixes overlay allocator; upstream file unchanged')

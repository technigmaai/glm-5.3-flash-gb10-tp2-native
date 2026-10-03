#!/usr/bin/env python3
"""Exercise exact KV storage, CUDA views, boundary writes, and graph replay without weights."""
import json
import os
os.environ['GLM53_DISPLAY_KV_ENABLE'] = '1'
import torch
from glm53_display_kv import allocate_display_backed_kv, _owners, DISPLAY_BYTES

torch.cuda.set_device(0)
torch.cuda.init()
size = 4 * 2**30 + 4096  # also test the exact logical length with a 64KiB alignment tail
buf = allocate_display_backed_kv(size, torch.int8, torch.device('cuda:0'))
owner = _owners[0]
assert buf.numel() == size and buf.untyped_storage().nbytes() == size
assert buf.data_ptr() == owner.pointer
boundary = owner.ordinary_bytes
slices = [buf[:4096], buf[boundary-4096:boundary+4096], buf[-4096:]]
for i, view in enumerate(slices, 1):
    view.fill_(i * 7)
torch.cuda.synchronize()
for i, view in enumerate(slices, 1):
    assert bool((view == i * 7).all()), f'Boundary write failed for view {i}'
float_view = buf[boundary-4096:boundary+4096].view(torch.float32)
float_view.fill_(1.5)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    float_view.add_(1)
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    float_view.add_(1)
graph.replay()
graph.replay()  # capture records operations; the two replays execute them
torch.cuda.synchronize()
assert bool((float_view == 4.5).all()), 'CUDA graph replay failed at memory boundary'
assert owner.size >= size and owner.size-size < 65536
print(json.dumps({'status':'PASS', 'torch':torch.__version__, 'logical_bytes':size,
                  'ordinary_bytes':owner.ordinary_bytes, 'display_bytes':DISPLAY_BYTES,
                  'tested':['exact-size storage','shared UVA pointer','ordinary/display boundary','float32 views','CUDA graph replay']}))

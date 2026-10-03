# SPDX-License-Identifier: AGPL-3.0-only
"""Display-reserved KV backing ported to the pinned Kindling nightly allocator."""
from __future__ import annotations

import ctypes
import json
import os
import threading
from pathlib import Path

DISPLAY_BYTES = 1792 * 2**20
QUANTUM = 65536
MAX_ORDINARY_BYTES = 16 * 2**30
MIN_ALLOCATION_BYTES = int(os.environ.get("GLM53_DISPLAY_KV_MIN_BYTES", 4 * 2**30))
LIBRARY = Path("/opt/glm53-display-kv/libglm53_display_kv.so")
_owners: list["Owner"] = []
_lock = threading.Lock()


def _memory() -> dict[str, int]:
    wanted = {"MemFree", "MemAvailable"}
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key = line.split(":", 1)[0]
        if key in wanted:
            result[key] = int(line.split()[1]) * 1024
    return result


class Owner:
    def __init__(self, ordinary_bytes: int, logical_bytes: int):
        import torch

        if not torch.cuda.is_initialized() or torch.cuda.current_device() != 0:
            raise RuntimeError("Display KV requires an initialized CUDA:0 context")
        if ordinary_bytes < 0 or ordinary_bytes > MAX_ORDINARY_BYTES:
            raise ValueError(f"ordinary display-KV prefix out of range: {ordinary_bytes}")
        if ordinary_bytes % QUANTUM:
            raise ValueError("ordinary display-KV prefix is not 64KiB aligned")

        self.lib = ctypes.CDLL(str(LIBRARY))
        self.lib.ds41_display_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self.lib.ds41_display_create.restype = ctypes.c_void_p
        self.lib.ds41_display_pointer.argtypes = [ctypes.c_void_p]
        self.lib.ds41_display_pointer.restype = ctypes.c_uint64
        self.lib.ds41_display_error.restype = ctypes.c_char_p
        self.handle = self.lib.ds41_display_create(ordinary_bytes, DISPLAY_BYTES)
        if not self.handle:
            raise RuntimeError(self.lib.ds41_display_error().decode())
        self.pointer = self.lib.ds41_display_pointer(self.handle)
        self.ordinary_bytes = ordinary_bytes
        self.size = ordinary_bytes + DISPLAY_BYTES
        if logical_bytes <= 0 or logical_bytes > self.size:
            raise ValueError("logical display-KV size exceeds the reserved allocation")
        self.logical_bytes = logical_bytes
        self.__cuda_array_interface__ = {
            # Expose only the exact vLLM descriptor size. The physical mapping
            # may contain a short alignment tail, but vLLM requires the tensor
            # storage size to be exactly divisible by its serving block count.
            "shape": (self.logical_bytes,),
            "strides": None,
            "typestr": "|i1",
            "data": (self.pointer, False),
            "version": 3,
        }

    def tensor(self):
        import torch

        tensor = torch.as_tensor(self, device="cuda:0")
        if tensor.data_ptr() != self.pointer or tensor.dtype != torch.int8:
            raise RuntimeError("CUDA array interface copied display-KV storage")
        return tensor


def allocate_display_backed_kv(size: int, dtype, device):
    import torch

    if os.environ.get("GLM53_DISPLAY_KV_ENABLE", "0") != "1":
        return torch.zeros(size, dtype=dtype, device=device)
    if size < MIN_ALLOCATION_BYTES:
        return torch.zeros(size, dtype=dtype, device=device)
    if dtype != torch.int8 or torch.device(device) != torch.device("cuda:0"):
        raise RuntimeError(f"unsupported display-KV allocation dtype={dtype} device={device}")

    with _lock:
        if _owners:
            raise RuntimeError("display-KV final backing was allocated more than once")
        ordinary_needed = max(0, size - DISPLAY_BYTES)
        ordinary_bytes = ((ordinary_needed + QUANTUM - 1) // QUANTUM) * QUANTUM
        before = _memory()
        owner = Owner(ordinary_bytes, size)
        # The registered ordinary prefix is outside PyTorch's caching allocator.
        # Reserve it inside the upstream worker cap instead of silently allowing
        # the same bytes again as ordinary torch allocations.
        original_fraction = float(os.environ.get("TORCH_MEM_FRACTION", "0.92"))
        total_memory = torch.cuda.get_device_properties(0).total_memory
        adjusted_fraction = original_fraction - ordinary_bytes / total_memory
        if not 0 < adjusted_fraction < 1:
            raise RuntimeError("display-KV ordinary prefix exceeds the worker memory budget")
        torch.cuda.set_per_process_memory_fraction(adjusted_fraction, 0)
        full = owner.tensor()
        if full.numel() != size or full.untyped_storage().nbytes() != size:
            raise RuntimeError("display-KV tensor does not expose the exact requested storage size")
        _owners.append(owner)
        result = full
        result.zero_()
        torch.cuda.synchronize()
        receipt = {
            "stage": "glm53_kindling_display_kv_allocated",
            "logical_bytes": size,
            "ordinary_bytes": ordinary_bytes,
            "display_bytes": DISPLAY_BYTES,
            "reserved_bytes": owner.size,
            "storage_pointer": owner.pointer,
            "torch_memory_fraction_original": original_fraction,
            "torch_memory_fraction_adjusted": adjusted_fraction,
            "before": before,
            "after": _memory(),
        }
        print(json.dumps(receipt), flush=True)
        return result

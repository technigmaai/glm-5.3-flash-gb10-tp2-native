"""Local-disk snapshot of processed weights, for fast restarts.

The first boot loads the checkpoint as usual and writes, per rank, every tensor
reachable from the processed model. Later boots build the model through the
normal path on dummy weights (seconds: the slow part of a real load is reading
and handling the checkpoint, not processing) and then overwrite every tensor in
place from the snapshot. Building through the normal path keeps every object,
view and kernel exactly as a real load makes them; vLLM's pre-processed rebuild
(weights_already_processed) does not, e.g. it breaks the MLA absorbed weights'
views of kv_b_proj and the router's reference to e_score_correction_bias.

Enabled by VLLM_WEIGHT_SNAPSHOT_DIR. VLLM_WEIGHT_SNAPSHOT_TAG joins the key for
image-level patches that the vLLM version string cannot see. A snapshot is used
only when its key matches exactly and it covers every tensor of the model;
otherwise the rank loads normally and writes a new one. Names start with
SNAPSHOT_VERSION, and opening the directory deletes snapshots of any other
version.
"""
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Bump when a change to weight processing or to this file's format makes
# existing snapshots wrong in a way the key cannot see.
SNAPSHOT_VERSION = 2
# A snapshot or a partly written one, of any version (version 1 had no prefix).
_SNAPSHOT_NAME = re.compile(r"(v\d+-)?(target|draft)-tp\d+of\d+-dp\d+-[0-9a-f]{16}(\.tmp\d+)?")

_ALIGN = 4096
_CHUNK = 64 << 20  # bytes per read; two pinned buffers of this size


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name.removeprefix("torch."))


class Snapshot:
    def __init__(self, path: str, key: dict):
        self.path = path
        self.key = key

    # ---- key ----------------------------------------------------------------

    @classmethod
    def for_model(cls, vllm_config, model_config) -> "Snapshot | None":
        """The snapshot slot for this rank and model, or None when disabled.

        Must run before weights load: processing may mutate the quant config
        that the key hashes.
        """
        base = os.environ.get("VLLM_WEIGHT_SNAPSHOT_DIR")
        if not base:
            return None
        from vllm.distributed import (
            get_dp_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.model_loader.weight_cache.protocol import WeightCacheKey, hash_checkpoint
        from vllm.model_executor.model_loader.model_source import snapshot_fingerprint_config

        spec = vllm_config.speculative_config
        is_draft = spec is not None and model_config is spec.draft_model_config
        dp = get_dp_group()
        key = asdict(
            WeightCacheKey.from_model_config(
                snapshot_fingerprint_config(model_config, hash_checkpoint),
                tp_size=get_tensor_model_parallel_world_size(),
                tp_rank=get_tensor_model_parallel_rank(),
                is_draft=is_draft,
                dp_size=dp.world_size,
                dp_rank=dp.rank_in_group,
            )
        )
        key["tag"] = os.environ.get("VLLM_WEIGHT_SNAPSHOT_TAG", "")
        # The indexer's top-k scratch buffer has one row per batched token.
        key["max_num_batched_tokens"] = vllm_config.scheduler_config.max_num_batched_tokens
        # The MoE backend decides the processed expert layout (marlin repacks
        # and drops the input scales flashinfer_cutlass keeps), so a snapshot
        # from one backend cannot load under another.
        key["moe_backend"] = str(getattr(vllm_config.kernel_config, "moe_backend", ""))
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        role = "draft" if is_draft else "target"
        name = f"v{SNAPSHOT_VERSION}-{role}-tp{key['tp_rank']}of{key['tp_size']}-dp{key['dp_rank']}-{digest}"
        _remove_other_versions(base)
        return cls(_native_snapshot_path(base, name), key)

    def complete(self) -> bool:
        return os.path.isfile(os.path.join(self.path, "COMPLETE"))

    # ---- save ---------------------------------------------------------------

    def save(self, model: nn.Module) -> None:
        """Write every tensor reachable from the processed model, by path."""
        t0 = time.monotonic()
        tmp = self.path + f".tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        entries, aliases, canonical = [], {}, {}
        offset = 0
        with open(os.path.join(tmp, "data.bin"), "wb") as f:
            # Parameters and buffers first, so they are the entries and the
            # caches that alias them (which a dummy build lacks) the aliases.
            for path, t in sorted(_tensors(model).items(), key=lambda kv: "|" in kv[0]):
                # A view that repeats one already written (same bytes, same
                # layout) is restored by writing that one.
                ident = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape),
                         tuple(t.stride()), t.dtype, t.device.type)
                if ident in canonical:
                    aliases[path] = canonical[ident]
                    continue
                canonical[ident] = path
                flat = t.detach().contiguous().reshape(-1).view(torch.uint8)
                nbytes = flat.numel()
                for s in range(0, nbytes, _CHUNK):
                    f.write(memoryview(flat[s:s + _CHUNK].cpu().numpy()))
                pad = -nbytes % _ALIGN
                if pad:
                    f.write(b"\0" * pad)
                entries.append({"path": path, "dtype": str(t.dtype), "shape": list(t.shape),
                                "offset": offset, "nbytes": nbytes})
                offset += nbytes + pad
        manifest = {"key": self.key, "entries": entries, "aliases": aliases,
                    "module_attrs": _module_flags(model), "bytes": offset}
        with open(os.path.join(tmp, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        open(os.path.join(tmp, "COMPLETE"), "w").close()
        shutil.rmtree(self.path, ignore_errors=True)
        os.rename(tmp, self.path)
        logger.info("Weight snapshot: wrote %d tensors (+%d views), %.1f GiB to %s in %.1f s",
                    len(entries), len(aliases), offset / 2**30, self.path, time.monotonic() - t0)

    # ---- restore ------------------------------------------------------------

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Overwrite, in place, every tensor of a model that went through the
        normal load and processing path on dummy weights.

        The normal path builds every object, view and kernel exactly as a real
        load does; only the bytes are dummy, and this replaces all of them.
        Reads with O_DIRECT through two pinned buffers so the next read
        overlaps the previous copy.
        """
        t0 = time.monotonic()
        with open(os.path.join(self.path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["key"] != self.key:  # the digest collided or the file was edited
            raise RuntimeError(f"weight snapshot {self.path} does not match this model")
        live = _tensors(model)
        # Caches that a model's own load_weights hangs on a module (the DFlash
        # drafter's fused KV weight, say) do not exist after a dummy build.
        # Recreate those that sit directly on a module; anything deeper fails.
        modules = dict(model.named_modules(remove_duplicate=False))
        created = 0
        for e in manifest["entries"]:
            path = e["path"]
            if path in live or "|" not in path:
                continue
            module_name, attr = path.split("|", 1)
            module = modules.get(module_name)
            if module is None or "." in attr or hasattr(module, attr):
                continue
            ref = next(iter(live.values()))
            t = torch.empty(e["shape"], dtype=_dtype(e["dtype"]), device=ref.device)
            object.__setattr__(module, attr, t)
            live[path] = t
            created += 1
        missing = [e["path"] for e in manifest["entries"] if e["path"] not in live]
        if missing:
            raise RuntimeError(f"weight snapshot {self.path}: {len(missing)} tensors have no counterpart, "
                               f"e.g. {missing[:3]}; delete the snapshot")
        covered = {e["path"] for e in manifest["entries"]} | set(manifest["aliases"])
        uncovered = [p for p in live if p not in covered]
        if uncovered:
            raise RuntimeError(f"weight snapshot {self.path}: {len(uncovered)} tensors would keep dummy "
                               f"values, e.g. {uncovered[:3]}; delete the snapshot")
        bufs = [torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        views = [memoryview(b.numpy()) for b in bufs]
        done = [torch.cuda.Event(), torch.cuda.Event()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        fd = os.open(os.path.join(self.path, "data.bin"), os.O_RDONLY | os.O_DIRECT)
        turn, late = 0, []
        try:
            for e in manifest["entries"]:
                t = live[e["path"]]
                if str(t.dtype) != e["dtype"] or list(t.shape) != e["shape"]:
                    raise RuntimeError(f"weight snapshot: {e['path']} is {t.dtype}{list(t.shape)}, "
                                       f"snapshot has {e['dtype']}{e['shape']}")
                # Contiguous tensors take the bytes directly; others via a
                # contiguous staging copy, written back once the stream is done.
                dst = t if t.is_contiguous() else torch.empty(t.shape, dtype=t.dtype, device=t.device)
                raw = dst.reshape(-1).view(torch.uint8)
                on_gpu = t.device.type == "cuda"
                n = e["nbytes"]
                for s in range(0, n, _CHUNK):
                    m = min(_CHUNK, n - s)
                    done[turn].synchronize()
                    got = os.preadv(fd, [views[turn][:(m + _ALIGN - 1) // _ALIGN * _ALIGN]], e["offset"] + s)
                    assert got >= m, f"short read in {e['path']}"
                    with torch.cuda.stream(stream):
                        raw[s:s + m].copy_(bufs[turn][:m], non_blocking=on_gpu)
                        done[turn].record(stream)
                    turn ^= 1
                if dst is not t:
                    late.append((t, dst))
            stream.synchronize()
            for t, dst in late:
                t.copy_(dst)
            # An alias is the same memory as its entry in the saving boot. Here
            # it may not be: a recreated cache is new memory, so the parameter
            # it aliased would keep its dummy values without this copy.
            copied = 0
            for path, src in manifest["aliases"].items():
                if path not in live:  # a cache the dummy build lacks; the model rebuilds it
                    continue
                a, s = live[path], live[src]
                if a.data_ptr() != s.data_ptr() or a.stride() != s.stride():
                    a.copy_(s)
                    copied += 1
            torch.cuda.synchronize()
        finally:
            os.close(fd)
        # The dummy build already set every flag from this boot's config;
        # the saved ones come from the boot that wrote the snapshot, whose
        # config (speculation, say) may differ. Report differences only.
        differ = [f"{m}.{k}: {getattr(modules[m], k, None)!r} here, {v!r} in snapshot"
                  for m, attrs in manifest.get("module_attrs", {}).items() if m in modules
                  for k, v in attrs.items() if getattr(modules[m], k, None) != v]
        if differ:
            logger.warning("Weight snapshot: %d module flags differ from the saving boot, e.g. %s",
                           len(differ), differ[:3])
        logger.info("Weight snapshot: restored %d tensors (+%d views, %d copied), %d recreated caches, "
                    "%.1f GiB from %s in %.1f s", len(manifest["entries"]), len(manifest["aliases"]), copied,
                    created, manifest["bytes"] / 2**30, self.path, time.monotonic() - t0)


def _remove_other_versions(base: str) -> None:
    """Delete every snapshot in base whose version is not SNAPSHOT_VERSION.

    Other files in base are left alone. Snapshots of this version for other
    TP layouts or keys stay.
    """
    current = f"v{SNAPSHOT_VERSION}-"
    try:
        names = os.listdir(base)
    except FileNotFoundError:
        return
    for n in names:
        if not _SNAPSHOT_NAME.fullmatch(n) or n.startswith(current):
            continue
        path = os.path.join(base, n)
        try:
            size = os.path.getsize(os.path.join(path, "data.bin"))
        except OSError:
            size = 0
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Weight snapshot: removed %s (%.1f GiB), not version %d", path, size / 2**30, SNAPSHOT_VERSION)


def _module_flags(model: nn.Module) -> dict:
    """Plain public attributes of every module: the flags processing sets."""
    out = {}
    for name, module in model.named_modules():
        attrs = {k: v for k, v in vars(module).items()
                 if not k.startswith("_") and k != "training"
                 and (v is None or isinstance(v, (bool, int, float, str)))}
        if attrs:
            out[name] = attrs
    return out


# ---- verification (VLLM_WEIGHT_SNAPSHOT_VERIFY=1) ------------------------------

_SKIP_ATTRS = ("_modules", "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
               "_backward_pre_hooks", "_state_dict_hooks", "_state_dict_pre_hooks",
               "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks")


def _walk(model: nn.Module, describe: bool = False, roots=()) -> dict:
    """Every tensor and scalar reachable from the model's modules, by path:
    parameters and buffers, and whatever quant methods, kernels and attention
    backends hang off the modules.

    With describe, also records the identity of every function and the type
    of every object, and follows closures, bound methods and partials without
    entering the VllmConfig. verify uses it to find state that save skips."""
    import functools
    import types

    out, seen = {}, set()
    # Registered modules are walked from named_modules; any other module (a MoE
    # kernel held by a quant method, say) is walked where it is found.
    registered = {id(m) for _, m in model.named_modules(remove_duplicate=False)}

    def visit(obj, path, depth):
        if isinstance(obj, torch.Tensor):
            out[path] = obj
            return
        if obj is None or isinstance(obj, (bool, int, float, str)):
            out[path] = obj
            return
        if depth > 8 or id(obj) in seen or id(obj) in registered or isinstance(obj, type):
            return
        if describe and type(obj).__name__ == "VllmConfig" and not path.startswith("config"):
            return
        if isinstance(obj, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
            if not describe:
                return
            seen.add(id(obj))
            if isinstance(obj, types.MethodType):
                out[path] = f"<method {obj.__func__.__qualname__}>"
                visit(obj.__self__, f"{path}.__self__", depth + 1)
            elif isinstance(obj, types.FunctionType):
                out[path] = f"<function {obj.__module__}.{obj.__qualname__}>"
                for i, cell in enumerate(obj.__closure__ or ()):
                    try:
                        visit(cell.cell_contents, f"{path}.<cell{i}>", depth + 1)
                    except ValueError:  # empty cell
                        pass
            else:
                out[path] = f"<builtin {getattr(obj, '__qualname__', obj)!s}>"
            return
        seen.add(id(obj))
        if describe:
            out[f"{path}.__class__"] = type(obj).__qualname__
        if isinstance(obj, dict):
            items = [(repr(k), v) for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            items = [(str(i), v) for i, v in enumerate(obj)]
        elif describe and isinstance(obj, functools.partial):
            items = [("func", obj.func), ("args", obj.args), ("keywords", obj.keywords)]
        elif hasattr(type(obj), "__dict__") and isinstance(getattr(obj, "__dict__", None), dict):
            items = list(obj.__dict__.items())
        else:
            items = []
        if describe:
            for cls in type(obj).__mro__:
                for s in cls.__dict__.get("__slots__", ()):
                    if s in ("__dict__", "__weakref__"):
                        continue
                    try:
                        items.append((s, object.__getattribute__(obj, s)))
                    except Exception:  # unset slot
                        pass
        for k, v in items:
            visit(v, f"{path}.{k}", depth + 1)

    for name, module in model.named_modules(remove_duplicate=False):
        for k, v in vars(module).items():
            if k in _SKIP_ATTRS:
                continue
            if k in ("_parameters", "_buffers"):
                for pk, pv in v.items():
                    if pv is not None:
                        out[f"{name}.{pk}"] = pv
                continue
            visit(v, f"{name}|{k}", 0)
    for path, obj in roots:
        try:
            visit(obj, path, 0)
        except Exception as e:  # lazy-import placeholders and the like
            out[path] = f"<unwalkable {type(e).__name__}>"
    return out


def dump_state(model: nn.Module, vllm_config, path: str) -> None:
    """Write, one line per path, every tensor checksum and scalar reachable
    from the model, the VllmConfig and the vllm and flashinfer module
    globals. Lines hold no addresses, so two boots can be diffed."""
    import functools
    import sys
    import types

    roots = [("config", vllm_config)]
    for mname, mod in sorted(sys.modules.items()):
        if not mname.startswith(("vllm", "flashinfer")) or mod is None:
            continue
        for k, v in sorted(vars(mod).items()):
            if isinstance(v, (types.ModuleType, type)) or k.startswith("__"):
                continue
            if isinstance(v, functools._lru_cache_wrapper):
                roots.append((f"global:{mname}.{k}.cache_size", v.cache_info().currsize))
            if not isinstance(v, (types.FunctionType, types.BuiltinFunctionType)):
                roots.append((f"global:{mname}.{k}", v))
    lines = []
    for p, x in _walk(model, describe=True, roots=roots).items():
        if isinstance(x, torch.Tensor):
            if x.device.type == "meta":
                v = f"meta {x.dtype} {list(x.shape)}"
            else:
                flat = x.detach().contiguous().reshape(-1).view(torch.uint8)
                pad = (-flat.numel()) % 4
                if pad:
                    flat = torch.cat([flat, flat.new_zeros(pad)])
                w = flat.view(torch.int32)
                s = sum(int(c.to(torch.int64).sum()) for c in w.split(1 << 26)) if w.numel() else 0
                v = f"{x.dtype} {list(x.shape)} {x.stride()} {s}"
        else:
            v = repr(x)
            if " at 0x" in v:
                v = v.split(" at 0x")[0] + ">"
        lines.append(f"{p}\t{v}")
    with open(path, "w") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    logger.warning("State dump: %d paths to %s", len(lines), path)


def _tensors(model: nn.Module) -> dict:
    """Every tensor reachable from the model, by path: parameters, buffers,
    and those held by quant methods, kernels and attention backends. Skips the
    shared vllm_config and the KV-cache placeholders bound after loading."""
    return {p: x for p, x in _walk(model).items()
            if isinstance(x, torch.Tensor) and x.device.type != "meta"
            and "vllm_config" not in p and not p.endswith("|kv_cache")}


def verify(snap_model: nn.Module, ref_model: nn.Module) -> None:
    """Log every tensor or scalar that differs between a snapshot-built model
    and a normally loaded one."""
    a, b = _walk(snap_model, describe=True), _walk(ref_model, describe=True)
    bad = []
    # vllm_config is shared by both models and points into the snapshot's own
    # layer registry, so everything under it is noise here.
    a = {k: v for k, v in a.items() if "vllm_config" not in k}
    b = {k: v for k, v in b.items() if "vllm_config" not in k}
    for path in sorted(set(a) | set(b)):
        if path.endswith("|kv_cache"):  # bound later, when the KV cache is allocated
            continue
        if path not in a or path not in b:
            bad.append(f"{path}: only in {'snapshot' if path in a else 'reference'}")
            continue
        x, y = a[path], b[path]
        if isinstance(x, torch.Tensor) != isinstance(y, torch.Tensor):
            bad.append(f"{path}: tensor vs {type(y).__name__}")
        elif isinstance(x, torch.Tensor):
            if x.device.type == "meta" or y.device.type == "meta":
                bad.append(f"{path}: meta tensor ({x.device.type} vs {y.device.type})")
            elif x.dtype != y.dtype or x.shape != y.shape:
                bad.append(f"{path}: {x.dtype}{list(x.shape)} vs {y.dtype}{list(y.shape)}")
            elif x.numel() and not torch.equal(x.contiguous().reshape(-1).view(torch.uint8).to(y.device),
                                               y.contiguous().reshape(-1).view(torch.uint8)):
                d = (x.float().to(y.device) - y.float()).abs()
                bad.append(f"{path}: {x.dtype}{list(x.shape)} values differ, max |d| {d.max().item():.3e}, "
                           f"{(d != 0).float().mean().item() * 100:.1f}% of elements")
        elif x != y:
            bad.append(f"{path}: {x!r} vs {y!r}")
    # Layout, which the byte comparison above cannot see: strides, and which
    # tensors share one storage (a kernel may read one past its neighbour).
    def groups(walk):
        g = {}
        for path, x in walk.items():
            if isinstance(x, torch.Tensor) and x.device.type != "meta" and "vllm_config" not in path:
                g.setdefault(x.untyped_storage().data_ptr(), []).append(path)
        return {tuple(sorted(v)) for v in g.values() if len(v) > 1}
    ga, gb = groups(a), groups(b)
    for grp in sorted(gb - ga):
        bad.append(f"LAYOUT: reference shares one storage across {len(grp)} tensors, snapshot does not: {list(grp)[:4]}")
    for grp in sorted(ga - gb):
        bad.append(f"LAYOUT: snapshot shares one storage across {len(grp)} tensors, reference does not: {list(grp)[:4]}")
    for path in sorted(set(a) & set(b)):
        x, y = a[path], b[path]
        if (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and "vllm_config" not in path
                and x.device.type != "meta" and y.device.type != "meta" and x.shape == y.shape
                and (x.stride() != y.stride() or x.storage_offset() != y.storage_offset())):
            bad.append(f"LAYOUT: {path}: stride {x.stride()} offset {x.storage_offset()} vs "
                       f"stride {y.stride()} offset {y.storage_offset()}")
    bad.sort(key=lambda s: not s.startswith("LAYOUT"))  # layout findings first
    logger.warning("Weight snapshot verify: %d paths compared, %d differ", len(set(a) | set(b)), len(bad))
    for line in bad[:2000]:
        logger.warning("  snapshot differs: %s", line)


def _native_snapshot_path(base, name):
    local = os.path.join(base, name)
    seed = os.environ.get("VLLM_WEIGHT_SNAPSHOT_SEED_DIR", "")
    if not seed or os.environ.get("VLLM_WEIGHT_SNAPSHOT_NORMAL") == "1":
        return local
    if os.path.isfile(os.path.join(local, "COMPLETE")):
        return local
    candidate = os.path.join(seed, name)
    if os.path.isfile(os.path.join(candidate, "COMPLETE")):
        return candidate
    return local

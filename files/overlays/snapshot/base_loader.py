# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.reload import finalize_layerwise_processing
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


def _optional(module: str, *names: str):
    """`module`, or a stand-in whose `names` do nothing when it is not installed."""
    import importlib
    from types import SimpleNamespace

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return SimpleNamespace(**{n: (lambda *args, **kwargs: None) for n in names})


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def create_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Create a model with the given configurations."""
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
        )
        log_online_quantization(vllm_config)
        log_model_inspection(model)
        return model

    @instrument(span_name="Load model")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Load a model with the given configurations."""
        # Local weight snapshot (weight_snapshot.py): rebuild from it when
        # this rank has one, otherwise load normally and write one.
        # fp8.yaml and megamoe.yaml mount these; without them their steps do nothing.
        dense_fp8 = _optional("vllm.model_executor.model_loader.dense_fp8", "convert", "simulate_nvfp4")
        megamoe_vllm = _optional("vllm.model_executor.layers.fused_moe.megamoe_vllm", "install")
        from vllm.model_executor.model_loader.weight_snapshot import Snapshot

        snapshot = Snapshot.for_model(vllm_config, model_config)
        normal = os.environ.get("VLLM_WEIGHT_SNAPSHOT_NORMAL") == "1"  # diagnostic: load, dump, no save
        if snapshot is not None and snapshot.complete() and not normal:
            # The normal path on dummy weights, then every tensor overwritten
            # in place: the object graph is exactly a real load's.
            from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

            model = self._load_from_checkpoint(
                vllm_config, model_config, prefix,
                weights_from=DummyModelLoader(self.load_config).load_weights,
            )
            dense_fp8.convert(model)
            snapshot.restore(model)
            # What a model's own load_weights does after loading, which the
            # dummy loader skips: the DFlash drafter fuses its KV weights.
            for m in model.modules():
                if hasattr(m, "_build_fused_kv_buffers"):
                    m._build_fused_kv_buffers()
            if os.environ.get("VLLM_WEIGHT_SNAPSHOT_VERIFY") == "1":
                from vllm.model_executor.model_loader.weight_snapshot import verify

                # The reference registers the same layer names; keep the
                # snapshot model's registrations and discard its.
                cc = vllm_config.compilation_config
                kept = {k: (dict(getattr(cc, k)) if isinstance(getattr(cc, k), dict) else list(getattr(cc, k)))
                        for k in ("static_forward_context", "static_all_moe_layers") if hasattr(cc, k)}
                for k in kept:
                    getattr(cc, k).clear()
                reference = self._load_from_checkpoint(vllm_config, model_config, prefix)
                for k, v in kept.items():
                    getattr(cc, k).clear()
                    getattr(cc, k).update(v) if isinstance(v, dict) else getattr(cc, k).extend(v)
                verify(model, reference)
                # Diagnostic only: never serve the model under test.
                logger.warning("Weight snapshot verify done; this worker now idles. Take the stack down.")
                import time

                while True:
                    time.sleep(3600)
            self._dump_state(model, vllm_config, snapshot, "snapshot")
            dense_fp8.simulate_nvfp4(model)
            megamoe_vllm.install(model)
            return model
        model = self._load_from_checkpoint(vllm_config, model_config, prefix)
        dense_fp8.convert(model)
        for m in model.modules():  # the drafter fuses its KV weights from whatever format they are now in
            if hasattr(m, "_build_fused_kv_buffers"):
                m._build_fused_kv_buffers()
        if snapshot is not None:
            self._dump_state(model, vllm_config, snapshot, "normal")
            if not normal:
                snapshot.save(model)
        dense_fp8.simulate_nvfp4(model)
        megamoe_vllm.install(model)
        return model

    @staticmethod
    def _dump_state(model, vllm_config, snapshot, mode):
        out = os.environ.get("VLLM_WEIGHT_SNAPSHOT_DUMP")
        if out and snapshot is not None:
            from vllm.model_executor.model_loader.weight_snapshot import dump_state

            name = os.path.basename(snapshot.path).rsplit("-", 1)[0]
            dump_state(model, vllm_config, os.path.join(out, f"{mode}-{name}.txt"))

    def _load_from_checkpoint(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "", weights_from=None
    ) -> nn.Module:
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = self.create_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            logger.debug("Loading weights on %s ...", load_device)
            (weights_from or self.load_weights)(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda_alike() or current_platform.is_xpu():
                peak_memory = torch.accelerator.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                )

            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from vllm.model_inspection import format_model_inspection

    logger.info("vLLM model structure:\n%s", format_model_inspection(model))


def log_online_quantization(vllm_config: VllmConfig) -> None:
    """Log the online-quantized layer count and types, when applicable."""
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )

    quant_config = vllm_config.quant_config
    online_quantization_config = getattr(
        quant_config, "online_quantization_config", None
    )
    if isinstance(online_quantization_config, OnlineQuantizationConfig):
        quant_config = online_quantization_config
    if not isinstance(quant_config, OnlineQuantizationConfig):
        return

    logger.info(
        "Quantized %d layers of types: %s",
        len(quant_config.quantized_layers),
        "; ".join(quant_config.quantized_layer_summaries),
    )


def _has_online_quant(model: nn.Module):
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        if getattr(quant_method, "uses_meta_device", False):
            return True

    return False

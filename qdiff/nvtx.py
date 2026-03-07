from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Callable, Iterable, Optional

import torch


def nvtx_enabled(enabled: bool) -> bool:
    return bool(enabled and torch.cuda.is_available())


@contextmanager
def nvtx_range(message: str, enabled: bool = True):
    if not nvtx_enabled(enabled):
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class DenoisingStepTracker:
    def __init__(self):
        self.step_idx = 0
        self.active = False

    def reset(self):
        self.step_idx = 0
        self.active = False

    def next_label(self) -> str:
        label = f"denoising_step_{self.step_idx:03d}"
        self.step_idx += 1
        return label

    def begin_step(self, enabled: bool = True):
        if not nvtx_enabled(enabled) or self.active:
            return
        torch.cuda.nvtx.range_push(self.next_label())
        self.active = True

    def end_step(self, enabled: bool = True):
        if not nvtx_enabled(enabled) or not self.active:
            return
        torch.cuda.nvtx.range_pop()
        self.active = False


def _coerce_labels(
    labels: Optional[Iterable[str] | str | Callable[..., Optional[Iterable[str] | str]]],
    *args,
    **kwargs,
) -> list[str]:
    if callable(labels):
        labels = labels(*args, **kwargs)
    if labels is None:
        return []
    if isinstance(labels, str):
        return [labels]
    return [label for label in labels if label]


def wrap_module_forward(module, labels, enabled: bool = True):
    if not nvtx_enabled(enabled):
        return module
    if getattr(module, "_nvtx_wrapped", False):
        return module

    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        resolved_labels = _coerce_labels(labels, *args, **kwargs)
        if not resolved_labels:
            return original_forward(*args, **kwargs)
        with ExitStack() as stack:
            for label in resolved_labels:
                stack.enter_context(nvtx_range(label, enabled=True))
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward
    module._nvtx_wrapped = True
    module._nvtx_original_forward = original_forward
    return module


def wrap_named_modules_by_suffix(root_module, suffixes: dict[str, str], enabled: bool = True):
    if not nvtx_enabled(enabled):
        return

    ordered_suffixes = sorted(suffixes.items(), key=lambda item: len(item[0]), reverse=True)
    for name, module in root_module.named_modules():
        if not name:
            continue
        for suffix, label_prefix in ordered_suffixes:
            if name.endswith(suffix):
                wrap_module_forward(module, f"{label_prefix}:{name}", enabled=enabled)
                break


def wrap_named_modules_by_predicate(root_module, predicate, label_prefix: str, enabled: bool = True):
    if not nvtx_enabled(enabled):
        return

    for name, module in root_module.named_modules():
        if not name:
            continue
        if predicate(name, module):
            wrap_module_forward(module, f"{label_prefix}:{name}", enabled=enabled)


def wrap_object_method(obj, method_name: str, labels, enabled: bool = True):
    if not nvtx_enabled(enabled) or not hasattr(obj, method_name):
        return

    original_method = getattr(obj, method_name)
    if getattr(original_method, "_nvtx_wrapped_method", False):
        return

    def wrapped_method(*args, **kwargs):
        resolved_labels = _coerce_labels(labels, *args, **kwargs)
        if not resolved_labels:
            return original_method(*args, **kwargs)
        with ExitStack() as stack:
            for label in resolved_labels:
                stack.enter_context(nvtx_range(label, enabled=True))
            return original_method(*args, **kwargs)

    wrapped_method._nvtx_wrapped_method = True
    wrapped_method._nvtx_original_method = original_method
    setattr(obj, method_name, wrapped_method)

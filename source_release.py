"""Opt-in source-cache release through ComfyUI's reviewed execution integration."""

import gc
import logging
import weakref

import torch

import nodes as comfy_nodes
from comfy_execution.cache_provider import _has_cache_providers
from comfy_execution.caching import NullCache
from comfy_execution.graph_utils import is_link

from .diagnostics import snapshot
from .materialize import MaterializationError


def prepare_release(execution_list, unique_id):
    if execution_list is None:
        # Direct Python callers own their input references; there is no executor cache.
        return None
    if getattr(execution_list, "OUTPUT_RELEASE_API_VERSION", None) != 1:
        raise MaterializationError(
            "Actual source release requires the reviewed ComfyUI cache integration. "
            "See integration/README.md and scripts/install_cache_integration.py. "
            "No source cache entries were changed. Use release_sources=False for stock merging."
        )
    if _has_cache_providers():
        raise MaterializationError("Source release does not support external cache providers, which may retain source tensors.")
    if isinstance(execution_list.output_cache, NullCache):
        raise MaterializationError("Source release needs output caching to retain the final merge; do not use --cache-none.")
    prompt = execution_list.dynprompt
    release = {}
    pending = [value[0] for value in prompt.get_node(unique_id)["inputs"].values() if is_link(value)]
    visited = {unique_id}
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get_node(node_id)
        cls = comfy_nodes.NODE_CLASS_MAPPINGS[node["class_type"]]
        if not hasattr(cls, "RETURN_TYPES"):
            raise MaterializationError(f"Cannot inspect output ownership of {node['class_type']}; source release was not scheduled.")
        slots = {index for index, kind in enumerate(cls.RETURN_TYPES) if kind in ("MODEL", "CLIP")}
        if not slots:
            continue
        release[node_id] = slots
        for value in node["inputs"].values():
            if not is_link(value):
                continue
            parent = prompt.get_node(value[0])
            parent_cls = comfy_nodes.NODE_CLASS_MAPPINGS[parent["class_type"]]
            if not hasattr(parent_cls, "RETURN_TYPES"):
                raise MaterializationError(f"Cannot inspect source output types of {parent['class_type']}")
            if parent_cls.RETURN_TYPES[value[1]] in ("MODEL", "CLIP"):
                pending.append(value[0])
    return release


def schedule_release(execution_list, release, kind):
    if release is None:
        return
    refs = {}
    for node_id, slots in release.items():
        for slot in slots:
            entry = execution_list.output_cache.get_local(node_id, slot)
            if entry is None:
                continue
            for value in entry.outputs[slot]:
                patcher = value.patcher if hasattr(value, "patcher") else value
                tensors = list(patcher.model.parameters()) + list(patcher.model.buffers())
                tensors.extend(backup.weight for backup in patcher.backup.values())
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                        refs[id(tensor)] = weakref.ref(tensor)
    before = snapshot()

    def report_release():
        gc.collect()
        after = snapshot()
        alive = sum(ref() is not None for ref in refs.values())
        logging.info("SimpleMerge %s source-cache release completed: %d/%d tracked CPU source tensors collected; "
                     "RSS %.3f -> %.3f GiB. Other live branches/aliases may retain weights.",
                     kind, len(refs) - alive, len(refs), before.get("rss_gib", float("nan")),
                     after.get("rss_gib", float("nan")))

    execution_list.request_output_release(release, on_release=report_release, retire_previous_outputs=(0,))

"""Bake native static dense patches without loading, unpatching, or changing inputs."""

import copy
import enum
import logging
import re
import types
import uuid
import weakref

import torch
from tokenizers import AddedToken, Tokenizer
from transformers import CLIPTokenizer

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils

from .diagnostics import psutil


class MaterializationError(RuntimeError):
    pass


def _unsupported(reason):
    raise MaterializationError(f"SimpleMerge cannot materialize this input: {reason}. "
                               "Use release_sources=False for the installed stock merge, "
                               "or merge plain dense checkpoints before applying this customization.")


def _audit_copy(value, memo, seen=None):
    """deepcopy shares functions and weakrefs; reject hidden owners rather than retaining them."""
    if seen is None:
        seen = set()
    if id(value) in seen or id(value) in memo:
        return
    seen.add(id(value))
    if isinstance(value, comfy.model_patcher.ModelPatcher):
        _unsupported("a setting retains another model patcher")
    if isinstance(value, weakref.ReferenceType):
        _unsupported("a setting contains a weak reference")
    if isinstance(value, types.MethodType):
        _unsupported("a setting contains a bound method")
    if isinstance(value, types.FunctionType):
        if value.__closure__ or value.__defaults__ or value.__kwdefaults__ or value.__dict__:
            _unsupported(f"a setting contains a closure or function-owned state: {value.__module__}.{value.__qualname__}")
        return
    if isinstance(value, CLIPTokenizer) and hasattr(value, "backend_tokenizer"):
        # Transformers' CLIP decoder closure captures its Rust backend. Recreate
        # that backend from serialized tokenizer data and bind a fresh decoder.
        owned = object.__new__(type(value))
        memo[id(value)] = owned
        memo[id(value.backend_tokenizer)] = Tokenizer.from_str(value.backend_tokenizer.to_str())
        _audit_copy(vars(value), memo, seen)
        vars(owned).update(copy.deepcopy(vars(value), memo))
        owned._wrap_decode_method_backend_tokenizer()
        return
    if isinstance(value, AddedToken):
        memo[id(value)] = copy.deepcopy(value)
        return
    if isinstance(value, comfy.supported_models_base.BASE):
        # BASE.__getattr__ returns None for missing pickle protocol methods.
        # Copy its declared state directly, without invoking that fallback.
        owned = object.__new__(type(value))
        memo[id(value)] = owned
        _audit_copy(vars(value), memo, seen)
        vars(owned).update(copy.deepcopy(vars(value), memo))
        return
    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided or value.is_meta:
            _unsupported("a quantized, sparse, meta, or custom tensor")
        memo[id(value)] = value.detach().to(device="cpu", copy=True)
        return
    if isinstance(value, (str, bytes, int, float, bool, complex, type(None), type, enum.Enum,
                          torch.device, torch.dtype, uuid.UUID, re.Pattern, types.BuiltinFunctionType)):
        return
    if isinstance(value, dict):
        children = list(value.keys()) + list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    elif hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        _unsupported(f"unsupported metadata object {type(value).__name__}")
    for child in children:
        _audit_copy(child, memo, seen)


def _copy_owned(value, memo=None):
    if memo is None:
        memo = {}
    _audit_copy(value, memo)
    return copy.deepcopy(value, memo)


def _validate(patcher):
    if type(patcher) is not comfy.model_patcher.ModelPatcher or patcher.is_dynamic():
        _unsupported("dynamic or third-party patcher (restart ComfyUI with --disable-dynamic-vram for dense models)")
    for name in ("attachments", "additional_models", "weight_wrapper_patches", "injections",
                 "hook_patches", "hook_backup", "cached_hook_patches"):
        if getattr(patcher, name):
            _unsupported(f"active {name}")
    if patcher.current_hooks or patcher.forced_hooks or patcher.is_injected:
        _unsupported("active hooks or injections")
    for name in ("callbacks", "wrappers"):
        if any(items for group in getattr(patcher, name).values() for items in group.values()):
            _unsupported(f"active {name}")
    if patcher.backup_buffers or patcher.is_multigpu_base_clone:
        _unsupported("buffer backups or a multi-GPU delegate")
    state_keys = patcher.model.state_dict().keys()
    for key in patcher.object_patches.keys() | patcher.object_patches_backup.keys():
        if key == "model_sampling" or key.startswith("model_sampling."):
            continue
        if any(weight == key or weight.startswith(key + ".") for weight in state_keys):
            _unsupported(f"object patch replaces registered weights at {key}")


def validate_sources(*patchers):
    for patcher in patchers:
        _validate(patcher)
        for tensor in list(patcher.model.parameters()) + list(patcher.model.buffers()):
            if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.is_meta
                    or tensor.layout != torch.strided
                    or (tensor.is_floating_point() and tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64))):
                _unsupported("quantized/custom source weights")


@torch.inference_mode()
def materialize_model(patcher):
    _validate(patcher)
    source = patcher.model
    tensors = dict(source.named_parameters(remove_duplicate=False))
    tensors.update(source.named_buffers(remove_duplicate=False))
    state = source.state_dict()
    if not set(state).issubset(tensors) or not set(patcher.patches).issubset(state):
        _unsupported("state dict keys do not map to registered parameters/buffers")
    storages = set()
    total = largest = 0
    memo = {}
    for key, tensor in tensors.items():
        if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.is_meta
                or tensor.layout != torch.strided or not tensor.is_contiguous()
                or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                                        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)):
            _unsupported(f"unsupported dtype/layout for {key}")
        storage = (tensor.device, tensor.untyped_storage().data_ptr())
        if tensor.numel() and storage in storages:
            _unsupported(f"tied/shared storage at {key}")
        storages.add(storage)
        size = tensor.numel() * tensor.element_size()
        total += size
        largest = max(largest, size)
        placeholder = torch.empty_like(tensor, device="meta")
        memo[id(tensor)] = torch.nn.Parameter(placeholder, requires_grad=tensor.requires_grad) if isinstance(tensor, torch.nn.Parameter) else placeholder
        _, setter, converter = comfy.model_patcher.get_key_weight(source, key)
        if key in patcher.patches and (setter is not None or converter is not None):
            _unsupported(f"custom tensor setter/converter for {key}")

    # Nested merge evaluation temporarily allocates multiple fp32 versions of one key.
    # This is a guard, not a reservation or a promise about total process headroom.
    required = total + largest * (8 + 4 * _patch_depth(patcher.patches)) + 2**30
    if psutil is not None and psutil.virtual_memory().available < required:
        raise MaterializationError(f"SimpleMerge needs approximately {required / 2**30:.1f} GiB free RAM "
                                   "for this output and tensor scratch space. Free memory first, "
                                   "or use release_sources=False. Cached source weights are still retained.")

    # Only these native runtime references/functions are regenerated by ModelPatcher.load.
    current_patcher = getattr(source, "current_patcher", None)
    if current_patcher is not None:
        memo[id(current_patcher)] = None
    for module in source.modules():
        for name in ("weight_function", "bias_function"):
            functions = getattr(module, name, [])
            if any(type(f) is not comfy.model_patcher.LowVramPatch for f in functions):
                _unsupported(f"custom {name}")
            if functions:
                memo[id(functions)] = []
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"):
            if getattr(module, name):
                _unsupported(f"module {name}")

    target = _copy_owned(source, memo)
    for module in target.modules():
        comfy.model_patcher.wipe_lowvram_weight(module)
        if hasattr(module, "comfy_patched_weights"):
            module.comfy_patched_weights = False
    if hasattr(target, "current_patcher"):
        target.current_patcher = None
    target.device = torch.device("cpu")
    target.model_loaded_weight_memory = 0
    target.model_offload_buffer_memory = 0
    target.lowvram_patch_counter = 0
    target.model_lowvram = False
    target.current_weight_patches_uuid = None

    # Evaluate against each ORIGINAL base, even when another clone has loaded patches
    # into the shared module. The native return_weight path does not consult backup.
    evaluator = comfy.model_patcher.ModelPatcher(target, patcher.load_device, torch.device("cpu"))
    evaluator.patches = patcher.patches
    parameters = set(dict(source.named_parameters(remove_duplicate=False)))
    nonfinite_keys = []
    for key, tensor in tensors.items():
        comfy.model_management.throw_exception_if_processing_interrupted()
        base = patcher.backup[key].weight if key in patcher.backup else tensor
        _set_tensor(target, key, base, key in parameters)
        effective = evaluator.patch_weight_to_device(key, device_to=torch.device("cpu"), return_weight=True)
        if effective.shape != tensor.shape or effective.dtype != tensor.dtype:
            _unsupported(f"patch changes shape/dtype at {key}")
        if effective.is_floating_point() and not torch.isfinite(effective).all():
            nonfinite_keys.append(key)
        owned = effective.detach().to(device="cpu", copy=True)
        _set_tensor(target, key, owned, key in parameters)
        del base, effective, owned
    # Keep the temporary evaluator's destructor away from output/source state.
    evaluator.patches = {}
    result = comfy.model_patcher.ModelPatcher(target, patcher.load_device, patcher.offload_device,
                                              weight_inplace_update=patcher.weight_inplace_update)
    result.model_options = _copy_owned(patcher.model_options)
    result.object_patches = _copy_owned(patcher.object_patches)
    # The visible module may currently carry a DIFFERENT clone's object patches.
    for key, original in patcher.object_patches_backup.items():
        comfy.utils.set_attr(target, key, _copy_owned(original))
    result.force_cast_weights = patcher.force_cast_weights
    result.is_clip = patcher.is_clip
    result.hook_mode = patcher.hook_mode
    if nonfinite_keys:
        logging.warning("SimpleMerge preserved native non-finite values in %d tensors (first: %s). "
                        "Check source checkpoints and text conditioning; no values were repaired.",
                        len(nonfinite_keys), nonfinite_keys[0])
    return result


def _set_tensor(model, key, tensor, parameter):
    module_name, _, name = key.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    if parameter:
        module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        module._buffers[name] = tensor


def _patch_depth(value):
    if isinstance(value, dict):
        return max((_patch_depth(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_patch_depth(v) for v in value), default=0)
    return 0


def materialize_clip(clip):
    result = comfy.sd.CLIP(no_init=True)
    result.patcher = materialize_model(clip.patcher)
    result.cond_stage_model = result.patcher.model
    # Copy tokenizer data as well; no reliance on undocumented tokenizer immutability.
    for name, value in vars(clip).items():
        if name not in ("patcher", "cond_stage_model"):
            setattr(result, name, _copy_owned(value))
    return result

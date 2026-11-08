#!/usr/bin/env python3
"""CPU test against two real checkpoints. Does not connect to a running server."""

import argparse
import gc
import json
from pathlib import Path
import sys
import weakref

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint_a")
parser.add_argument("checkpoint_b")
parser.add_argument("--comfy-root", type=Path, required=True)
parser.add_argument("--quick", action="store_true", help="Only ratio .5; full run also tests boundaries and a chain")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import os
os.environ["COMFYUI_PATH"] = str(args.comfy_root)
from support import effective, extension, torch
import comfy.sd
from comfy_extras.nodes_model_merging import CLIPMergeSimple, ModelMergeSimple
from simplemerge_release.diagnostics import snapshot

torch.set_num_threads(4)
reports = []


def record(stage, **kwargs):
    report = {"stage": stage, **snapshot(), **kwargs}
    reports.append(report)
    print(json.dumps(report), flush=True)


def compare(native, output):
    actual = output.model.state_dict()
    assert set(actual) == set(native.model.state_dict())
    source_storages = {tensor.untyped_storage().data_ptr() for source in sources
                       for tensor in (source[1].patcher if output.is_clip else source[0]).model.state_dict().values()}
    patches = native.get_key_patches()
    for key, tensor in actual.items():
        # Reuse the reference descriptors; this does not copy full state dictionaries.
        import comfy.float
        import comfy.lora
        import comfy.utils
        value = patches[key]
        reference = value[0][0].detach().to(device="cpu", copy=True)
        if key in native.patches:
            reference = comfy.lora.calculate_weight(value[1:], reference.float(), key)
            reference = comfy.float.stochastic_rounding(reference, tensor.dtype,
                                                       seed=comfy.utils.string_to_seed(key))
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0, equal_nan=True, msg=lambda msg: key + ": " + msg)
        if tensor.numel():
            assert tensor.untyped_storage().data_ptr() not in source_storages, key
    return len(actual)


with torch.inference_mode():
    record("baseline")
    options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu"), "dtype": torch.float16}
    sources = []
    for path in (args.checkpoint_a, args.checkpoint_b):
        sources.append(comfy.sd.load_checkpoint_guess_config(path, output_vae=False,
                        model_options=options, te_model_options={**options, "initial_device": torch.device("cpu")}, disable_dynamic=True))
        assert sources[-1][0] is not None and sources[-1][1] is not None
        record("source_loaded", count=len(sources))
    refs = [weakref.ref(obj) for item in sources for obj in (item[0], item[0].model, item[1], item[1].cond_stage_model)]
    ratios = (.5,) if args.quick else (0., .25, .5, .75, 1.)
    retained = []
    for index, stock, name in ((0, ModelMergeSimple, "ModelMergeSimpleRelease"),
                                (1, CLIPMergeSimple, "CLIPMergeSimpleRelease")):
        for ratio in ratios:
            native = stock().merge(sources[0][index], sources[1][index], ratio)[0]
            output = extension.NODE_CLASS_MAPPINGS[name]().merge(sources[0][index], sources[1][index], ratio, True)[0]
            patcher = output.patcher if index else output
            count = compare(native.patcher if index else native, patcher)
            record("parity", node=name, ratio=ratio, keys=count, exact=True)
            del native
            if ratio == ratios[-1]:
                retained.append(output)
            del output, patcher
            gc.collect()
        if not args.quick:
            first = stock().merge(sources[0][index], sources[1][index], .25)[0]
            native = stock().merge(first, sources[0][index], .75)[0]
            output = extension.NODE_CLASS_MAPPINGS[name]().merge(first, sources[0][index], .75, True)[0]
            count = compare(native.patcher if index else native, output.patcher if index else output)
            record("chain_parity", node=name, keys=count, exact=True)
            del first, native, output
            gc.collect()
    record("before_source_drop")
    del sources
    gc.collect()
    assert all(ref() is None for ref in refs), "Source objects still referenced outside the executor"
    record("sources_collected", collected=len(refs))
    assert torch.isfinite(next(retained[0].model.parameters())).all()
    tokens = retained[1].tokenize("a peaceful mountain landscape")
    assert tokens
    record("independent_tokenizer_pass")
    conditioning = retained[1].encode_from_tokens_scheduled(tokens)
    for tensor, metadata in conditioning:
        assert torch.isfinite(tensor).all(), "Non-finite text conditioning"
        if "pooled_output" in metadata:
            assert torch.isfinite(metadata["pooled_output"]).all(), "Non-finite pooled conditioning"
    record("independent_text_encode_pass")
    del conditioning, retained
    gc.collect()

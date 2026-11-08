#!/usr/bin/env python3
"""Queue three small generations and an optional post-merge LoRA generation."""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint_a")
parser.add_argument("checkpoint_b")
parser.add_argument("--url", default="http://127.0.0.1:8189")
parser.add_argument("--lora")
parser.add_argument("--server-pid", type=int)
args = parser.parse_args()


def api(route, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(args.url + route, data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def memory():
    result = api("/system_stats")
    if args.server_pid:
        values = {}
        for line in Path(f"/proc/{args.server_pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, amount, _ = line.split()
                values[key.rstrip(":") + "_gib"] = int(amount) / 2**20
        result.update(process=values)
    return result


prompt = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": args.checkpoint_a}},
    "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": args.checkpoint_b}},
    "3": {"class_type": "ModelMergeSimpleRelease", "inputs": {
        "model1": ["1", 0], "model2": ["2", 0], "ratio": .5, "release_sources": True}},
    "4": {"class_type": "CLIPMergeSimpleRelease", "inputs": {
        "clip1": ["1", 1], "clip2": ["2", 1], "ratio": .5, "release_sources": True}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": "a peaceful mountain landscape"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": "blurry"}},
    "7": {"class_type": "EmptyLatentImage", "inputs": {"width": 256, "height": 256, "batch_size": 1}},
    "8": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "positive": ["5", 0], "negative": ["6", 0],
        "latent_image": ["7", 0], "seed": 101, "steps": 2, "cfg": 4., "sampler_name": "euler", "scheduler": "normal", "denoise": 1.}},
    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
    "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "SimpleMerge_smoke"}},
}

queue = api("/queue")
assert not queue["queue_running"] and not queue["queue_pending"], "Test server must be idle"
print(json.dumps({"stage": "baseline", "memory": memory()}), flush=True)
for generation in range(4 if args.lora else 3):
    prompt["8"]["inputs"]["seed"] = 101 + generation
    prompt["5"]["inputs"]["text"] = f"a peaceful mountain landscape, season {generation + 1}"
    if generation == 3:
        prompt["11"] = {"class_type": "LoraLoader", "inputs": {"model": ["3", 0], "clip": ["4", 0],
                          "lora_name": args.lora, "strength_model": .3, "strength_clip": .3}}
        prompt["8"]["inputs"]["model"] = ["11", 0]
        prompt["5"]["inputs"]["clip"] = ["11", 1]
        prompt["6"]["inputs"]["clip"] = ["11", 1]
    started = time.monotonic()
    submitted = api("/prompt", {"prompt": prompt})
    assert not submitted.get("node_errors"), submitted
    prompt_id = submitted["prompt_id"]
    while True:
        history = api("/history/" + prompt_id).get(prompt_id)
        if history and history["status"]["completed"]:
            break
        if history and history["status"]["status_str"] == "error":
            raise RuntimeError(history["status"])
        if time.monotonic() - started > 900:
            raise TimeoutError(prompt_id)
        time.sleep(1)
    assert history["status"]["status_str"] == "success", history["status"]
    cached = next(data["nodes"] for event, data in history["status"]["messages"] if event == "execution_cached")
    if generation:
        assert {"3", "4"}.issubset(cached), cached
        assert not {"1", "2"}.intersection(cached), "Source loader MODEL/CLIP outputs were not released"
    images = history["outputs"]["10"]["images"]
    print(json.dumps({"stage": "generation", "generation": generation + 1, "lora": generation == 3,
                      "seconds": time.monotonic() - started, "cached_nodes": cached,
                      "images": images, "memory": memory()}), flush=True)

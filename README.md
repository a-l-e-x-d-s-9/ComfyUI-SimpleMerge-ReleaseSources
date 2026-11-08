# ComfyUI-SimpleMerge-ReleaseSources

Two drop-in merge nodes with an optional `release_sources` toggle:

- **Model Merge Simple (Release Sources)** — `ModelMergeSimpleRelease`
- **CLIP Merge Simple (Release Sources)** — `CLIPMergeSimpleRelease`

**This extension materializes independent weights. It does not evict source checkpoints from ComfyUI's execution cache.** Enabling it can increase RAM usage. Use it on the final MODEL and CLIP merges first, and keep intermediate toggles off.

## Install

From your ComfyUI directory:

```bash
git clone https://github.com/a-l-e-x-d-s-9/ComfyUI-SimpleMerge-ReleaseSources.git custom_nodes/ComfyUI-SimpleMerge-ReleaseSources
```

Restart ComfyUI, then search for the two display names above. There are no additional pip requirements beyond ComfyUI's existing PyTorch, Transformers and Tokenizers dependencies. `psutil` is optional for diagnostics and RAM preflight.

## Behavior

Inputs, connections and ratios match the stock nodes. `ratio=1` selects model1's effective weights; `ratio=0` selects model2's effective weights for matching mergeable keys. CLIP keeps CLIP1's tokenizer/settings and the stock exclusions for `.position_ids` and `.logit_scale`.

With `release_sources=False` (default), the node delegates directly to the installed built-in merge, adding no full-model copies. With `True`, it first constructs that same native lazy merge, then evaluates each tensor through the installed `ModelPatcher.patch_weight_to_device(..., return_weight=True)` path, including nested patches and native dtype rounding. The output owns a separate module and CPU storage, with fresh patch/backup/parent state. Original inputs are never loaded, unpatched, or changed by materialization. Subsequent LoRAs can be applied normally.

The module skeleton is copied with meta-device placeholders; full weight tensors are filled individually. Native loaded-model backups supply the original base when another clone has already patched the shared module. Nonpersistent buffers, model sampling/configuration, declared model options, object patches, precision and load/offload settings are copied. CLIP's tokenizer backend is reconstructed independently, including its decoder closure. This avoids copying the full source module's weights before baking them.

GPU VRAM release moves or unloads device allocations; it does not necessarily free CPU weights. Materialization creates independently owned effective weights, which costs another output-sized allocation in CPU RAM. Reclaiming source CPU RAM requires **every owner** of the source weights to disappear: loader outputs, intermediate lazy patch lists, execution-cache entries, current node inputs and other graph branches may still retain them. This extension neither clears those owners nor calls global model unloading or CUDA-cache clearing.

## Convert an existing workflow

```bash
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/convert_workflow.py \
  user/default/workflows/Sassy_Girls_Illustrious_2.0.json
```

This creates `Sassy_Girls_Illustrious_2.0_ReleaseSources.json` beside the source. Load that copy in the visual editor. All nine MODEL and nine CLIP node types are replaced; only final MODEL **2023** and final CLIP **2022** get `True`. IDs, link IDs, ratios, positions, other nodes and settings are preserved. A boolean socket/widget and the workflow's named widget values are updated; misleading `comfy-core` metadata is removed from the converted nodes. Conversion is idempotent, checks link endpoints/socket references, and refuses to overwrite any output file. Other workflows can specify `--final-model ID --final-clip ID`.

The user's full workflow is intentionally not distributed in this public repository. Its converted copy stays in the user's workflow directory.

## Cache and RAM limits

Queue three generations with different downstream seeds/prompts. The final merge nodes should be listed in ComfyUI's `execution_cached` event after the first run. Ordinary input fingerprints include the boolean, so toggling it invalidates the node. There is no `IS_CHANGED` override or output-node flag forcing re-execution.

Default RAM-pressure caching may evict outputs under pressure; it cannot guarantee the final merge stays resident. `--cache-classic` more aggressively retains unchanged outputs, including source loaders: it favors reuse at a higher RAM cost. Neither mode becomes a source-release mechanism just because this toggle is enabled. The object cache retains these stateless node instances, not their input weights; the output cache and currently executing input lists are separate owners. Static model-manager entries can also retain loaded modules. No cache integration or core monkeypatch is installed.

For ten dense SDXL/Illustrious checkpoints, measured two-model sizes imply roughly **63 GiB of source MODEL+CLIP weights plus 6.3 GiB of independent final outputs**, before Python, VAEs, tensor scratch and sampling. That exceeds 60 GiB usable physical RAM. Repeated materialization at intermediate stages makes this worse while caches retain sources. A node cannot schedule serial checkpoint loading or safely delete arbitrary upstream cache outputs while other consumers may still need them.

The smallest reliable operational change is a separate build phase that saves the final merged checkpoint, followed by a generation workflow loading that single artifact. This uses existing checkpoint-save/load nodes and requires no core edits. A future streaming merger would need its own serial loader and accumulation lifecycle, rather than two drop-in nodes. This release does not replace the user's graph or add that feature.

## Supported inputs and diagnostics

Tested with native static dense SDXL/Illustrious fp16 checkpoints on ComfyUI `c9ea36b9d2140c89b2ef84f93f428df474e72ba5` (`v0.41.0-14-gc9ea36b9d`), Python 3.10 and PyTorch `2.12.1+cu130`. Small parity tests also cover float32 and bfloat16. Materialization requires the native static `ModelPatcher`; use `--disable-dynamic-vram` for these dense models. The lazy path remains whatever your installed stock node supports.

Materialization rejects dynamic/third-party patchers, quantized/custom/sparse/meta tensors, tied/noncontiguous weight storage, custom tensor setters for patched keys, object patches replacing registered weights (sampling settings excepted), active hooks, injections, attachments, additional models, weight wrappers and callback wrappers. Settings with model-owning closures/bound methods are also rejected rather than silently retained or discarded. Plain native model options and object patches are preserved. An actionable error suggests `release_sources=False` where stock behavior is needed. Available-RAM checks estimate output storage plus per-key nested patch scratch; they are not an allocation reservation or an OOM guarantee. Cancellation is checked between tensors.

Logs distinguish process RSS, process peak RSS, available host RAM and PyTorch CUDA allocated/reserved memory in GiB. The peak is cumulative for the process, not an isolated merge peak. Without `psutil`, RSS/headroom fields and preflight are unavailable; the Linux process-peak diagnostic remains available. `materialized` does **not** mean `source RAM freed`: references still exist in the caller when the node returns, and cache ownership is not inspected from inside the node.

Some tested source checkpoints already contain NaNs in six tensors of CLIP-L's last transformer layer. This extension preserves the native result and logs that fact; it does not sanitize source values. Native parity compares matching NaN masks explicitly. SDXL uses CLIP-L's penultimate hidden state and discards its pooled output; finite text conditioning and sampling must still be checked. An all-weights-finite assertion cannot pass for those source files.

Actual measurements and integration coverage are recorded in [TEST_RESULTS.md](TEST_RESULTS.md).

## Tests

Run from your ComfyUI root with its interpreter:

```bash
venv/bin/python -m compileall -q custom_nodes/ComfyUI-SimpleMerge-ReleaseSources
venv/bin/python -m unittest discover -s custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/tests -v

# Full real-checkpoint CPU parity; loads exactly two checkpoints.
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/integration_parity.py \
  /path/to/checkpoint-a.safetensors /path/to/checkpoint-b.safetensors --comfy-root .
```

The real-model test checks every native state-dict key at ratios 0, .25, .5, .75 and 1, a chained merge, storage separation, weak-reference collection and independent tokenization. `--quick` tests only .5. It deliberately runs outside the executor to distinguish actual source-object collection from executor cache retention. It does not claim to measure queued generation or GPU sampling. Small tests exercise loaded backups, native LoRA load/apply/unpatch, low-VRAM functions, cancellation, errors, branch-safe nonmutation, insufficient memory, boolean invalidation and both installed cache implementations.

License: GPL-3.0-or-later.

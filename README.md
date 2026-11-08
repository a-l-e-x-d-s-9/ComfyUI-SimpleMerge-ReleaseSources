# ComfyUI-SimpleMerge-ReleaseSources

Two drop-in ComfyUI merges with a `release_sources` boolean:

- **Model Merge Simple (Release Sources)** — `ModelMergeSimpleRelease`
- **CLIP Merge Simple (Release Sources)** — `CLIPMergeSimpleRelease`

**Actual source release requires the explicit, reviewed [three-file ComfyUI cache integration](integration/README.md).** The extension never installs that patch automatically. Without it, `release_sources=True` fails before allocating an independent model; `False` uses the installed stock merge.

## What the boolean does

`False` delegates directly to the installed native merge, with no full-model copies. `True` first creates the same native lazy merge, then bakes its effective weights into a separately owned CPU module. Once it succeeds, it requests release of upstream MODEL/CLIP cache outputs. The executor drops each requested output only after its remaining consumers finish, including shared branches and async work. It retains non-model outputs such as VAE and leaves the independent final merge cached.

This also removes source references from current-prompt input caches. Inactive cached versions of the affected producers are retired so that an old lazy merge does not keep the same checkpoints alive. Entries used by the active prompt remain protected. No global model unloading, CUDA-cache clearing, or executor/cache monkeypatch is installed by the node.

Changing only downstream seeds/prompts can reuse the final MODEL, CLIP and retained VAE without reloading source checkpoints. Changing merge ratios, checkpoint names, or the toggle invalidates the relevant native cache signatures; missing source slots then cause ordinary loader execution. Normal cache policy still applies: severe RAM pressure can evict a final merge and require rebuilding it.

Source cache release is real; other live branches, extensions or tensor aliases can still retain weights. Logs report tracked CPU source tensors collected and actual process RSS before/after, rather than treating GPU unloading or garbage-collector counts as proof.

## Install and enable

From ComfyUI's root:

```bash
git clone https://github.com/a-l-e-x-d-s-9/ComfyUI-SimpleMerge-ReleaseSources.git custom_nodes/ComfyUI-SimpleMerge-ReleaseSources

# Read-only compatibility check: no core files are modified.
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/install_cache_integration.py --comfy-root .

# After explicitly reviewing/authorizing the core integration:
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/install_cache_integration.py --comfy-root . --apply
```

Restart ComfyUI and search for the display names above. There are no extra mandatory pip dependencies beyond ComfyUI's existing PyTorch/Transformers/Tokenizers stack. `psutil` is optional for RAM diagnostics/preflight. The installer checks exact source hashes and refuses unknown/modified core versions. [Integration details and rollback](integration/README.md).

## Existing workflows

Inputs, connections and ratios match stock nodes. `ratio=1` selects model1's effective weights; `ratio=0` selects model2's effective weights for matching mergeable keys. CLIP preserves CLIP1's tokenizer/settings and native `.position_ids`/`.logit_scale` exclusions. Fresh LoRAs can be applied afterward.

```bash
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/convert_workflow.py \
  user/default/workflows/Sassy_Girls_Illustrious_2.0.json
```

This creates `Sassy_Girls_Illustrious_2.0_ReleaseSources.json` beside the source. All nine MODEL and nine CLIP merge types are replaced; only final MODEL **2023** and final CLIP **2022** enable the boolean. IDs, link IDs, positions, ratios and unrelated nodes/settings are preserved. Added widgets/named values and node heights are updated, misleading `comfy-core` metadata is removed, link endpoints are validated, and conversion is idempotent. The script refuses to overwrite files. Other graphs can specify `--final-model ID --final-clip ID`.

The user's complete workflow, private logs and generated test images are not distributed in this public repository.

## Memory limits

With only the final merges enabled, source release occurs **after those merges finish**. It does not reduce the RAM needed to load all checkpoints and bake the first final output. The measured two-source sizes imply roughly 63 GiB of MODEL+CLIP source weights for ten checkpoints, plus 6.3 GiB for the independent final pair and additional scratch/VAEs/runtime memory. That exceeds the rig's 60.7 GiB physical RAM.

Enabling intermediate materialization can let earlier sources be reclaimed sooner, but adds work and rounds intermediate weights back to storage dtype. That can differ slightly from a fully lazy native merge chain, which evaluates nested patches at higher precision. It is a separate experiment and is not automatically enabled throughout the ten-model workflow. The full ten-model workflow was not run during these bounded tests.

Default RAM-pressure caching is a reasonable starting policy once sources can actually be released. Classic caching can retain the final output more aggressively, but does not reserve RAM or guarantee that a model stays on the GPU. Avoid `--cache-none`, which cannot retain the final merge. External cache providers are rejected for source-release mode because they may retain additional source copies.

For the lowest build-time peak, a future serial/streaming loader would need its own loading and accumulation lifecycle. Saving the merged checkpoint in a build phase and generating from that single file remains another reliable option. Neither feature replaces the user's graph automatically.

## Implementation and compatibility

Tested with ComfyUI `c9ea36b9d2140c89b2ef84f93f428df474e72ba5` (`v0.41.0-14-gc9ea36b9d`), Python 3.10.6, PyTorch 2.12.1+cu130, Transformers 5.2.0 and Tokenizers 0.22.1. Native static dense SDXL/Illustrious fp16 checkpoints were used. Small native parity tests also cover float32 and bfloat16. Use `--disable-dynamic-vram` for these static models.

Baking uses native `patch_weight_to_device(..., return_weight=True)`, including loaded-source backups, nested patch evaluation and native dtype rounding. The skeleton uses meta placeholders, filled one tensor at a time. Nonpersistent buffers, sampling/configuration, declared model options, object patches, precision and device settings are preserved. CLIP's tokenizer backend and decoder closure are independently rebuilt. Output patch/backup/parent state is fresh; input weights are not modified.

Unsupported inputs raise actionable errors: dynamic/third-party patchers; quantized/custom/sparse/meta/tied/noncontiguous weight storage; custom setters for patched weights; weight-replacing object patches other than sampling settings; active hooks, injections, attachments, additional models and wrappers; or settings that retain model-owning closures/bound methods. `False` remains the stock path for those inputs. Available-RAM preflight is an estimate, not an allocation reservation. Cancellation is checked between tensors.

Some tested source files contain NaNs in six CLIP-L last-layer tensors. The extension preserves the native values and logs them. SDXL uses CLIP-L's penultimate hidden state and discards its pooled output; actual conditioning and sampling were verified. An all-weights-finite assertion cannot pass for those source files, so native parity compares matching NaN masks explicitly.

## Validation

See [SOURCE_RELEASE_RESULTS.md](SOURCE_RELEASE_RESULTS.md) for actual cache release and generation measurements. [TEST_RESULTS.md](TEST_RESULTS.md) records the historical materialize-only implementation and its limitations.

Run from the matching ComfyUI root:

```bash
venv/bin/python -m compileall -q custom_nodes/ComfyUI-SimpleMerge-ReleaseSources
venv/bin/python -m unittest discover -s custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/tests -v

# Real-checkpoint CPU parity. Direct calls have no executor cache to release.
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/integration_parity.py \
  /path/to/checkpoint-a.safetensors /path/to/checkpoint-b.safetensors --comfy-root .
```

Executor-release tests are skipped on an unpatched core; the missing-integration fail-fast test still runs. GPU queue testing uses `scripts/integration_queue.py` against a dedicated idle test server. It queues three 256×256 two-step images and optionally a post-merge LoRA generation, then checks retained final merges and absent full source-loader entries.

License: GPL-3.0-or-later.

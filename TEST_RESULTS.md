# Historical validation — materialize-only implementation

These results describe commit `78ca615`, which did not release executor cache ownership. For the actual release implementation and core integration, see [SOURCE_RELEASE_RESULTS.md](SOURCE_RELEASE_RESULTS.md).

Validation date: 2026-11-08

## Environment

- ComfyUI `c9ea36b9d2140c89b2ef84f93f428df474e72ba5`, `v0.41.0-14-gc9ea36b9d`.
- Existing local change to `get_key_patches` skips quantized sidecar keys; it was inspected and left unchanged. This extension does not modify ComfyUI core.
- Python 3.10.6; PyTorch 2.12.1+cu130; Transformers 5.2.0; Tokenizers 0.22.1; psutil 7.2.2.
- NVIDIA RTX 4060 Ti, 16 GiB; 60.74 GiB usable host RAM, 31 GiB swap.
- Original GPU flags preserved: `CUDA_VISIBLE_DEVICES=0`, `--reserve-vram 2 --use-pytorch-cross-attention --fast fp16_accumulation --disable-dynamic-vram`.
- Normal server: default RAM-pressure caching. Isolated loopback test server: first default caching, then an actual restart with `--cache-classic`; only this custom extension was enabled.
- Checkpoints: Platinum Illusion Illustrious 1.0 and Shady Fay Illustrious 1.0, dense fp16 MODEL+CLIP weights. Native model/tokenizer implementations came from this installed checkout, not online master.

## Correctness and ownership

The standalone real-checkpoint test compared **all 1,690 MODEL keys and all 716 CLIP keys**, with zero numerical tolerance, at ratios **0, .25, .5, .75 and 1**. Both chained merges `(A+B)+A` also matched exactly. Shapes/key sets matched, output weight storages differed from source storages, and stock CLIP exclusions were retained. A final .5 rerun checked the final implementation and finite text conditioning after source collection.

Eight tracked source objects (two MODEL patchers, their modules, two CLIP wrappers and their modules) became unreachable after dropping the standalone test's source/reference outputs. The independent outputs remained usable; CLIP tokenization and actual `encode_from_tokens_scheduled` produced finite conditioning and pooled output after collection. Small native forward tests also exercise the independent MODEL after collection. The separate GPU queue test sampled the real materialized output while cached sources remained present.

Source-data exception: both input files contain NaNs in six CLIP-L last-layer tensors. Matching NaN locations were compared explicitly with `equal_nan=True`; no values were sanitized. Thus **the all-weights-finite criterion is not met by these checkpoints**. Actual SDXL text conditioning was finite, and GPU sampling succeeded. This is reported as an input-data limitation, not a clean all-finite pass.

The final automated suite contains **16 passing tests** covering native lazy/eager parity in float32/fp16/bfloat16; prior patches/LoRAs and chained merges; loaded-source backups; fresh native LoRA load/apply/unpatch; independent tokenizers and decoder closures; low-VRAM runtime functions; source/output collection; cancellation midway through baking; errors, unsafe weight-replacing object patches and RAM preflight; cache signatures in both installed cache classes; toggle invalidation; and idempotent workflow conversion. Syntax compilation also passed with the installed interpreter.

## Host RAM: standalone CPU tests

All figures are GiB. Peak RSS is the cumulative process high-water mark. Available host RAM includes other ComfyUI/test processes that were running concurrently and is not attributable to this process alone. CPU parity tests allocated **0 CUDA VRAM**.

| Final .5 parity / ownership run | Process RSS | Peak RSS | Available host RAM |
|---|---:|---:|---:|
| Baseline | 0.908 | 0.908 | 56.087 |
| One source MODEL+CLIP | 7.274 | 13.586 | 49.757 |
| Two sources | 13.624 | 19.933 | 43.393 |
| Sources + independent final MODEL+CLIP | 20.064 | 20.884 | 16.376 |
| After dropping sources; all eight weakrefs dead | 9.414 | 20.884 | 26.379 |
| Independent finite CLIPTextEncode completed | 9.423 | 20.884 | 26.365 |

The exhaustive five-ratio/chained run peaked at **24.339 GiB**; its RSS dropped from **23.338 to 12.689 GiB** after sources were collected. These higher figures include repeated reference/output construction and allocator retention. They are not the peak of a single production merge. RSS does not drop by exactly the tensor-byte count, and object collection alone is not proof that every allocator page returned to the OS.

## Actual queued GPU generations

Each mode queued three 256×256, two-step Euler generations with changed seed/prompt, then a fourth generation using native `LoraLoader` **after** both independent merges. A compatible existing Illustrious CGI style LoRA was applied at MODEL/CLIP strength .3. These are execution smoke tests, not image-quality assessments. All eight PNGs were nonblank. Corresponding pixels were identical across cache modes, including the LoRA pass.

| Cache | Generation | Wall time, seconds | Process RSS, GiB | Process peak RSS, GiB | CUDA allocated / reserved, GiB |
|---|---|---:|---:|---:|---:|
| Default | Baseline | — | 1.314 | 1.314 | 0 / 0 |
| Default | 1, loads + merges | 48.417 | 14.034 | 19.297 | 7.995 / 8.000 |
| Default | 2, changed seed/prompt | 1.029 | 14.051 | 19.297 | 7.995 / 8.156 |
| Default | 3, changed seed/prompt | 1.032 | 14.051 | 19.297 | 7.995 / 8.156 |
| Default | 4, post-merge LoRA | 8.140 | 20.210 | 20.361 | 7.995 / 8.000 |
| Classic | Baseline | — | 1.313 | 1.313 | 0 / 0 |
| Classic | 1, loads + merges | 45.507 | 15.618 | 20.799 | 6.470 / 6.500 |
| Classic | 2, changed seed/prompt | 1.041 | 15.635 | 20.799 | 6.470 / 6.656 |
| Classic | 3, changed seed/prompt | 1.035 | 15.635 | 20.799 | 6.470 / 6.688 |
| Classic | 4, post-merge LoRA | 8.077 | 21.789 | 21.940 | 6.470 / 6.500 |

Wall time includes HTTP submission/history polling at one-second intervals. CUDA columns use the server's PyTorch allocator figures, distinct from driver-wide used VRAM. Different source device placement accounts for part of the CPU/GPU split between runs; this is not a general benchmark proving one policy uses less total memory.

Node timings, including native lazy construction: default CLIP **7.69 s**, MODEL **31.53 s**; classic CLIP **6.07 s**, MODEL **31.55 s**. Each merge logged exactly once per test-server run. Native merge construction itself is substantial: the installed algorithm repeatedly calls `add_patches`; this extension deliberately delegates rather than replacing its math/API.

Generations 2 and 3 emitted exactly these cached IDs: **1, 2, 3, 4, 6, 7**. IDs **1/2** are checkpoint loaders and **3/4** are the final materialized MODEL/CLIP merges; 6 is unchanged negative text and 7 is the empty latent. The LoRA generation retained cached **1, 2, 3, 4, 7**. Therefore no checkpoint reload or merge repeat occurred. **The loader outputs were still cache-held in both modes: source CPU-RAM release in the live executor was not achieved.** No cache entries were deleted by this extension. Pressure-forced eviction of the final merge was not induced in this small test; reuse under extreme RAM pressure is not guaranteed.

Default RAM-pressure caching is the recommended starting mode on the constrained host. Classic caching offered no reuse advantage in this bounded test and intentionally keeps unchanged upstream outputs. It is available when retention is more important than host-RAM pressure, but it does not solve source release.

## Workflow and deployment

The actual workflow contained 93 nodes and 95 links, including ten loaders, nine MODEL merges and nine CLIP merges. A new `Sassy_Girls_Illustrious_2.0_ReleaseSources.json` was written next to the original. Original source contents were preserved. All link IDs/endpoints, node IDs, positions, ratios and unrelated nodes were compared; converter idempotence passed. Only IDs **2023** and **2022** enable materialization. Converted merge heights allow the added widget to be visible.

Both custom classes resolved through the running server's `/object_info`, with required input order and default-false booleans verified. Actual queued API prompts serialized both new classes successfully. Manual browser/visual-editor interaction was not automated; the original full workflow's unrelated third-party nodes were not executed.

The ten-checkpoint workflow was **not run**: measured source sizes imply about 63 GiB of MODEL+CLIP sources plus 6.3 GiB for the final independent pair, exceeding available physical RAM before scratch/VAEs/sampling. The graph remains available for a user-controlled experiment; no intermediate materialization or unsafe cache eviction was added. A separate build/save phase followed by a single-checkpoint generation workflow remains the reliable source-release option.

## Release files

```text
.gitignore
LICENSE
README.md
TEST_RESULTS.md
__init__.py
nodes.py
materialize.py
diagnostics.py
scripts/convert_workflow.py
scripts/integration_parity.py
scripts/integration_queue.py
tests/support.py
tests/test_stock_parity.py
tests/test_independence.py
tests/test_cache_reexecution.py
tests/test_workflow.py
```

Private workflow copies, logs and test images are excluded from git. The normal ComfyUI server stays in its original default cache mode; the isolated test server is stopped after validation.

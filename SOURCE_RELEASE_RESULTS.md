# Actual source release — 2026-11-08

The source-release implementation was first tested against an **isolated copy** of the installed ComfyUI checkout. After explicit user approval, the reviewed three-file integration and updated nodes were installed on the normal server and verified with the installed plugin stack enabled. Core installation is never automatic.

## What changed

`release_sources=True` now materializes independent native weights **and requests actual cache-output release** through the execution integration. MODEL/CLIP slots are removed after their last current consumers finish. VAE outputs remain independently accessible. The final merge's current output is kept. Old inactive variants of affected producers are retired; cache entries used by the active prompt are protected. The extension does not monkeypatch execution/cache classes or call global unloading/CUDA-cache clearing.

The cache entry retains its existing **two-field tuple interface**. Released-slot/producer metadata does not change tuple unpacking. Partial entries are not treated as complete loader hits; retained VAE sockets can still be read without executing the checkpoint loader. Ratios/checkpoint/toggle changes cause missing source sockets to reload normally.

## Environment and scope

- Same ComfyUI base, Python/PyTorch/Transformers/Tokenizers versions and two fp16 Illustrious checkpoints as [the historical validation](TEST_RESULTS.md).
- RTX 4060 Ti 16 GiB, 60.74 GiB host RAM.
- Native static patchers, `--disable-dynamic-vram`, original GPU/attention/precision flags.
- Dedicated loopback server; only this custom extension enabled. Default RAM-pressure cache first; then an actual restart with `--cache-classic`.
- Three queued 256×256, two-step Euler generations with changed seed/prompt, followed by a native post-merge LoRA pass at MODEL/CLIP strength .3.
- These tests establish execution, ownership and reuse. Two denoising steps are not an image-quality assessment.

## Actual cache release and CPU RAM

In both modes, the diagnostics observed **4,816/4,816 tracked CPU source tensor objects collected** for the combined upstream graph. The MODEL merge's overlapping snapshot observed **3,384/3,384** collected. These are overlapping observations, not additive counts. Actual RSS fell during the release callback:

| Cache | MODEL materialized, sources still present | After source cache release / collection | After first sampled image |
|---|---:|---:|---:|
| Default | 17.292 GiB | 11.733 GiB | 8.224 GiB |
| Classic | 17.300 GiB | 11.741 GiB | 8.230 GiB |

The later RSS reduction includes moving final weights to the GPU for sampling. The first reduction happened when source-cache ownership was removed and tracked source tensors were collected. No `unload_all_models()` or `torch.cuda.empty_cache()` was used as a substitute. CPU allocator retention and other runtime buffers mean RSS does not fall by the exact logical tensor-byte sum.

## Generations and retained final merges

All figures are per test-server process. Peak RSS is cumulative; CUDA figures describe the PyTorch allocator, separately from host RAM. Wall time includes one-second HTTP history polling.

| Cache | Generation | Seconds | RSS GiB | Peak RSS GiB | CUDA allocated / reserved GiB |
|---|---|---:|---:|---:|---:|
| Default | Baseline | — | 1.313 | 1.313 | 0 / 0 |
| Default | 1, build/merge/sample | 45.589 | 8.224 | 20.799 | 6.470 / 6.500 |
| Default | 2, changed seed/prompt | 1.035 | 8.226 | 20.799 | 6.470 / 6.656 |
| Default | 3, changed seed/prompt | 1.034 | 8.226 | 20.799 | 6.470 / 6.656 |
| Default | 4, post-merge LoRA | 6.067 | 11.399 | 20.799 | 6.470 / 6.656 |
| Classic | Baseline | — | 1.314 | 1.314 | 0 / 0 |
| Classic | 1, build/merge/sample | 40.224 | 8.230 | 20.801 | 6.470 / 6.500 |
| Classic | 2, changed seed/prompt | 1.033 | 8.231 | 20.801 | 6.470 / 6.656 |
| Classic | 3, changed seed/prompt | 1.032 | 8.231 | 20.801 | 6.470 / 6.656 |
| Classic | 4, post-merge LoRA | 6.065 | 11.416 | 20.801 | 6.470 / 6.656 |

Generations 2/3 cached exactly **3, 4, 6, 7**: final MODEL/CLIP merges, negative conditioning and the empty latent. Full checkpoint-loader entries **1/2 were absent**, while the VAE socket remained reusable. The LoRA generation cached **3, 4, 7**. Each mode logged exactly **two checkpoint builds, one MODEL materialization and one CLIP materialization** across all four generations. Source checkpoints were not reloaded and final merges were not rebuilt.

## Approved normal-server deployment

The normal server was restarted with its original default cache/GPU flags and installed plugins enabled. It collected all **4,816/4,816** tracked CPU source tensor objects; the overlapping MODEL snapshot collected **3,384/3,384**. RSS during the MODEL release callback fell from **18.428 to 14.551 GiB**. Later sampling changed device placement and allocator usage further. Absolute RSS differs from the isolated process's footprint; it does not measure logical weight bytes alone.

| Normal server, default cache | Seconds | RSS GiB | Peak RSS GiB | Cached nodes |
|---|---:|---:|---:|---|
| Baseline | — | 1.875 | 1.875 | — |
| Generation 1 | 40.695 | 12.135 | 21.345 | none |
| Generation 2 | 1.037 | 12.136 | 21.345 | 3, 4, 6, 7 |
| Generation 3 | 1.033 | 12.136 | 21.345 | 3, 4, 6, 7 |
| Generation 4, LoRA | 5.088 | 13.982 | 21.345 | 3, 4, 7 |

Exactly two checkpoints were built and each merge executed once. The source-loader entries were partial rather than complete cache hits; the retained VAE supported later decoding without checkpoint reload. The normal-server test therefore confirmed the same cache-release/reuse behavior with the user's installed plugins, rather than relying solely on the isolated whitelist test.

## Regression coverage

**27 tests passed** against the patched core. Besides the existing native weight-parity/independence tests, actual `PromptExecutor` tests verify:

- Source object collection and retained VAE across three generations in both cache implementations.
- A second branch and async work finish before their source outputs are released.
- Duplicate checkpoint nodes sharing one cache key remain valid.
- Ratio changes reload missing source slots; toggling off restores stock behavior.
- Enabling release on previously cached lazy merges also retires their inactive variants.
- Current-prompt references are trimmed even when pressure already evicted the global entry.
- A downstream exception leaves final cached merges usable without source reload.
- The existing two-field cache tuple interface remains intact.
- An unpatched executor fails before materializing when actual release is requested.

The explicit installer was tested on disposable original-file copies: read-only check, apply, installed-state check, revert and original-state check all passed. A read-only check confirmed the user's real core files still matched the original hashes. Syntax compilation passed.

## Limits and installation status

The ten-checkpoint workflow was not executed. Final-only release still requires enough build-time RAM/swap to reach and finish the final materialization. Enabling all intermediate materializations is a separate experiment: it can reduce source retention sooner but adds copies/work and intermediate dtype rounding. The original workflow and its converted copy were not changed in this revision.

Existing NaNs in six source CLIP-L last-layer tensors remain a documented input-data limitation; weights are not sanitized. Other branches/extensions/aliases may retain tensors, and severe RAM pressure can still evict final outputs. These cases are reported rather than represented as guaranteed source deallocation or permanent final-model pinning.

**The normal-core change was explicitly approved and installed.** Further installations still require explicit review/action. See [the integration explanation](integration/README.md), [the exact patch](integration/cached-output-release.patch), and [the hash-checking installer](scripts/install_cache_integration.py).

New files for this revision: `source_release.py`, `tests/test_source_release.py`, `scripts/install_cache_integration.py`, `integration/README.md`, `integration/base_sha256.json`, `integration/cached-output-release.patch`, and this report. `nodes.py`, `diagnostics.py`, the queue-test script and documentation were updated. Raw logs and images remain private and excluded from git.

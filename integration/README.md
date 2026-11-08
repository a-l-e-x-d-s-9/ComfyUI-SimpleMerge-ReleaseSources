# Reviewed ComfyUI source-cache integration

The `release_sources` boolean needs this integration for actual cache release. Installing the custom-node folder alone does not install or modify any core code. On an unpatched executor, `True` fails before materialization with an actionable error; `False` remains stock behavior. Direct Python calls have no executor cache and only materialize independent weights.

The patch is prepared against ComfyUI `c9ea36b9d2140c89b2ef84f93f428df474e72ba5`. The installer checks exact before/after SHA-256 hashes and refuses other versions or locally changed files. The existing separate `model_patcher.py` change is untouched.

## Why three core files change

- `comfy_execution/caching.py`: a cache entry records released output indices. Individual MODEL/CLIP slots can be dropped while preserving the VAE. Whole-node cache lookups reject partial entries; socket-aware lookups can still serve retained outputs. Re-executing a loader replaces its partial entry normally.
- `comfy_execution/graph.py`: a node can request upstream output release through its supplied `EXECUTION_LIST`. The executor defers each slot until all current consumers finish, including pending branches and async work. It also drops stale references from current-prompt input caches. No classes or cache methods are monkeypatched by the extension.
- `execution.py`: imports the cache-owned entry type, tags its producer, and releases startup prefetch results (including the async gather waiter's lifetime) before nodes run. This prevents prefetched sources from surviving the whole prompt after eviction.

The extension visits only its upstream MODEL/CLIP graph and requests those typed output slots. It retains its current output, while retiring inactive cached versions of affected producers that could still hold sources through old lazy merges. Cache keys used by the active prompt are protected. All other output types are preserved. The released slots remain absent across seed/prompt changes; final cached merges and a retained VAE can be reused. Changing a ratio, checkpoint or toggle invalidates the affected cache signatures; missing source slots then trigger ordinary loader execution. A new dynamic dependency can likewise reload a missing source output.

Release removes cache ownership; a live lazy branch, another extension or caller may still retain tensors. Diagnostics report tracked CPU source tensors collected and actual RSS before/after. There is no global model unloading or CUDA-cache clearing. External cache providers are rejected because their source retention cannot be managed here. `--cache-none` is rejected because it cannot retain the final merge.

## Check and explicit installation

Run from the real ComfyUI root using its interpreter:

```bash
# Read-only check. This is the default action.
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/install_cache_integration.py --comfy-root .

# Only after reviewing/authorizing the three-file core change:
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/install_cache_integration.py --comfy-root . --apply
```

Restart ComfyUI. The existing workflow and boolean widgets remain compatible. Keep the final pair enabled for the first test; intermediate materialization is a separate memory/rounding tradeoff and is not automatically enabled.

Revert only this reviewed integration:

```bash
venv/bin/python custom_nodes/ComfyUI-SimpleMerge-ReleaseSources/scripts/install_cache_integration.py --comfy-root . --revert
```

Restart afterward. Use `release_sources=False` on an unpatched executor.

## Remaining limit

Enabling only the final merges releases sources **after** those merges finish. It does not reduce the RAM needed to load all ten checkpoints and construct the final outputs in the first place. Keeping a final merge cached is still subject to the selected cache policy; memory pressure can evict it. These two constraints are not bypassed by dropping upstream cache ownership.

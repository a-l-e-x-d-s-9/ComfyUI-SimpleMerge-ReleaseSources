import logging
import sys
import time

import torch

try:
    import resource
except ImportError:
    resource = None

try:
    import psutil
except ImportError:
    psutil = None


def snapshot():
    result = {"time": time.monotonic()}
    if resource is not None:
        divisor = 2**30 if sys.platform == "darwin" else 2**20
        result["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    if psutil is not None:
        result.update(rss_gib=psutil.Process().memory_info().rss / 2**30,
                      available_gib=psutil.virtual_memory().available / 2**30)
    if torch.cuda.is_available():
        result.update(cuda_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                      cuda_reserved_gib=torch.cuda.memory_reserved() / 2**30)
    return result


def log_materialization(kind, before):
    after = snapshot()
    logging.info("SimpleMerge %s materialized in %.2fs; memory=%s; "
                 "source release is deferred until consumers finish when running with cache integration.",
                 kind, after["time"] - before["time"],
                 {key: round(value, 3) for key, value in after.items() if key != "time"})

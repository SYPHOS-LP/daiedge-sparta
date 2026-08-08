import torch
import contextlib
import math

_LOG_RANGES: bool = False
_RANGE_STATS: dict = {}


@contextlib.contextmanager
def range_logging_mode():
    global _LOG_RANGES, _RANGE_STATS
    _RANGE_STATS = {}
    _LOG_RANGES = True
    try:
        yield _RANGE_STATS
    finally:
        _LOG_RANGES = False


def _range_update(key: str, t: torch.Tensor):
    lo, hi = float(t.min()), float(t.max())
    if key not in _RANGE_STATS:
        _RANGE_STATS[key] = {"min": lo, "max": hi}
    else:
        _RANGE_STATS[key]["min"] = min(_RANGE_STATS[key]["min"], lo)
        _RANGE_STATS[key]["max"] = max(_RANGE_STATS[key]["max"], hi)


def print_range_stats():
    if not _RANGE_STATS:
        print("[range_stats] No data — call inside range_logging_mode().")
        return
    print("[range_stats] Observed input ranges and recommended FPGA loop bounds:")
    for key, s in _RANGE_STATS.items():
        lo, hi = s["min"], s["max"]
        print(f"  {key}: min={lo:.4e}  max={hi:.4e}")
        if key == "isqrt_x":
            halve = max(0, math.ceil(math.log2(hi)) - 1) if hi >= 2.0 else 0
            double = max(0, math.ceil(-math.log2(lo))) if lo > 0 and lo < 1.0 else 0
            print(f"    halving iters needed: {halve}+1  (default 14)")
            print(f"    doubling iters needed: {double}+1  (default 17)")
        elif key == "inv_x":
            halve = max(0, math.ceil(math.log2(hi)) - 1) if hi >= 2.0 else 0
            double = max(0, math.ceil(-math.log2(lo))) if lo > 0 and lo < 1.0 else 0
            print(f"    halving iters needed: {halve}+1  (default 10)")
            print(f"    doubling iters needed: {double}+1  (default 3)")

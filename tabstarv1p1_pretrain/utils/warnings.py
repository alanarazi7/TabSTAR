import builtins

import contextlib


@contextlib.contextmanager
def silence_baselines_prints():
    # Filter only LightGBM's noisy lines; let other prints through.
    noisy_markers = (
        "No further splits with positive gain, best gain",
        "Stopped training because there are no more leaves that meet the split",
    )
    orig_print = builtins.print
    def _filtered_print(*args, **kwargs):
        s = " ".join(str(a) for a in args)
        if any(m in s for m in noisy_markers):
            return
        return orig_print(*args, **kwargs)

    builtins.print = _filtered_print
    try:
        yield
    finally:
        builtins.print = orig_print

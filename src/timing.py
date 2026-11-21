"""
Wall-clock timing of training stages (never evaluation), merged into one
JSON file per concept:  <output_dir>/<prefix>_<name>/timings.json

Only stages that actually run are recorded; a stage skipped because its
output already exists keeps whatever time an earlier run recorded.

Python:
    with timed(path, "completion_rfm"):
        completion_rfm.train_completion_rfm(...)

Shell (for steps timed around a whole python call, e.g. joint training):
    python timing.py --path P --stage joint --seconds 123.4
"""
import contextlib
import json
import os
import time


def record(path, stage, seconds):
    timings = {}
    if os.path.exists(path):
        with open(path) as f:
            timings = json.load(f)
    timings[stage] = round(seconds, 1)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(timings, f, indent=2)
    os.replace(tmp, path)
    print(f"  [timing] {stage}: {seconds:.1f}s ({seconds / 60:.1f} min) -> {path}")


@contextlib.contextmanager
def timed(path, stage):
    start = time.perf_counter()
    yield
    record(path, stage, time.perf_counter() - start)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--seconds", type=float, required=True)
    args = p.parse_args()
    record(args.path, args.stage, args.seconds)

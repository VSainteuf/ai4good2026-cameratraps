"""Turn the runs you have done into flat tables you can analyse.

    uv run python -m iwildcam.summarise

Every finished run appends one JSON line to `results/runs.jsonl`. That file holds
everything, which is exactly what makes it awkward to read: the numbers you compare sit
next to per-epoch, per-camera and per-species lists. This script flattens it into three
CSVs under `results/summary/`:

* `runs.csv` -- one row per run: the settings, the val and test scores, how long it took.
* `by_config.csv` -- one row per setting with its seeds averaged, best score first. This
  is the table to read when deciding whether a change helped, because a difference
  smaller than the spread across seeds is not a difference.
* `per_class.csv` -- one row per run and species: F1 and how many test images that
  species had. This is where you see *which* animals a method helps.

A run that crashes never reaches `runs.jsonl`, and a method that dies on every third seed
is worth knowing about. So `logs/` is scanned too, and any per-run log whose run is
missing from the JSONL adds a row to `runs.csv` with `status = incomplete`, the number of
epochs it finished, and the last line of its traceback. Only files named the way
`train.py` names them (`...-seed<n>-<hash>.log`) are read, so a hand-named log or the
data-preparation log is left alone.

Nothing here opens `results/preds/*.npz`. Those hold one entry per test image and there
is one per run, so loading them together is gigabytes; every number below comes from the
per-run summaries. For per-image analysis, open one `.npz` at a time yourself.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .train_utils import JSONL, LOG_DIR, RESULTS

# One row per run. Fixed order, so the CSV looks the same whatever the runs contain, and
# so a column that only newer runs record still gets a place (left empty for the others).
RUN_COLUMNS = ["status", "split", "model", "pretrained", "size", "height", "seed",
               "epochs", "epochs_done", "batch_size", "lr", "weight_decay", "n_params",
               "n_classes", "n_train", "n_val", "n_test",
               "val_macro_f1_present", "test_macro_f1_present", "test_macro_f1",
               "test_accuracy", "test_n_classes_present", "seconds", "error",
               "run_key", "config_key", "pred_file", "log_file"]

# What gets averaged over the seeds of one setting.
METRICS = ["val_macro_f1_present", "test_macro_f1_present", "test_macro_f1",
           "test_accuracy", "seconds"]

# The settings that are the same for every seed of one config, kept beside the averages
# so `by_config.csv` is readable without joining back to `runs.csv`.
SETTINGS = ["split", "model", "pretrained", "size", "epochs"]

# Config fields worth a column of their own. The rest of the config stays in runs.jsonl.
FROM_CONFIG = ["lr", "weight_decay", "height"]


# --- reading what the runs left behind --------------------------------------------------

def load_runs(path: Path | None = None) -> list[dict]:
    """Read every completed run from `results/runs.jsonl`.

    Args:
        path: the JSONL file. Defaults to the shipped location.

    Returns:
        One dict per run, in the order they were written. Empty if the file is missing,
        which is the normal state before the first run.
    """
    path = path or JSONL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# `tee_console` starts each block with this line, so one log file may hold several blocks:
# the same config re-run appends rather than overwriting.
_BLOCK = re.compile(r"^=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)  (.*)$")
_KEYS = re.compile(r"^run_key (\w+)  config_key (\w+)")
_EPOCH = re.compile(r"^ +epoch +\d+ ")


def _parse_block(lines: list[str]) -> dict | None:
    """Pull what a single log block says about its run.

    Args:
        lines: the block's lines, without its `=== ...` header.

    Returns:
        A dict with the run's keys, config, epochs finished and error, or None if the
        block has no `run_key` line and so cannot be matched to a run.
    """
    keys, config = None, {}
    for i, line in enumerate(lines):
        m = _KEYS.match(line)
        if m:
            keys = (m[1], m[2])
            # `tee_console` writes the config as one JSON line directly underneath.
            if i + 1 < len(lines) and lines[i + 1].startswith("{"):
                config = json.loads(lines[i + 1])
            break
    if keys is None:
        return None

    body = [ln for ln in lines if ln.strip()]
    crashed = any("Traceback (most recent call last)" in ln for ln in body)
    return {"run_key": keys[0], "config_key": keys[1], "config": config,
            "epochs_done": sum(1 for ln in lines if _EPOCH.match(ln)),
            "error": body[-1].strip() if crashed and body else ""}


def scan_logs(log_dir: Path | None = None,
              known: set[str] | None = None) -> list[dict]:
    """Find runs that left a log but never reached `runs.jsonl`.

    These are the runs that crashed or were interrupted. Only files following the name
    `train.py` gives them are read, so a hand-named log, or the large data-preparation
    log, is skipped rather than parsed as a run.

    Args:
        log_dir: the folder of logs. Defaults to `logs/`.
        known: run keys already recorded in `runs.jsonl`, which are skipped.

    Returns:
        One dict per unfinished run. A run re-tried several times appears once, described
        by its most recent attempt.
    """
    log_dir = log_dir or LOG_DIR
    known = known or set()
    if not log_dir.exists():
        return []

    found: dict[str, dict] = {}
    for path in sorted(log_dir.glob("*-seed*.log")):
        blocks, current = [], None
        for line in path.read_text(errors="replace").splitlines():
            header = _BLOCK.match(line)
            if header:
                current = []
                blocks.append(current)
            elif current is not None:
                current.append(line)
        for block in blocks:
            run = _parse_block(block)
            if run is not None and run["run_key"] not in known:
                # Later blocks win: the newest attempt describes the run best.
                found[run["run_key"]] = {**run, "log_file": str(path)}
    return list(found.values())


# --- the tables -------------------------------------------------------------------------

def runs_table(rows: list[dict], incomplete: list[dict] | None = None) -> pd.DataFrame:
    """One row per run, complete ones first.

    Args:
        rows: the records from `runs.jsonl`.
        incomplete: the unfinished runs found by `scan_logs`.

    Returns:
        A DataFrame with `RUN_COLUMNS`, sorted by setting and then by seed.
    """
    out = []
    for r in rows:
        config = r.get("config", {})
        out.append({**{k: r.get(k) for k in RUN_COLUMNS},
                    **{k: r.get(k, config.get(k)) for k in FROM_CONFIG},
                    "status": "complete",
                    "epochs_done": len(r.get("history", [])),
                    "error": ""})
    for r in incomplete or []:
        config = r["config"]
        out.append({**{k: config.get(k) for k in RUN_COLUMNS if k in config},
                    **{k: config.get(k) for k in FROM_CONFIG},
                    "status": "incomplete",
                    "epochs_done": r["epochs_done"], "error": r["error"],
                    "run_key": r["run_key"], "config_key": r["config_key"],
                    "log_file": r["log_file"]})

    df = pd.DataFrame(out).reindex(columns=RUN_COLUMNS)
    sort_by = ["status"] + SETTINGS + ["seed"]
    return df.sort_values(sort_by, kind="stable").reset_index(drop=True)


def by_config(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per setting, its seeds averaged, best test score first.

    The standard deviation is over the seeds of that setting, so it is empty for a
    setting run once. Read it before believing a difference between two rows.

    Args:
        runs: the table from `runs_table`. Unfinished runs are left out, having no scores.

    Returns:
        A DataFrame with the settings, `n_seeds`, and the mean and standard deviation of
        each metric in `METRICS`.
    """
    df = runs[runs["status"] == "complete"].copy()
    if df.empty:
        return pd.DataFrame(columns=["config_key"] + SETTINGS + ["n_seeds"])

    # Runs recorded before `config_key` existed still have to group with their own seeds.
    # Falling back to the readable settings is what config_key stands for anyway.
    readable = df[SETTINGS].astype(str).agg("-".join, axis=1)
    df["config_key"] = df["config_key"].fillna("").mask(lambda s: s == "", readable)

    agg = {"n_seeds": ("seed", "count")}
    agg.update({f"{m}_{stat}": (m, stat) for m in METRICS for stat in ("mean", "std")})
    out = df.groupby(["config_key"] + SETTINGS, dropna=False).agg(**agg).reset_index()
    return out.sort_values("test_macro_f1_present_mean", ascending=False,
                           kind="stable").reset_index(drop=True)


_NAMES: dict[tuple[int, int], tuple[str, ...]] = {}


def class_names(min_per_camera: int, min_cameras: int) -> tuple[str, ...]:
    """The species names of a class set, in the order the model indexes them.

    Read from `data/metadata.csv.gz`, which ships with the repository, so this works with
    no images on disk. `data.py` is imported here rather than at the top of the file: you
    will be editing it, and a summary of your finished runs should still come out when it
    is halfway through a change.

    Args:
        min_per_camera: images a camera needs before it counts for a species.
        min_cameras: cameras a species needs to join the class set.

    Returns:
        The species names, or an empty tuple if they could not be read.
    """
    key = (min_per_camera, min_cameras)
    if key not in _NAMES:
        try:
            from .data import load_task
            _NAMES[key] = tuple(load_task(min_per_camera, min_cameras,
                                          require_files=False).classes)
        except Exception as e:  # noqa: BLE001 -- the names are a nicety, not the point
            print(f"note: could not read the species names ({e}); "
                  f"per_class.csv will carry class indices only")
            _NAMES[key] = ()
    return _NAMES[key]


def per_class_table(rows: list[dict], names: bool = True) -> pd.DataFrame:
    """One row per run and species: its F1 and how many test images it had.

    A species with a handful of test images swings a macro F1 far more than its share of
    the data, which is why `support` sits next to `f1` here.

    Args:
        rows: the records from `runs.jsonl`.
        names: look the species names up from the metadata. False keeps indices only.

    Returns:
        A DataFrame with `run_key`, `config_key`, the setting, `class_index`, `species`,
        `f1` and `support`.
    """
    out = []
    warned = False
    for r in rows:
        f1s, support = r.get("test_per_class_f1", []), r.get("test_support", [])
        labels = class_names(r.get("min_per_camera", 10),
                             r.get("min_cameras", 5)) if names else ()
        if labels and len(labels) != len(f1s) and not warned:
            print(f"note: run {r['run_key'][:8]} scored {len(f1s)} classes but the "
                  f"metadata gives {len(labels)}; leaving those species unnamed")
            warned = True
        for i, (f1, n) in enumerate(zip(f1s, support)):
            out.append({"run_key": r.get("run_key"), "config_key": r.get("config_key"),
                        "split": r.get("split"), "model": r.get("model"),
                        "pretrained": r.get("pretrained"), "seed": r.get("seed"),
                        "class_index": i,
                        "species": labels[i] if i < len(labels) else "",
                        "f1": f1, "support": n})
    return pd.DataFrame(out)


# --- CLI --------------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point. Writes the three CSVs and prints the by-config table."""
    ap = argparse.ArgumentParser(description="Flatten your runs into CSVs to analyse.")
    ap.add_argument("--runs", type=Path, default=JSONL,
                    help="the JSONL train.py appends to (default: results/runs.jsonl)")
    ap.add_argument("--logs", type=Path, default=LOG_DIR,
                    help="folder of per-run logs, read for runs that never finished")
    ap.add_argument("--out", type=Path, default=RESULTS / "summary",
                    help="where the CSVs go (default: results/summary)")
    ap.add_argument("--no-logs", action="store_true",
                    help="skip the logs; report only runs that finished")
    ap.add_argument("--no-names", action="store_true",
                    help="skip the species names, which are read from the metadata")
    a = ap.parse_args()

    rows = load_runs(a.runs)
    incomplete = [] if a.no_logs else scan_logs(a.logs, {r["run_key"] for r in rows})
    if not rows and not incomplete:
        raise SystemExit(f"no runs in {a.runs} and none found in {a.logs} -- "
                         f"run `python -m iwildcam.train --config configs/fast.yaml` first")

    runs = runs_table(rows, incomplete)
    configs = by_config(runs)
    per_class = per_class_table(rows, names=not a.no_names)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, table in [("runs", runs), ("by_config", configs),
                        ("per_class", per_class)]:
        table.to_csv(a.out / f"{name}.csv", index=False, float_format="%.6g")

    print(f"{len(rows)} finished runs from {a.runs}"
          + (f", {len(incomplete)} unfinished from {a.logs}" if incomplete else "")
          + f"\nwrote runs.csv, by_config.csv, per_class.csv to {a.out}\n")
    if not configs.empty:
        # Two rows can look identical here and still be different runs -- they differ in
        # something the columns shown do not carry -- so the config key leads the table.
        # A run too old to carry one gets `(none)`; `by_config.csv` names it in full.
        key = configs["config_key"].astype(str)
        show = configs.assign(
            config=key.str[:8].where(key.str.fullmatch(r"[0-9a-f]{16}"), "(none)"))
        print(show[["config", "split", "model", "pretrained", "size", "epochs",
                    "n_seeds", "test_macro_f1_present_mean",
                    "test_macro_f1_present_std"]].to_string(index=False,
                                                            float_format="%.4f"))
    if not runs.empty and (runs["status"] == "incomplete").any():
        print("\nunfinished:")
        bad = runs[runs["status"] == "incomplete"]
        print(bad[["split", "model", "seed", "epochs_done", "epochs", "error"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()

"""What `iwildcam/summarise.py` promises about the tables it writes.

Built on a handful of made-up runs rather than on your `results/`, so it passes on day
one and does not change meaning as you add runs.

    uv run --extra dev python -m pytest tests/test_summarise.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iwildcam import summarise

# Two seeds of one setting, plus a second setting, cut down to the fields the tables use.
RUNS = [
    {"run_key": "aaaa1111", "config_key": "cfgA", "split": "official_ood",
     "model": "resnet18", "pretrained": True, "size": 224, "seed": 0, "epochs": 2,
     "batch_size": 128, "n_params": 11207292, "min_per_camera": 10, "min_cameras": 5,
     "n_classes": 3, "val_macro_f1_present": 0.40, "test_macro_f1_present": 0.30,
     "test_macro_f1": 0.28, "test_accuracy": 0.47, "seconds": 63.7,
     "test_per_class_f1": [0.1, 0.2, 0.3], "test_support": [10, 20, 30],
     "history": [{"epoch": 1}, {"epoch": 2}],
     "config": {"lr": 0.0001, "weight_decay": 0.0001, "height": 224}},
    {"run_key": "aaaa2222", "config_key": "cfgA", "split": "official_ood",
     "model": "resnet18", "pretrained": True, "size": 224, "seed": 1, "epochs": 2,
     "batch_size": 128, "n_params": 11207292, "min_per_camera": 10, "min_cameras": 5,
     "n_classes": 3, "val_macro_f1_present": 0.44, "test_macro_f1_present": 0.34,
     "test_macro_f1": 0.32, "test_accuracy": 0.49, "seconds": 64.1,
     "test_per_class_f1": [0.2, 0.3, 0.4], "test_support": [10, 20, 30],
     "history": [{"epoch": 1}, {"epoch": 2}],
     "config": {"lr": 0.0001, "weight_decay": 0.0001, "height": 224}},
    {"run_key": "bbbb1111", "config_key": "cfgB", "split": "random_burst",
     "model": "resnet50", "pretrained": True, "size": 224, "seed": 0, "epochs": 2,
     "batch_size": 128, "n_params": 23528522, "min_per_camera": 10, "min_cameras": 5,
     "n_classes": 3, "val_macro_f1_present": 0.70, "test_macro_f1_present": 0.63,
     "test_macro_f1": 0.60, "test_accuracy": 0.71, "seconds": 120.0,
     "test_per_class_f1": [0.6, 0.6, 0.7], "test_support": [10, 20, 30],
     "history": [{"epoch": 1}, {"epoch": 2}],
     "config": {"lr": 0.0001, "weight_decay": 0.0001, "height": 224}},
]

# A run that died in its second epoch: a log block, and no line in runs.jsonl.
CRASHED = """
=== 2026-09-09 10:00:00  iwildcam/train.py --config configs/fast.yaml
run_key cccc3333  config_key cfgC
{"split": "official_ood", "model": "resnet18", "seed": 2, "epochs": 10, "size": 224}

  epoch  1  loss 0.783  val macroF1(present) 0.394  48.3s
Traceback (most recent call last):
  File "train.py", line 1, in <module>
RuntimeError: CUDA out of memory
"""


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a runs.jsonl and a logs folder holding one crashed run."""
    runs = tmp_path / "runs.jsonl"
    runs.write_text("".join(json.dumps(r) + "\n" for r in RUNS))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "official_ood-resnet18-224-pre-e10-seed2-cccc3333.log").write_text(CRASHED)
    # Neither of these is a run: a hand-named log and a stray file. Both must be ignored.
    (logs / "ablation.log").write_text("=== masked ===\n  epoch  1  loss 0.1\n")
    (logs / "README.md").write_text("# not a log\n")
    return runs, logs


def test_finished_runs_become_one_row_each(tmp_path):
    runs, _ = _fixture(tmp_path)
    df = summarise.runs_table(summarise.load_runs(runs))
    assert len(df) == 3
    assert set(df["status"]) == {"complete"}
    assert list(df.columns) == summarise.RUN_COLUMNS
    # `lr` is recorded inside the nested config, not beside the scores.
    assert df["lr"].tolist() == [0.0001] * 3
    assert df["epochs_done"].tolist() == [2, 2, 2]


def test_crashed_run_is_reported_not_dropped(tmp_path):
    runs, logs = _fixture(tmp_path)
    rows = summarise.load_runs(runs)
    incomplete = summarise.scan_logs(logs, {r["run_key"] for r in rows})
    assert [r["run_key"] for r in incomplete] == ["cccc3333"]

    df = summarise.runs_table(rows, incomplete)
    bad = df[df["status"] == "incomplete"]
    assert len(bad) == 1
    assert bad["epochs_done"].iloc[0] == 1          # it managed one of ten
    assert "CUDA out of memory" in bad["error"].iloc[0]
    assert pd.isna(bad["test_macro_f1_present"].iloc[0])   # never got a score


def test_finished_runs_are_not_scanned_from_logs(tmp_path):
    """A run already in runs.jsonl must not appear twice."""
    runs, logs = _fixture(tmp_path)
    (logs / "official_ood-resnet18-224-pre-e2-seed0-aaaa1111.log").write_text(
        "\n=== 2026-09-09 09:00:00  train.py\nrun_key aaaa1111  config_key cfgA\n{}\n")
    rows = summarise.load_runs(runs)
    found = summarise.scan_logs(logs, {r["run_key"] for r in rows})
    assert "aaaa1111" not in [r["run_key"] for r in found]


def test_seeds_are_averaged_per_setting(tmp_path):
    runs, _ = _fixture(tmp_path)
    df = summarise.by_config(summarise.runs_table(summarise.load_runs(runs)))
    assert len(df) == 2
    # Best setting first.
    assert df["config_key"].tolist() == ["cfgB", "cfgA"]
    assert df["n_seeds"].tolist() == [1, 2]
    a = df[df["config_key"] == "cfgA"].iloc[0]
    assert a["test_macro_f1_present_mean"] == 0.32           # (0.30 + 0.34) / 2
    assert a["test_macro_f1_present_std"] > 0
    # One seed cannot have a spread, and saying 0 would claim it had none.
    assert pd.isna(df[df["config_key"] == "cfgB"].iloc[0]["test_macro_f1_present_std"])


def test_unfinished_runs_are_left_out_of_the_averages(tmp_path):
    runs, logs = _fixture(tmp_path)
    rows = summarise.load_runs(runs)
    incomplete = summarise.scan_logs(logs, {r["run_key"] for r in rows})
    df = summarise.by_config(summarise.runs_table(rows, incomplete))
    assert "cfgC" not in df["config_key"].tolist()


def test_per_class_is_one_row_per_run_and_species(tmp_path):
    runs, _ = _fixture(tmp_path)
    df = summarise.per_class_table(summarise.load_runs(runs), names=False)
    assert len(df) == 3 * 3                     # three runs, three classes each
    assert df["support"].tolist() == [10, 20, 30] * 3
    assert df[df["run_key"] == "aaaa1111"]["f1"].tolist() == [0.1, 0.2, 0.3]


def test_no_runs_yet_is_not_an_error(tmp_path):
    assert summarise.load_runs(tmp_path / "missing.jsonl") == []
    assert summarise.scan_logs(tmp_path / "missing") == []


def test_a_new_hyperparameter_gets_its_own_column(tmp_path):
    """A knob added to the YAML must reach the tables without editing summarise.py."""
    runs = [dict(r) for r in RUNS[:2]]
    for r, alpha in zip(runs, [0.2, 0.8]):
        r["config"] = {**r["config"], "mixup_alpha": alpha,
                       "model_kwargs": {"dropout": 0.5},
                       # Recorded in the config, but cannot change the result.
                       "device": "cuda", "num_workers": 8}
        r["config_key"] = f"cfg{alpha}"
        r["run_key"] = f"rk{alpha}"
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in runs))

    df = summarise.runs_table(summarise.load_runs(path))
    assert df["mixup_alpha"].tolist() == [0.2, 0.8]
    assert df["model_kwargs.dropout"].tolist() == [0.5, 0.5]   # nested, flattened
    assert "device" not in df.columns                          # cannot change a result
    assert "num_workers" not in df.columns
    # The fixed columns keep their order, so the CSV stays predictable.
    assert list(df.columns)[:len(summarise.RUN_COLUMNS)] == summarise.RUN_COLUMNS

    configs = summarise.by_config(df)
    assert configs["mixup_alpha"].tolist() == [0.8, 0.2]       # best score first


def test_runs_without_a_recorded_config_still_summarise(tmp_path):
    """Older runs stored no config. They lose the knob columns, not their row."""
    old = {k: v for k, v in RUNS[0].items() if k != "config"}
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(old) + "\n")
    df = summarise.runs_table(summarise.load_runs(path))
    assert len(df) == 1
    assert pd.isna(df["lr"].iloc[0])
    assert df["test_macro_f1_present"].iloc[0] == 0.30

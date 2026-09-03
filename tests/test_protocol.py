"""The evaluation protocol, written as executable rules.

Not a developer test suite. The class is evaluated on 48 cameras the model has never
seen, students WILL modify `iwildcam/data.py`, and the README requires this file to keep
passing afterwards. Each test is one rule turned into an assertion, so a broken rule
fails here, not as an inflated, wrong macro F1 later.

Most of these check the shipped folds themselves, reading `data/metadata.csv.gz` (5.0 MB)
but no images, so the suite runs on day one before anything is downloaded.

    uv run --extra dev python -m pytest tests/ -q     # normal way
    uv run python -m tests.test_protocol              # no pytest needed
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from iwildcam.data import (FOLDS, Task, UnlabelledDataset, build_transform, load_fold,
                          load_task, load_taxonomy, official_test_cameras,
                          unlabelled_frames)

_REAL: Task | None = None

def real_task() -> Task:
    """The shipped task, loaded once and reused. Needs the metadata, not the images."""
    global _REAL
    if _REAL is None:
        _REAL = load_task(require_files=False)
    return _REAL

def toy_task(n_cameras: int = 20, n_species: int = 6, burst_len: int = 4,
             bursts_per_cam: int = 10, seed: int = 0) -> Task:
    """A small synthetic task with the same shape as the real one: cameras, bursts,
    species, and empty frames -- small enough to build and split in milliseconds."""
    rng = np.random.default_rng(seed)
    rows, empties = [], []
    for cam in range(n_cameras):
        for s in range(bursts_per_cam):
            sp = f"species_{rng.integers(0, n_species)}"
            bid = f"cam{cam}_burst{s}"
            for f in range(burst_len):
                rows.append({"image_id": f"{bid}_{f}", "file_name": f"{bid}_{f}.jpg",
                             "location": cam, "burst_id": bid, "category": sp})
        for e in range(3):
            empties.append({"image_id": f"cam{cam}_e{e}", "file_name": f"cam{cam}_e{e}.jpg",
                            "location": cam, "burst_id": f"cam{cam}_eburst{e}",
                            "category": "empty"})
    df = pd.DataFrame(rows)
    classes = sorted(df["category"].unique())
    df["y"] = df["category"].map({c: i for i, c in enumerate(classes)}).astype("int64")
    return Task(df=df, empty=pd.DataFrame(empties), classes=classes,
                image_dir=Path("/nonexistent"))

# shared assertion helpers -- not tests themselves, called from several tests below
def _partition(task: Task, fold: dict) -> None:
    all_idx = np.concatenate([fold["train"], fold["val"], fold["test"]])
    assert len(all_idx) == len(set(all_idx.tolist())), "an index is in two splits at once"
    assert len(all_idx) == len(task.df), "some images vanished, or were duplicated"
    assert all(len(fold[k]) for k in fold), "a split is empty"

def _no_split_overlap(values: np.ndarray, fold: dict, what: str, pairwise: bool = False) -> None:
    tr, va, te = set(values[fold["train"]]), set(values[fold["val"]]), set(values[fold["test"]])
    assert not (tr & te), f"a {what} appears in both train and test"
    assert not (tr & va), f"a {what} appears in both train and val"
    if pairwise:
        assert not (va & te), f"a {what} appears in both val and test"

def test_official_ood_holds_up_as_a_protocol():
    """The headline fold. A camera on both sides, or a burst split across it, would
    inflate macro F1 without failing anything else. The fold ships as a column of the
    metadata, so this checks the shipped data, not code that builds it."""
    task = real_task()
    fold = load_fold(task, "official_ood")
    _partition(task, fold)
    _no_split_overlap(task.camera, fold, "camera", pairwise=True)
    _no_split_overlap(task.df[task.burst_col].to_numpy(), fold, "burst")
    assert set(task.camera[fold["test"]]) <= set(official_test_cameras()), \
        "a test camera is not one of the official 48"

def test_random_burst_is_in_distribution_and_burst_respecting():
    """The reference the transfer gap is measured against, so it has to be wrong in
    exactly one way and right in the other: cameras MUST appear on both sides (that is
    what makes it in-distribution) and bursts must NOT (a burst leak would inflate the
    reference and make the headline gap look smaller than it is)."""
    task = real_task()
    fold = load_fold(task, "random_burst")
    _partition(task, fold)
    _no_split_overlap(task.df[task.burst_col].to_numpy(), fold, "burst")
    assert set(task.camera[fold["train"]]) & set(task.camera[fold["test"]]), \
        "cameras do not overlap train/test -- this is no longer in-distribution"

def test_folds_do_not_move_when_images_are_missing():
    """The whole reason the folds ship as data. A student with part of the archive
    prepared must put every image on the same side as a student with all of it --
    otherwise their numbers are not comparable to anyone's."""
    task = real_task()
    part = load_task(require_files=False)
    keep = np.arange(len(part.df)) % 5 != 0          # a student with 80% of the images
    part.df = part.df.iloc[keep].reset_index(drop=True)
    for name in FOLDS:
        full_side = dict(zip(task.df["file_name"], task.df[name]))
        part_side = dict(zip(part.df["file_name"], part.df[name]))
        assert all(full_side[f] == s for f, s in part_side.items()), \
            f"{name} moved images between splits when some were missing"

def test_official_cameras_never_train_under_any_class_filter():
    """The fold columns are fixed, but the class set is not: `min_per_camera` and
    `min_cameras` are yours to change. Two of the official 48 cameras have no images in
    the default 60-class task, so a fold built from the cameras that task happens to
    contain would quietly file them as training data and leak them the moment somebody
    widened the class set."""
    off = set(official_test_cameras())
    for min_per_camera, min_cameras in ((10, 5), (5, 3), (1, 1)):
        task = load_task(min_per_camera, min_cameras, require_files=False)
        fold = load_fold(task, "official_ood")
        for split in ("train", "val"):
            leaked = set(task.camera[fold[split]]) & off
            assert not leaked, \
                f"official test cameras {sorted(leaked)} are in {split} at " \
                f"({min_per_camera}, {min_cameras})"

def test_unknown_fold_names_fail_loudly():
    """A typo in `split:` must not fall through to a default fold and silently report a
    number from the wrong protocol."""
    task = real_task()
    for bad in ("random_seq", "camera_ood", "official"):
        try:
            load_fold(task, bad)
            raise AssertionError(f"load_fold accepted {bad!r}")
        except ValueError:
            pass

def test_unlabelled_frames_is_safe_target_domain_data():
    """A camera leak here trains "target-domain" adaptation on cameras it should never
    see; a label leak lets "unsupervised" adaptation quietly use target labels; and a
    tuple return (instead of a bare tensor) would let `for x, y in loader` silently
    unpack the wrong thing instead of raising. Checked on one real item from a tiny
    synthetic image in a temp dir -- the rest of the suite still touches no images."""
    task = toy_task()
    cams = [0, 1, 2]
    ds = unlabelled_frames(task, cameras=cams)
    assert len(ds) == (task.empty["location"].isin(cams)).sum()
    src = task.empty[task.empty["location"].isin(cams)]
    assert set(src["category"].unique()) == {"empty"}
    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (8, 8)).save(Path(d) / "img.jpg")
        item = UnlabelledDataset(Path(d), ["img.jpg"], build_transform(8))[0]
    assert isinstance(item, torch.Tensor) and not isinstance(item, (tuple, list)), \
        "unlabelled dataset returned a tuple -- a species label leaked through"

def test_domain_indexing_is_contiguous_and_stable():
    """Camera ids are sparse (0..551 with gaps); a non-contiguous `domain` would index a
    domain-adversarial head out of bounds. It must also be fixed once for the whole task:
    if it were derived per fold, one camera would get different indices in train and in
    test and a trained adversarial head would be meaningless."""
    task = real_task()
    dom = task.domain.copy()
    assert dom.min() == 0 and dom.max() == task.n_domains - 1
    assert set(np.unique(dom).tolist()) == set(range(task.n_domains))
    fold = load_fold(task, "official_ood")
    for a, b in (("train", "val"), ("train", "test")):
        shared = set(task.camera[fold[a]]) & set(task.camera[fold[b]])
        for cam in list(shared)[:20]:
            ia = np.unique(task.domain[fold[a]][task.camera[fold[a]] == cam])
            ib = np.unique(task.domain[fold[b]][task.camera[fold[b]] == cam])
            assert np.array_equal(ia, ib), f"camera {cam} has different domain indices"
    assert np.array_equal(dom, task.domain), "reading a fold changed the domain mapping"

def test_class_set_has_exactly_60_species_and_no_junk():
    """A junk category (an annotator fallback like "unknown", or a marker like
    "start"/"end") in the class set makes the model learn a category that does not
    exist and score near zero on it, hurting the headline for a reason unrelated to
    transfer. And a class
    missing from the taxonomy would fail the hierarchical-label join for that species
    ("get the order right even if the species is wrong") -- silently, not by raising."""
    task = load_task(require_files=False)
    assert task.n_classes == 60
    junk = {"motorcycle", "unknown", "unidentifiable", "misfire", "start", "end"}
    assert not (set(task.classes) & junk)
    assert not any(c.startswith("unknown ") for c in task.classes)
    missing = set(task.classes) - set(load_taxonomy().index)
    assert not missing, f"classes missing from the taxonomy: {missing}"

def test_official_test_cameras_is_read_only():
    """This function previously wrote a file as a side effect of reading one. If that
    regressed, calling it during evaluation could silently modify the shipped data
    directory every fold is built from."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    before = {p: p.stat().st_mtime_ns for p in data_dir.iterdir()}
    cams = official_test_cameras()
    after = {p: p.stat().st_mtime_ns for p in data_dir.iterdir()}
    assert len(cams) == 48 and all(isinstance(c, int) for c in cams)
    assert before == after, "official_test_cameras() modified the data directory"

def main() -> None:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()

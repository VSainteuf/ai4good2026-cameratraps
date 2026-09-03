#!/usr/bin/env python
"""Turn the downloaded iWildCam archive into a folder of JPEGs the training code reads.

You supply the archive; this resizes it. Get it from Kaggle -- competition
`iwildcam-2020-fgvc7`, which needs an account and accepting the competition rules:

    kaggle competitions download -c iwildcam-2020-fgvc7    # ~112 GB, hours
    unzip iwildcam-2020-fgvc7.zip                          # 111 GB, both halves

Then point this script at the folder holding `train/`:

    uv run python prepare.py --source ~/iWildCam2020 --check       # count what is there
    uv run python prepare.py --source ~/iWildCam2020 --height 224  # 6.7 GB, iterate on this
    uv run python prepare.py --source ~/iWildCam2020 --height 448  # 23.7 GB, report on this

**Both halves are prepared.** `train/` (217,959 images) is the labelled half, and the task
is built from it. `test/` (62,894 images) is the 2020 competition's own held-out set: its
labels were never released, so nothing can be trained on it or scored against it. It is
62,894 real camera-trap frames, and unlabelled frames are what domain adaptation,
self-supervised pretraining and consistency training run on -- so they are resized into
`data/images_unlabelled_h*/` and left there for you to use. `--skip-unlabelled` turns that
off, and a source folder with no `test/` directory just skips it.

Be clear about what that pool is and is not: its 91 cameras appear **nowhere else in the
dataset**, so it is not target-domain data for the 48 cameras you are scored on. It is more
frames and more deployments to learn a camera-invariant representation from, not a preview
of the test set.

Disk, end to end: 112 GB (zip) + 111 GB (unzipped) at the peak, then 30 GB once the zip and
the originals are deleted and both heights of both halves are built.

**Why resize at all**, when the loader can read the originals directly (set `image_dir:`
in a config and skip this script)? Because a 1920x1080 JPEG decodes at about 54 images/s
per core and a 224px one at about 1,640 -- 30x. On 132k images that is roughly five
minutes of pure CPU per epoch against ten seconds, before any I/O and before the GPU does
anything. The resize is the whole saving; nothing here packs, indexes or compresses beyond
plain JPEG in a plain folder, so you can `ls` it and open one.

**It is resumable.** Kill it whenever. Each image is written to a temporary name and
renamed, so a half-written file never survives, and a re-run skips what is already there.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
METADATA = DATA / "metadata.csv.gz"

# The re-encode settings. Do not change them casually: every baseline in the README was
# measured on images produced exactly this way, so a different quality or resampling
# filter makes those numbers no longer comparable to yours.
JPEG_QUALITY = 90
RESAMPLE = Image.BILINEAR

# Measured average bytes per re-encoded image, for the disk projection.
BYTES_PER_IMAGE = {224: 25_500, 448: 90_000}


def out_dir(height: int, unlabelled: bool = False) -> Path:
    """Where resized images go. The two halves are kept in separate folders.

    Args:
        height: output height in pixels.
        unlabelled: True for the competition test half, which carries no labels.

    Returns:
        The folder path. `data/images_h224/` matches `metadata.csv.gz` exactly, so nothing
        that counts or indexes it has to filter; the unlabelled pool sits beside it.
    """
    return DATA / (f"images_unlabelled_h{height}" if unlabelled else f"images_h{height}")


def find_images(source: Path) -> Path:
    """The folder holding the labelled JPEGs, given whatever the user pointed us at."""
    for candidate in (source / "train", source):
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    raise SystemExit(
        f"no .jpg files under {source} or {source / 'train'}.\n"
        f"Point --source at the folder that holds the unzipped `train/` directory.")


def find_unlabelled(source: Path) -> Path | None:
    """The competition's own test half, if it was unzipped. None if it was not.

    Args:
        source: the folder --source points at.

    Returns:
        The `test/` folder, or None. Missing is not an error: earlier versions of this
        repo told people to unzip `train/` only.
    """
    candidate = source / "test"
    return candidate if candidate.is_dir() and any(candidate.glob("*.jpg")) else None


def projected_gb(height: int, n: int) -> float:
    per = BYTES_PER_IMAGE.get(height) or 25_500 * (height / 224) ** 1.82
    return n * per / 1e9


def resize_one(src: Path, dst: Path, height: int) -> str:
    """Read one JPEG, write it back at `height`. Returns "" on success, else the error."""
    if not src.exists():
        # Not fatal: the archive may be short of a few images, and `load_task` drops
        # whatever is not in the folder. Counted and reported at the end.
        return "missing from the archive"
    try:
        with Image.open(src) as im:
            # draft() lets libjpeg downscale during decode, so a 1920x1080 frame is never
            # fully decoded just to be thrown away. Worth 3.3x on its own.
            im.draft("RGB", (im.width * height // max(im.height, 1), height))
            out = im.convert("RGB")
        if out.height != height:
            out = out.resize((max(1, round(out.width * height / out.height)), height),
                             RESAMPLE)
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=False)
        # Write then rename, so a kill mid-write cannot leave a truncated file that the
        # next run would mistake for finished work.
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, dst)
        return ""
    except Exception as e:                                        # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def build(src_dir: Path, names: list[str], height: int, workers: int,
          unlabelled: bool = False) -> None:
    dst_dir = out_dir(height, unlabelled)
    tag = f"h{height} unlabelled" if unlabelled else f"h{height}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    done = {p.name for p in dst_dir.iterdir() if p.suffix == ".jpg"}
    # Sorted, not metadata order. Directory entries on disk are ordered by name, so this
    # reads the source roughly sequentially. On a fast local disk it makes no difference;
    # on a slow, network-mounted or NTFS-via-FUSE one it is worth several times.
    todo = sorted(n for n in names if n not in done)
    if done:
        print(f"  {tag}: {len(done):,} already there, {len(todo):,} to do")
    if not todo:
        print(f"  {tag}: nothing to do")
        return

    errors: dict[str, int] = {}
    bar = tqdm(total=len(todo), desc=tag, unit="img", smoothing=0.05)
    with ThreadPoolExecutor(workers) as ex:
        # PIL releases the GIL inside decode and encode, so threads are enough. Only a
        # window of jobs is ever submitted: `Executor.map` would queue all 201k at once.
        queue = iter(todo)
        pending = deque(ex.submit(resize_one, src_dir / n, dst_dir / n, height)
                        for n in islice(queue, 4 * workers))
        while pending:
            err = pending.popleft().result()
            for nxt in islice(queue, 1):
                pending.append(ex.submit(resize_one, src_dir / nxt, dst_dir / nxt, height))
            if err:
                errors[err.split(":")[0]] = errors.get(err.split(":")[0], 0) + 1
            bar.update(1)
    bar.close()
    written, total = 0, 0
    with os.scandir(dst_dir) as it:                 # one pass, not two
        for e in it:
            if e.name.endswith(".jpg"):
                written += 1
                total += e.stat().st_size
    print(f"  {tag}: {written:,} images, {total / 1e9:.2f} GB in {dst_dir}")
    if errors:
        print(f"  {tag}: {sum(errors.values()):,} failed -- "
              + ", ".join(f"{k} x{v}" for k, v in sorted(errors.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, type=Path,
                    help="folder holding the unzipped archive's `train/` and `test/` "
                         "directories")
    ap.add_argument("--height", type=int, action="append",
                    help="output height; repeatable. Default: 224")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8),
                    help="parallel decode/encode jobs. The default is fine; on a slow or "
                         "network-mounted source disk the disk is the limit, not this.")
    ap.add_argument("--check", action="store_true",
                    help="report coverage and disk, then stop")
    ap.add_argument("--skip-unlabelled", action="store_true",
                    help="prepare only the labelled half. The unlabelled `test/` images "
                         "are for adaptation and pretraining, not for scoring, so this "
                         "costs you research options rather than correctness.")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N images of each half (a quick end-to-end test)")
    a = ap.parse_args()

    heights = sorted(set(a.height or [224]))
    if not METADATA.exists():
        raise SystemExit(f"{METADATA} is missing -- it ships with the repository.")

    names = list(pd.read_csv(METADATA, usecols=["file_name"])["file_name"])
    if a.limit:
        names = names[:a.limit]
    source = a.source.expanduser()
    src_dir = find_images(source)

    # The unlabelled half has no shipped metadata -- there is no list of it anywhere in
    # this repo -- so it has to be enumerated from the folder itself.
    unlab_dir = None if a.skip_unlabelled else find_unlabelled(source)
    unlab_names: list[str] = []
    if unlab_dir is not None:
        unlab_names = sorted(p.name for p in unlab_dir.iterdir() if p.suffix == ".jpg")
        if a.limit:
            unlab_names = unlab_names[:a.limit]

    print(f"source     {src_dir}")
    if unlab_dir is not None:
        print(f"unlabelled {unlab_dir}  ({len(unlab_names):,} images, never labelled)")
    elif not a.skip_unlabelled:
        print(f"unlabelled no `test/` folder under {source} -- unzip that half too if you "
              f"want the unlabelled pool")
    have = len(names)
    if a.check:
        # Only --check walks the source directory. It is 218k entries, which takes a while
        # on a slow or network-mounted disk, and a normal run has no use for it: a missing
        # file is just one more error in the report at the end.
        present = {p.name for p in src_dir.iterdir() if p.suffix == ".jpg"}
        have = sum(n in present for n in names)
        print(f"           {len(present):,} jpg files; {have:,} of the {len(names):,} "
              f"images the task needs ({100 * have / len(names):.1f}%)")
        if have < len(names):
            print(f"           {len(names) - have:,} missing -- they get skipped, and "
                  f"`load_task` drops them from the task")

    # One entry per folder to write: (height, is it the unlabelled half, how many images).
    jobs = [(h, False, have) for h in heights]
    jobs += [(h, True, len(unlab_names)) for h in heights if unlab_names]

    def already_gb(d: Path) -> float:
        return sum(p.stat().st_size for p in d.iterdir()) / 1e9 if d.exists() else 0.0

    need = sum(projected_gb(h, n) - already_gb(out_dir(h, u)) for h, u, n in jobs)
    free = shutil.disk_usage(DATA).free / 1e9
    print("projected  " + ", ".join(
        f"h{h}{' unlabelled' if u else ''} {projected_gb(h, n):.1f} GB" for h, u, n in jobs)
        + f"  ({need:.1f} GB still to write, {free:.1f} GB free)")
    if a.check:
        return
    if need > free:
        raise SystemExit(f"not enough room: needs {need:.1f} GB, {free:.1f} GB free.")

    for h, u, _ in jobs:
        build(unlab_dir if u else src_dir, unlab_names if u else names, h, a.workers,
              unlabelled=u)


if __name__ == "__main__":
    sys.exit(main())

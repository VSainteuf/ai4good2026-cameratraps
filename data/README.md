# data/

Four small files ship with the repository. The image folders are built locally by
`prepare.py` and are not in git.

## What ships

| file | size | what it is |
| --- | ---: | --- |
| `metadata.csv.gz` | 4.9 MB | one row per image, 201,399 rows. Everything is built from this. Read by `load_task`. |
| `taxonomy.csv` | 54 KB | 61 species with their genus, family, order and class, plus the full taxonomy string. Read by `load_taxonomy`. |
| `official_test_cameras.json` | 762 B | the 48 camera ids held out by the official protocol, under `cameras`. The other keys record how they were recovered from the WILDS release. Read by `official_test_cameras`. |
| `gps_locations.json` | 21 KB | `{location_id: {latitude, longitude}}` for 246 cameras, 179 of which are in the labelled task. No code reads it; the notebook does. |

## What `prepare.py` builds

| folder | what it is |
| --- | --- |
| `images_h224/`, `images_h448/` | the labelled half, resized. Plain JPEGs named by `file_name`. |
| `images_unlabelled_h224/`, `images_unlabelled_h448/` | the 2020 competition's own test half, 62,894 frames from 91 cameras that appear in no fold. No labels were ever released for it. |

Anything else in this folder is a local build artefact and is gitignored.

## `metadata.csv.gz` columns

| column | what it is |
| --- | --- |
| `image_id` | the archive's uuid for this image. `file_name` without the extension. |
| `file_name` | the JPEG's name. `Image.open(image_dir / file_name)` is the whole storage contract. |
| `location` | camera id, 323 distinct. This is the domain. |
| `datetime` | when the frame was taken, `YYYY-MM-DD HH:MM:SS`. The set spans 2013-01-01 to 2015-07-15. |
| `seq_id` | the raw camera sequence, capped at 10 frames, 36,292 of them. Kept for reference. **Do not split on it**: 4,034 bursts span more than one sequence, so splitting here puts near-duplicate frames on both sides. |
| `frame_num` | position within `seq_id`, 0 to 9. |
| `width`, `height` | the original pixel size, before any resizing. |
| `category_id` | the archive's numeric label. |
| `category` | species name as a lowercase latin binomial, or `empty`. 204 distinct, `empty` included. |
| `is_empty` | true when `category` is `empty`. 69,469 rows. |
| `burst_id` | the group a split must not break: images sharing a `seq_id`, plus consecutive frames at the same camera within 60 seconds, closed transitively. 24,658 groups, median 3 frames. This is what the folds split on. |
| `count` | animals in the frame, from the 2022 annotations. `-1` where none was recorded, which is 93% of rows. |
| `hour`, `date` | the hour and the date of `datetime`, split out for convenience. |
| `camera_make`, `camera_model` | from EXIF, missing for 1,741 rows. Panthera models are per-device serial numbers, so for those cameras the model is effectively a second camera id. |
| `image_description` | the EXIF `ImageDescription`, missing for 137,080 rows. Reconyx cameras write per-frame sensor settings here. |
| `official_ood` | this image's fold in the protocol split: `train`, `val` or `test`. |
| `random_burst` | this image's fold in the in-distribution reference split. |

## Three things to know

**The table is unfiltered.** It is every image in the 2022 release, 323 cameras and 204
categories. `load_task` cuts it down to the 60 species with at least 10 images at at least
5 cameras — 127,681 images from 294 cameras — and hands the 69,469 `empty` frames back
separately as the unlabelled pool. The 4,249 images in between belong to species too rare
to learn.

**The fold columns cover every row**, `empty` frames included, so the numbers you get from
`official_ood` here are larger than the ones `load_fold` returns. `load_fold` intersects
them with the labelled task.

**The folds are fixed and ship as columns**, so the partition is identical for everyone
however much of the archive they have prepared, and it does not depend on a seed.

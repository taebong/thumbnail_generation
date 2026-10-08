# thumbnail_generation

Small previews of large spinning-disk datasets so they can be reviewed without opening the raw data.

For every dataset under a folder (searched recursively):

| data | output |
|---|---|
| snapshot (1 time point), single or multi-channel, with or without Z | `<name>.png` |
| time-lapse | `<name>.mp4` (H.264, usually a few MB, capped at ~20 MB) + `<name>.png` poster (middle frame) |

It also writes a `<name>.json` metadata sidecar per output and an `index.html` gallery for the whole folder (open it through the OOD file browser or copy the folder to your laptop).

Layout: one grayscale panel per channel, plus a colour **merge** panel when there are ≥2 fluorescence channels (merge colours come from Micro-Manager's `DisplaySettings.json`). Panels are labelled with the channel name, movie frames with the elapsed time (from per-plane `ElapsedTime-ms`), and the last panel has a scale bar.

Readable formats: Micro-Manager MMStack (`*_MMStack_*.ome.tif`, multi-file, multi-position), NDTiff (`*_NDTiffStack*.tif`), and plain/ImageJ TIFFs.

## Install

```bash
git clone https://github.com/taebong/thumbnail_generation.git
cd thumbnail_generation
conda env create -f environment.yml        # creates env "thumbnails" (or use any env with these packages)
```

Tested with tifffile 2024.12–2026.3 and zarr 2.18 and 3.1.

## Run

Run it on a compute node. The script reads only the planes it needs, but a large time-lapse can still mean tens of GB of I/O (about 5 min per 300 GB dataset on Cannon).

```bash
# SLURM (FAS RC Cannon): one array task per folder, 10 at a time
conda activate thumbnails                  # or: export THUMB_PYTHON=/path/to/env/bin/python
bash submit_thumbnails.sh /path/to/data/20250127_experiment
bash submit_thumbnails.sh /path/to/data/2026*                            # many folders
bash submit_thumbnails.sh FOLDER -- --max-frames 150 --overwrite         # extra options after --

# or directly, e.g. inside an salloc session
unset PYTHONPATH
python make_thumbnails.py FOLDER [--out DIR]
```

`submit_thumbnails.sh` asks for 8 cores, 32 GB and 8 h on `shared,sapphire`. Edit its `#SBATCH` lines for other partitions or clusters. Logs go to `logs/` next to the script.

Default output: `<data folder>/_thumbnails/` (override with `--out DIR`). Datasets that already have outputs are skipped, so you can re-run safely after new data arrive. Use `--overwrite` to redo them.

## How data are reduced

- **Time**: at most `--max-frames` (300) evenly spaced time points. The fps is chosen so the movie lasts about `--movie-seconds` (20 s), between 3 and 30 fps.
- **Z**: max projection. For movies, only `--movie-max-z` (8) evenly spaced planes are read per frame to limit I/O; `0` uses all planes. Snapshots use all planes. Brightfield/DIC channels use the middle plane. `--z-mode mid` uses the middle plane for every channel.
- **XY**: mean-binned to at most `--movie-px` (512) / `--snap-px` (1024) pixels per panel.
- **Contrast**: per channel, fixed across the whole movie (0.5–99.9 percentile), so bleaching stays visible.
- **Aborted acquisitions**: only acquired time points are used. Micro-Manager stores the planned size, and tifffile fills the missing planes with zeros.
- **Multi-position**: one output per position (`<name>_<PositionName>`).

`python make_thumbnails.py -h` lists all options; `--dry-run` lists the datasets it would process.

#!/usr/bin/env python
"""Small preview files for large spinning-disk datasets (Micro-Manager MMStack / NDTiff, plain TIFF).

For every dataset found (recursively) under each input folder:
  * snapshot (single time point)  -> <name>.png
  * time-lapse (>1 time point)    -> <name>.mp4 (H.264) + <name>.png poster (middle frame)
Each multi-position dataset gives one output per position. Z stacks are max-projected
(brightfield-like channels use the middle plane instead). Channels are shown as separate
grayscale panels, plus a colour merge when there are >=2 fluorescence channels.
A <name>.json sidecar stores the metadata, and index.html in the output folder links everything.

Data are read plane-by-plane and only for the sampled time points / Z planes, so a 300 GB
time-lapse is previewed without loading it. Still, run it on a compute node (see submit_thumbnails.sh).

Usage:
    python make_thumbnails.py /path/to/data/20250127_experiment
    python make_thumbnails.py FOLDER --out DIR --max-frames 200 --movie-max-z 0
"""
import argparse
import html
import json
import math
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore', module='tifffile')
import logging
logging.getLogger('tifffile').setLevel(logging.ERROR)  # "MMStack is missing N pages" etc.

BF_PATTERN = re.compile(r'bright|^bf$|^bf[_ ]|dic|trans|phase', re.I)
MM_TAG = 51123  # MicroManagerMetadata per-plane JSON
OUT_SUBDIR = '_thumbnails'  # default output: <data folder>/_thumbnails


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


# ----------------------------------------------------------------------------------------------
# dataset discovery
# ----------------------------------------------------------------------------------------------
def find_datasets(root):
    """Group TIFF files into datasets. Returns [(name, representative_file)].

    MMStack / NDTiff datasets span many files (_1, _2 ... and one file per position); tifffile
    reassembles the whole dataset from one file, so only one representative file is opened.
    """
    groups = {}
    for p in sorted(root.rglob('*')):
        if p.name.startswith('.') or p.suffix.lower() not in ('.tif', '.tiff') or not p.is_file() \
                or OUT_SUBDIR in p.relative_to(root).parts:
            continue
        if '_MMStack' in p.name:
            prefix = p.name.split('_MMStack')[0]
        elif 'NDTiffStack' in p.name:
            prefix = p.name.split('_NDTiffStack')[0].split('NDTiffStack')[0] or p.parent.name
        else:
            prefix = re.sub(r'(\.ome)?\.tiff?$', '', p.name, flags=re.I)
        groups.setdefault((p.parent, prefix), []).append(p)

    datasets = []
    for (d, prefix), files in groups.items():
        rep = min(files, key=lambda f: (len(f.name), f.name))  # X_MMStack_Pos0.ome.tif before X_MMStack_Pos0_1.ome.tif
        # flat output name: relative folders (minus ones equal to the prefix) + prefix
        parts = [x for x in d.relative_to(root).parts if x != prefix]
        datasets.append(('__'.join(parts + [prefix]), rep))
    return sorted(datasets)


# ----------------------------------------------------------------------------------------------
# metadata helpers
# ----------------------------------------------------------------------------------------------
def page_metadata(page):
    if page is None:
        return {}
    try:
        if not hasattr(page, 'tags') or MM_TAG not in page.tags:
            page = page.aspage()  # TiffFrame -> TiffPage (reads its tags)
        md = page.tags[MM_TAG].value
        return md if isinstance(md, dict) else json.loads(md)
    except Exception:
        return {}


def display_colors(folder):
    """Channel name -> RGB (0..1) from Micro-Manager DisplaySettings.json, if present."""
    f = folder / 'DisplaySettings.json'
    out = {}
    try:
        for ch in json.loads(f.read_text())['map']['ChannelSettings']['array']:
            rgb = ch['Color']['scalar']['Components'][:3]
            out[ch['Channel']['scalar']] = tuple(float(v) for v in rgb)
    except Exception:
        pass
    return out


def fallback_color(name, i):
    m = re.search(r'(\d{3})', name)
    if m:
        wl = int(m.group(1))
        if wl < 460:
            return (0.0, 0.6, 1.0)   # 405/445: blue
        if wl < 530:
            return (0.0, 1.0, 0.0)   # 488/514: green
        if wl < 620:
            return (1.0, 0.0, 1.0)   # 561/594: magenta
        return (1.0, 1.0, 0.0)       # 640+: yellow (distinct from magenta)
    return [(0, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0), (0, 0, 1)][i % 6]


def fmt_time(sec, total):
    if total < 120:
        return f'{sec:.2f} s' if total < 10 else f'{sec:.1f} s'
    if total < 7200:
        return f'{int(sec // 60):02d}:{int(sec % 60):02d} (mm:ss)'
    return f'{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d} (hh:mm)'


def evenly(seq, n):
    """At most n evenly spaced elements of seq (n <= 0 -> all)."""
    seq = list(seq)
    if n <= 0 or len(seq) <= n:
        return seq
    idx = np.unique(np.round(np.linspace(0, len(seq) - 1, n)).astype(int))
    return [seq[i] for i in idx]


# ----------------------------------------------------------------------------------------------
# stack access
# ----------------------------------------------------------------------------------------------
class Stack:
    """Uniform T,R,C,Z,Y,X(,S) view of a tifffile series with a per-plane 'acquired' mask."""

    def __init__(self, path):
        self.path = path
        self.tf = tifffile.TiffFile(path)
        s = self.tf.series[0]
        try:
            axes, shape = s.get_axes(False), s.get_shape(False)
        except Exception:
            axes, shape = s.axes, s.shape
        self.rgb = 'S' in axes and shape[axes.index('S')] in (3, 4)
        self.z = zarr.open(s.aszarr(), mode='r')
        if not isinstance(self.z, zarr.Array):  # pyramidal -> full resolution level
            self.z = self.z['0']
        if tuple(self.z.shape) != tuple(shape):  # squeezed store
            axes, shape = s.axes, s.shape
        self.axes, self.shape = axes, tuple(shape)

        # map every non-image axis to a role in T,R,C,Z
        frame_axes = [a for a in axes if a not in 'YXS']
        roles = {}
        for i, a in enumerate(axes):
            if a in 'YXS':
                continue
            if a in 'TRCZ' and a not in roles:
                roles[a] = i
        for i, a in enumerate(axes):  # generic axes (I, Q, ...): long ones are time, short ones Z
            if a in 'YXS' or i in roles.values() or shape[i] == 1:
                continue
            if 'T' not in roles and (shape[i] > 100 or 'Z' in roles):
                roles['T'] = i
            elif 'Z' not in roles:
                roles['Z'] = i
        self.roles = roles
        self.other = [i for i, a in enumerate(axes) if a not in 'YXS' and i not in roles.values()]
        self.ny, self.nx = shape[axes.index('Y')], shape[axes.index('X')]
        self.n = {k: (shape[roles[k]] if k in roles else 1) for k in 'TRCZ'}

        # acquired-plane mask (MM marks aborted / skipped planes as None)
        frame_shape = tuple(shape[axes.index(a)] for a in frame_axes)
        pages = s.pages
        if len(pages) == int(np.prod(frame_shape)):
            self.pages = pages
            valid = np.array([p is not None for p in pages], bool).reshape(frame_shape)
        else:  # contiguous ImageJ etc.: cannot map planes to pages, assume all present
            self.pages = None
            valid = np.ones(frame_shape, bool)
        # reorder mask to T,R,C,Z
        fa = [axes.index(a) for a in frame_axes]
        sel = []
        for i in fa:
            sel.append(slice(None) if i in roles.values() else 0)
        valid = valid[tuple(sel)]
        kept = [i for i in fa if i in roles.values()]
        order = [kept.index(roles[k]) for k in 'TRCZ' if k in roles]
        valid = np.transpose(valid, order)
        for k_i, k in enumerate('TRCZ'):
            if k not in roles:
                valid = np.expand_dims(valid, k_i)
        self.valid = valid  # (T,R,C,Z)

        self.summary = {}
        mm = getattr(self.tf, 'micromanager_metadata', None)
        if isinstance(mm, dict):
            self.summary = mm.get('Summary', {}) or {}

    def index(self, t, r, c, zz):
        idx = [0] * len(self.axes)
        for k, v in zip('TRCZ', (t, r, c, zz)):
            if k in self.roles:
                idx[self.roles[k]] = v
        for i in self.other:
            idx[i] = 0
        return idx

    def page(self, t, r, c, zz):
        if self.pages is None:
            return None
        frame_axes = [i for i, a in enumerate(self.axes) if a not in 'YXS']
        idx = self.index(t, r, c, zz)
        flat = np.ravel_multi_index([idx[i] for i in frame_axes], [self.shape[i] for i in frame_axes])
        return self.pages[flat]

    def plane(self, t, r, c, zz):
        idx = self.index(t, r, c, zz)
        sl = tuple(slice(None) if a in 'YXS' else idx[i] for i, a in enumerate(self.axes))
        return np.asarray(self.z[sl])

    def close(self):
        self.tf.close()


def bin_mean(img, f):
    if f <= 1:
        return img.astype(np.float32)
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    img = img[:h, :w].astype(np.float32)
    return img.reshape(h // f, f, w // f, f, *img.shape[2:]).mean(axis=(1, 3))


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text(draw, xy, s, font):
    draw.text(xy, s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))


def nice_scalebar(width_um):
    for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        if v >= width_um / 6:
            return v
    return 1000


class Renderer:
    def __init__(self, names, colors, is_bf, lims, um_per_px, colored_panels):
        self.names, self.colors, self.is_bf, self.lims = names, colors, is_bf, lims
        self.um_per_px, self.colored = um_per_px, colored_panels
        self.fluor = [i for i, b in enumerate(is_bf) if not b]
        self.merge = len(self.fluor) >= 2

    def norm(self, img, c):
        lo, hi = self.lims[c]
        return np.clip((img - lo) / (hi - lo), 0, 1)

    def frame(self, chans, label=None):
        """chans: list of 2-D float arrays (one per channel) -> RGB uint8 image."""
        panels, titles = [], []
        normed = [self.norm(x, c) for c, x in enumerate(chans)]
        for c, n in enumerate(normed):
            col = np.array(self.colors[c] if (self.colored and not self.is_bf[c]) else (1, 1, 1), np.float32)
            panels.append(n[..., None] * col)
            titles.append(self.names[c])
        if self.merge:
            m = sum(normed[c][..., None] * np.array(self.colors[c], np.float32) for c in self.fluor)
            panels.append(np.clip(m, 0, 1))
            titles.append('merge')
        h, w = panels[0].shape[:2]
        ncol = len(panels) if len(panels) <= 3 else math.ceil(len(panels) / 2)
        nrow = math.ceil(len(panels) / ncol)
        gap = 4
        H, W = nrow * h + (nrow - 1) * gap, ncol * w + (ncol - 1) * gap
        canvas = np.full((H + H % 2, W + W % 2, 3), 40, np.uint8)  # even size for yuv420p
        for k, p in enumerate(panels):
            y, x = (k // ncol) * (h + gap), (k % ncol) * (w + gap)
            canvas[y:y + h, x:x + w] = (p * 255 + 0.5).astype(np.uint8)
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        fs = max(12, w // 28)
        font = get_font(fs)
        for k, t in enumerate(titles):
            y, x = (k // ncol) * (h + gap), (k % ncol) * (w + gap)
            text(d, (x + 6, y + 4), t, font)
        if label:
            text(d, (6, h - fs - 10), label, font)
        if self.um_per_px:  # scale bar on the last panel
            k = len(panels) - 1
            y0, x0 = (k // ncol) * (h + gap), (k % ncol) * (w + gap)
            L = nice_scalebar(w * self.um_per_px)
            Lpx = max(1, round(L / self.um_per_px))
            bh = max(3, h // 100)
            xb, yb = x0 + w - 10 - Lpx, y0 + h - 10 - bh
            d.rectangle([xb, yb, xb + Lpx, yb + bh], fill=(255, 255, 255))
            s = f'{L} um'
            tw = d.textlength(s, font=font)
            text(d, (xb + Lpx / 2 - tw / 2, yb - fs - 6), s, font)
        return np.asarray(im)


def write_mp4(path, frames, fps, target_mb, crf):
    H, W = frames[0].shape[:2]
    duration = len(frames) / fps
    maxrate = int(target_mb * 8e6 * 0.9 / max(duration, 1))  # cap so the file stays below ~target_mb
    tmp = path.with_suffix('.tmp.mp4')
    w = imageio_ffmpeg.write_frames(
        str(tmp), (W, H), fps=fps, codec='libx264', quality=None, macro_block_size=2,
        pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        output_params=['-crf', str(crf), '-preset', 'medium', '-maxrate', str(maxrate),
                       '-bufsize', str(2 * maxrate), '-movflags', '+faststart'])
    w.send(None)
    for f in frames:
        w.send(np.ascontiguousarray(f))
    w.close()
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------
# per-dataset processing
# ----------------------------------------------------------------------------------------------
def process(name, path, out_dir, a):
    t0 = time.time()
    st = Stack(path)
    try:
        nT, nR, nC, nZ = (st.n[k] for k in 'TRCZ')
        log(f'{name}: {path.name} axes={st.axes} shape={st.shape}')
        colors_ds = display_colors(path.parent)
        results = []
        for r in range(nR):
            v = st.valid[:, r]
            t_valid = np.flatnonzero(v.any(axis=(1, 2)))
            if len(t_valid) == 0:
                log(f'  pos {r}: no acquired planes, skipped')
                continue
            movie = len(t_valid) > 1
            t_sel = evenly(t_valid, a.max_frames) if movie else [t_valid[0]]
            px = a.movie_px if movie else a.snap_px
            f = max(1, math.ceil(max(st.ny, st.nx) / px))

            # channel names / pixel size / position name from the first acquired plane of each channel
            mds = []
            for c in range(nC):
                zc = np.flatnonzero(v[t_valid[0], c])
                mds.append(page_metadata(st.page(t_valid[0], r, c, zc[0])) if len(zc) else {})
            ch_names = st.summary.get('ChNames') if isinstance(st.summary.get('ChNames'), list) else None
            names = []
            for c in range(nC):
                n = mds[c].get('Channel') or (ch_names[c] if ch_names and c < len(ch_names) else None)
                names.append(str(n) if n else (f'C{c}' if nC > 1 else ''))
            if st.rgb:
                names = ['RGB']
            is_bf = [bool(BF_PATTERN.search(n)) for n in names]
            colors = [colors_ds.get(n) or fallback_color(n, c) for c, n in enumerate(names)]
            um = mds[0].get('PixelSizeUm') or st.summary.get('PixelSize_um')
            um = float(um) * f if um else None
            pos = mds[0].get('PositionName')
            pos_tag = '' if nR == 1 else '_' + (pos if pos and pos != 'Default' else f'Pos{r}')
            stem = re.sub(r'[^\w.+-]', '_', name + pos_tag)

            # Z planes per channel (same for all t; taken from first acquired time point)
            z_sel = []
            for c in range(nC):
                zc = np.flatnonzero(v[t_valid[0], c])
                if len(zc) == 0:
                    zc = np.flatnonzero(v[:, c].any(axis=0))
                if len(zc) == 0:
                    z_sel.append([])
                elif is_bf[c] or a.z_mode == 'mid':
                    z_sel.append([zc[len(zc) // 2]])
                else:
                    z_sel.append(evenly(zc, a.movie_max_z if movie else 0))

            def read_t(t):
                out = []
                for c in range(nC):
                    acc = None
                    for zz in z_sel[c]:
                        if not st.valid[t, r, c, zz]:
                            continue
                        img = st.plane(t, r, c, zz)
                        if not img.any():  # pre-allocated but never written (aborted acquisition)
                            continue
                        img = bin_mean(img, f)
                        acc = img if acc is None else np.maximum(acc, img)
                    out.append(acc)
                return out

            log(f'  pos {r}{pos_tag}: {"movie" if movie else "snapshot"}, {len(t_sel)}/{len(t_valid)} time points, '
                f'channels={names}, z planes/ch={[len(z) for z in z_sel]}, bin={f}')
            with ThreadPoolExecutor(a.threads) as ex:
                data = list(ex.map(read_t, t_sel))
            keep = [i for i, fr in enumerate(data) if any(x is not None for x in fr)]
            n_empty = len(data) - len(keep)
            if n_empty:
                log(f'  dropped {n_empty} empty (all-zero) time points')
            if not keep:
                log(f'  pos {r}: all sampled planes are empty, skipped')
                continue
            t_sel, data = [t_sel[i] for i in keep], [data[i] for i in keep]
            movie = len(t_sel) > 1
            # fill channels missing at some time points with the previous frame (or zeros)
            shape2 = next(x for fr in data for x in fr if x is not None).shape
            for i, fr in enumerate(data):
                for c in range(nC):
                    if fr[c] is None:
                        fr[c] = data[i - 1][c] if i else np.zeros(shape2, np.float32)
            if st.rgb:  # already-rendered colour image: just show it
                data = [[np.asarray(fr[0])[..., :3]] for fr in data]
                lims = [(0.0, 255.0 if np.max(data[0][0]) <= 255 else float(np.max(data[0][0])))]
            else:
                lims = []
                for c in range(nC):
                    pix = np.concatenate([fr[c][::4, ::4].ravel() for fr in evenly(data, 50)])
                    lo, hi = np.percentile(pix, (a.low_pct, a.high_pct if not is_bf[c] else 99.5))
                    lims.append((float(lo), float(hi) if hi > lo else float(lo) + 1))
            rend = Renderer(names, colors, is_bf, lims, um, a.colored)
            if st.rgb:
                rend.frame = lambda chans, label=None: (np.clip(chans[0], 0, lims[0][1]) / lims[0][1] * 255).astype(np.uint8)

            # time stamps
            secs = []
            interval = st.summary.get('Interval_ms') or 0
            for t in t_sel:
                el = None
                if movie:
                    c0 = next((c for c in range(nC) if len(z_sel[c])), 0)
                    el = page_metadata(st.page(t, r, c0, z_sel[c0][0] if z_sel[c0] else 0)).get('ElapsedTime-ms')
                secs.append(float(el) / 1000 if el is not None else (t * interval / 1000 if interval else None))
            if movie and all(s is not None for s in secs):
                total = secs[-1] - secs[0]
                labels = [f't = {fmt_time(s - secs[0], total)}' for s in secs]
            else:
                total = None
                labels = [f'frame {t}' for t in t_sel] if movie else [None]
            zlabel = 'max-Z' if any(len(z) > 1 for z in z_sel) else ''

            info = dict(name=stem, source=str(path), axes=st.axes, shape=list(st.shape), position=r,
                        position_name=pos, channels=names, n_timepoints_acquired=int(len(t_valid)),
                        n_empty_timepoints_dropped=n_empty,
                        n_timepoints_planned=int(nT), n_frames_in_movie=len(t_sel) if movie else 0,
                        z_planes=int(nZ), z_planes_used=[len(z) for z in z_sel], bin=f,
                        pixel_um_after_bin=um, duration_s=total,
                        interval_s=(interval / 1000 if interval else None))
            frames = [rend.frame(fr, (lab + ('  ' + zlabel if zlabel else '')) if lab else zlabel or None)
                      for fr, lab in zip(data, labels)]
            Image.fromarray(frames[len(frames) // 2]).save(out_dir / f'{stem}.png', optimize=True)
            if movie:
                fps = a.fps or float(np.clip(len(frames) / a.movie_seconds, 3, 30))
                write_mp4(out_dir / f'{stem}.mp4', frames, fps, a.target_mb, a.crf)
                info.update(fps=fps, mp4=f'{stem}.mp4',
                            mp4_mb=round((out_dir / f'{stem}.mp4').stat().st_size / 1e6, 2))
            info['png'] = f'{stem}.png'
            (out_dir / f'{stem}.json').write_text(json.dumps(info, indent=1))
            results.append(info)
            log(f'  -> {stem}.{"mp4" if movie else "png"} '
                f'{info.get("mp4_mb", "")}{" MB" if movie else ""} ({time.time() - t0:.0f} s)')
        return results
    finally:
        st.close()


def write_index(out_dir):
    rows = []
    for j in sorted(out_dir.glob('*.json')):
        try:
            d = json.loads(j.read_text())
        except Exception:
            continue
        if 'mp4' in d:
            media = (f'<video src="{html.escape(d["mp4"])}" poster="{html.escape(d["png"])}" controls loop '
                     f'muted preload="none"></video>')
        else:
            media = f'<a href="{html.escape(d["png"])}"><img src="{html.escape(d["png"])}" loading="lazy"></a>'
        meta = [f'channels: {", ".join(d["channels"])}',
                f'time points: {d["n_timepoints_acquired"]}'
                + (f' of {d["n_timepoints_planned"]} planned' if d["n_timepoints_planned"] != d["n_timepoints_acquired"] else ''),
                f'Z planes: {d["z_planes"]}', f'shape {d["axes"]} {d["shape"]}']
        if d.get('interval_s'):
            meta.append(f'interval: {d["interval_s"]:g} s')
        if d.get('duration_s'):
            meta.append(f'duration: {d["duration_s"] / 60:.1f} min')
        rows.append(f'<div class="card"><h3>{html.escape(d["name"])}</h3>{media}'
                    f'<p>{"<br>".join(html.escape(m) for m in meta)}</p></div>')
    (out_dir / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>' + html.escape(out_dir.name) + '</title><style>'
        'body{font-family:sans-serif;background:#111;color:#ddd;margin:16px}'
        '.card{display:inline-block;vertical-align:top;margin:8px;max-width:720px}'
        'img,video{max-width:720px;width:100%}h3{margin:4px 0;font-size:14px}p{font-size:12px;color:#999}'
        '</style><h2>' + html.escape(out_dir.name) + '</h2>' + '\n'.join(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+', type=Path)
    p.add_argument('--out', type=Path, help=f'output folder (default: <folder>/{OUT_SUBDIR}; with several input folders, '
                                             'one sub-folder each)')
    p.add_argument('--max-frames', type=int, default=300, help='max time points in a movie (evenly subsampled)')
    p.add_argument('--movie-px', type=int, default=512, help='max panel size (px) in movies')
    p.add_argument('--snap-px', type=int, default=1024, help='max panel size (px) in snapshots')
    p.add_argument('--movie-max-z', type=int, default=8,
                   help='max Z planes (evenly spaced) max-projected per movie frame; 0 = all (slow: reads everything)')
    p.add_argument('--z-mode', choices=['max', 'mid'], default='max', help='Z max projection or middle plane')
    p.add_argument('--movie-seconds', type=float, default=20, help='target movie length (fps clipped to 3-30)')
    p.add_argument('--fps', type=float, help='fixed frame rate (overrides --movie-seconds)')
    p.add_argument('--target-mb', type=float, default=20, help='bitrate cap so movies stay below ~this size')
    p.add_argument('--crf', type=int, default=23, help='H.264 quality (lower = better/larger)')
    p.add_argument('--low-pct', type=float, default=0.5, help='contrast: lower percentile')
    p.add_argument('--high-pct', type=float, default=99.9, help='contrast: upper percentile (fluorescence)')
    p.add_argument('--colored', action='store_true', help='colour single-channel panels with their LUT (default gray)')
    p.add_argument('--threads', type=int, default=4, help='parallel reader threads')
    p.add_argument('--overwrite', action='store_true', help='redo datasets that already have outputs')
    p.add_argument('--dry-run', action='store_true', help='only list datasets that would be processed')
    a = p.parse_args()

    n_err = 0
    for folder in a.folders:
        folder = folder.expanduser().absolute()
        if not folder.is_dir():
            log(f'not a folder: {folder}')
            n_err += 1
            continue
        if a.out:
            out_dir = a.out.expanduser().absolute() / (folder.name if len(a.folders) > 1 else '')
        else:
            out_dir = folder / OUT_SUBDIR
        datasets = find_datasets(folder)
        log(f'{folder}: {len(datasets)} datasets -> {out_dir}')
        if a.dry_run:
            for name, f in datasets:
                log(f'  {name}: {f}')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        done = set()
        for j in out_dir.glob('*.json'):
            try:
                done.add(json.loads(j.read_text())['source'])
            except Exception:
                pass
        for name, f in datasets:
            if not a.overwrite and str(f) in done:
                log(f'{name}: exists, skipped (use --overwrite)')
                continue
            try:
                process(name, f, out_dir, a)
            except Exception:
                n_err += 1
                log(f'{name}: FAILED\n{traceback.format_exc()}')
            write_index(out_dir)
        write_index(out_dir)
    sys.exit(1 if n_err else 0)


if __name__ == '__main__':
    main()

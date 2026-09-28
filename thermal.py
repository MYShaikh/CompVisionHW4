#!/usr/bin/env python3
"""
thermal_human_boundary.py
=========================
Find the exact boundaries of people in a THERMAL (infrared) image using only
classical computer vision (OpenCV + NumPy). The segmentation uses no deep
learning and no trained or learned model: no HOG-SVM, no Haar cascades, no
k-means. The result can optionally be compared against SAM2.

Supported inputs
----------------
* 8-bit grayscale thermal images, white-hot (the default) or black-hot
  (--polarity black), e.g. FLIR ADAS, LLVIP, KAIST.
* 16-bit radiometric PNG/TIFF frames. These are read at full bit depth.
* Pseudo-colour thermal images (ironbow, rainbow, "Spectra", ...), e.g.
  the Roboflow "Thermal Dogs and People" dataset. The palette is inverted
  back to a scalar heat value:
    - exactly, when it matches one of OpenCV's colormaps (auto-detected, or
      given with --palette jet/inferno/hot/...);
    - otherwise approximately, via perceived lightness (--palette luma). This
      works for palettes that brighten as temperature rises, which covers
      most thermal-camera palettes.

Pipeline
--------
1. Heat map: decode the image to one heat channel, normalise it robustly
   (1st-99th percentile), and denoise it with an edge-preserving filter.
2. Candidate people, from two complementary cues:
   a. Local thermal anomaly: a morphological top-hat (heat minus its grey
      opening with a disc wider than a person) isolates warm person-sized
      structures while ignoring large warm areas and step edges such as
      kerbs and walls. Hysteresis thresholding at k*sigma, with a robust
      noise estimate, keeps the warm parts: head, hands and legs.
      Growth from strong seeds is distance-limited (geodesic dilation), so it
      cannot chain across textured ground. A tall, thin vertical closing
      then bridges cooler clothing between warm parts of one body.
   b. Global contrast: a ladder of thresholds (Otsu, a second Otsu inside
      the warm part, and high percentiles). This handles close-ups, where a
      person is larger than the top-hat disc.
3. Rule-based human-shape filter. Each candidate blob is scored by its
   thermal contrast against a surrounding ring, and by shape: upright
   aspect ratio, solidity, and not box-like. Box-like shapes reject
   radiators, lamps, windows and car bonnets. Large blobs must also show the
   internal temperature mix of warm skin and cooler clothing, which uniform
   warm ground lacks. Overlapping candidates are de-duplicated, and a warm
   head separated from the body by a cool collar is merged back into it.
4. Exact boundary per person. GrabCut (graph-cut energy minimisation) runs
   on a crop around each candidate. It uses a 3-channel feature image
   (heat, local contrast, local anomaly) and is seeded with a definite
   warm core, probable foreground and background, and a definite
   background ring. Its pairwise term snaps the contour to the thermal
   edges.
5. Post-processing (fill small holes, light smoothing), then exact contour
   extraction with cv2.findContours(CHAIN_APPROX_NONE), which returns every
   boundary pixel of every person.

SAM2 comparison
---------------
SAM2 is prompted with one box per detected person, using the same boxes the
classical method found, so both methods get identical localisation. Masks
are compared both overall (all people together) and per person: IoU, Dice,
precision/recall, the boundary F-score, the Hausdorff distance, HD95 and the
average symmetric surface distance (ASSD).
  * --run-sam2       run SAM2 locally. Install it with:
                       pip install torch torchvision
                       pip install "git+https://github.com/facebookresearch/sam2.git"
  * --sam2-mask PATH use a mask exported from https://sam2.metademolab.com

Usage
-----
  python thermal_human_boundary.py ir.jpg
  python thermal_human_boundary.py ir.jpg --debug
  python thermal_human_boundary.py ir.jpg --polarity black       # black-hot
  python thermal_human_boundary.py seek.jpg --palette luma       # colour palette
  python thermal_human_boundary.py ir.jpg --rect 300,120,90,240  # given box(es)
  python thermal_human_boundary.py ir.jpg --interactive          # draw box(es)
  python thermal_human_boundary.py ir.jpg --run-sam2             # + compare
  python thermal_human_boundary.py ir.jpg --sam2-mask sam2.png   # + compare
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

COLORMAPS = {
    "autumn": cv2.COLORMAP_AUTUMN, "bone": cv2.COLORMAP_BONE, "jet": cv2.COLORMAP_JET,
    "rainbow": cv2.COLORMAP_RAINBOW, "ocean": cv2.COLORMAP_OCEAN,
    "hot": cv2.COLORMAP_HOT, "hsv": cv2.COLORMAP_HSV, "cool": cv2.COLORMAP_COOL,
    "pink": cv2.COLORMAP_PINK, "parula": cv2.COLORMAP_PARULA,
    "magma": cv2.COLORMAP_MAGMA, "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA, "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS, "turbo": cv2.COLORMAP_TURBO,
    "twilight": cv2.COLORMAP_TWILIGHT,
}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def ellipse_kernel(radius):
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def clip_rect(rect, w, h):
    x, y, rw, rh = [int(round(v)) for v in rect]
    x, y = max(0, x), max(0, y)
    return x, y, max(1, min(rw, w - x)), max(1, min(rh, h - y))


def robust_sigma(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-6


def resize_max(img, max_dim):
    h, w = img.shape[:2]
    s = min(1.0, float(max_dim) / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img, s


# --------------------------------------------------------------------------- #
# 1. Heat map
# --------------------------------------------------------------------------- #
def invert_colormap(bgr, cmap):
    """Map every pixel to the index of its nearest palette colour.
    Returns (heat in 0..255 as float32, mean colour error)."""
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cmap)
    lut = lut.reshape(256, 3).astype(np.float32)
    flat = bgr.reshape(-1, 3)
    packed = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    uniq, inv = np.unique(packed, return_inverse=True)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], 1).astype(np.float32)
    idx = np.empty(len(cols), np.int32)
    err = np.empty(len(cols), np.float32)
    for i in range(0, len(cols), 20000):  # chunked nearest-neighbour search
        d = ((cols[i:i + 20000, None, :] - lut[None]) ** 2).sum(-1)
        idx[i:i + 20000] = d.argmin(1)
        err[i:i + 20000] = np.sqrt(d.min(1))
    heat = idx[inv].reshape(bgr.shape[:2]).astype(np.float32)
    return heat, float(err[inv].mean())


def load_heat(path, palette="auto", polarity="white"):
    """Return (heat float32 in [0,1], display BGR uint8, description)."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit(f"Cannot read image: {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]

    if raw.ndim == 2 or raw.dtype != np.uint8:
        g = raw if raw.ndim == 2 else raw.mean(axis=2)
        heat, how = g.astype(np.float32), f"single-channel ({raw.dtype})"
    else:
        chan_diff = np.abs(raw.astype(np.int16) - raw[:, :, :1]).mean()
        if chan_diff < 3 and palette in ("auto", "gray"):
            heat, how = raw[:, :, 1].astype(np.float32), "grayscale"
        elif palette in COLORMAPS:
            heat, e = invert_colormap(raw, COLORMAPS[palette])
            how = f"palette '{palette}' inverted (mean colour error {e:.1f})"
        elif palette == "auto":
            best = None
            small, _ = resize_max(raw, 256)
            for name, cm in COLORMAPS.items():
                _, e = invert_colormap(small, cm)
                if best is None or e < best[1]:
                    best = (name, e)
            if best[1] < 12:
                heat, e = invert_colormap(raw, COLORMAPS[best[0]])
                how = f"palette auto-detected as '{best[0]}' (error {e:.1f})"
            else:
                heat = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
                how = "unknown palette -> perceived lightness (use --palette to override)"
        else:  # "luma" or "gray" on a colour image
            heat = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
            how = "perceived lightness (L*)"

    lo, hi = np.percentile(heat, [1, 99])
    heat = np.clip((heat - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    if polarity == "black":
        heat = 1.0 - heat
        how += ", black-hot inverted"

    if raw.ndim == 3 and raw.dtype == np.uint8:
        display = raw.copy()
    else:
        display = cv2.cvtColor((heat * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return heat, display, how


def denoise(heat):
    h8 = (heat * 255).astype(np.uint8)
    h8 = cv2.bilateralFilter(h8, 7, 25, 5)
    return h8.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# 2. Candidate cues
# --------------------------------------------------------------------------- #
def tophat_maps(heat, se_frac):
    """Morphological top-hat / black-hat with a disc wider than a person.

    Opening removes bright structures narrower than the disc (people) but
    keeps large regions and step edges (kerbs, walls), so
    heat - opening = warm, person-sized structures only. Closing does the
    same for cool structures (e.g. insulated clothing on warm ground).
    Computed on a reduced copy for speed.
    """
    h, w = heat.shape
    small, s = resize_max(heat, 512)
    r = max(2, int(se_frac * min(small.shape)))
    se = ellipse_kernel(r)
    opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    opened = cv2.resize(opened, (w, h), interpolation=cv2.INTER_LINEAR)
    closed = cv2.resize(closed, (w, h), interpolation=cv2.INTER_LINEAR)
    opened = np.minimum(opened, heat)
    closed = np.maximum(closed, heat)
    diameter_full = (2 * r + 1) / s
    return heat - opened, closed - heat, opened, diameter_full


def grow_limited(strong, weak, max_dist):
    """Hysteresis with a distance limit (geodesic dilation): grow the strong
    seeds into the weak mask by at most `max_dist` pixels. Unlike classic
    hysteresis this cannot chain across large textured areas."""
    cur = (strong & weak).astype(np.uint8)
    weak_u8 = weak.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    step = 2
    for _ in range(max(1, int(max_dist) // step)):
        nxt = cv2.dilate(cur, k, iterations=step) & weak_u8
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur > 0


def blob_features(comp, x, y, heat, W, H):
    """Hand-crafted shape/thermal descriptors of one blob (comp = its crop)."""
    h, w = comp.shape
    a = float(comp.sum())
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(cnts)
    hull_a = cv2.contourArea(cv2.convexHull(pts))
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    rot_extent = a / (rw * rh + 1e-6)
    # principal axis angle from image moments (0 deg = vertical)
    mu = cv2.moments(comp, binaryImage=True)
    theta = 0.5 * np.degrees(np.arctan2(2 * mu["mu11"], mu["mu20"] - mu["mu02"] + 1e-9))
    tilt = 90.0 - abs(theta)                   # |angle from vertical|
    # fraction of the outline lying on the image border
    gp = pts.reshape(-1, 2) + [x, y]
    on_border = ((gp[:, 0] <= 1) | (gp[:, 0] >= W - 2) |
                 (gp[:, 1] <= 1) | (gp[:, 1] >= H - 2)).mean()
    vals = heat[y:y + h, x:x + w][comp > 0]
    # width profile: an upright person is narrower at the head than at the body
    row_w = comp.sum(axis=1).astype(np.float32)
    top = row_w[:max(1, int(0.15 * h))]
    head_ratio = float(np.median(top) / (np.percentile(row_w, 90) + 1e-6))
    return dict(head=head_ratio, area=a, aspect=h / float(w), extent=a / float(w * h),
                solidity=a / (hull_a + 1e-6), rot_extent=rot_extent, tilt=tilt,
                border=float(on_border), p95=float(np.percentile(vals, 95)),
                spread=float(np.percentile(vals, 90) - np.percentile(vals, 10)))


def candidates_from_mask(mask, heat, anomaly, min_area, pose, source, verbose=False,
                         max_wh=None, big_size=None):
    """Apply the rule-based human-shape filter to the blobs of a binary mask."""
    H, W = mask.shape
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    hot_level = float(np.percentile(heat, 90))
    upright = pose == "upright"
    lo_ar, hi_ar = (0.9, 6.0) if upright else (0.25, 8.0)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area or a > 0.6 * H * W:
            continue
        comp = (lbl[y:y + h, x:x + w] == i).astype(np.uint8)
        f = blob_features(comp, x, y, heat, W, H)
        # thermal contrast against a surrounding ring
        pad = max(3, int(0.25 * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        full = np.zeros((y1 - y0, x1 - x0), np.uint8)
        full[y - y0:y - y0 + h, x - x0:x - x0 + w] = comp
        ring = cv2.dilate(full, ellipse_kernel(pad * 0.6)) & (1 - full)
        inside_vals = heat[y0:y1, x0:x1][full > 0]
        ring_vals = heat[y0:y1, x0:x1][ring > 0]
        f["contrast"] = float(np.percentile(inside_vals, 80) - np.median(ring_vals)) \
            if ring_vals.size else 0.0

        reasons = []
        if max_wh is not None and (w > max_wh[0] or h > max_wh[1]):
            reasons.append("larger than top-hat scale")
        if not (lo_ar <= f["aspect"] <= hi_ar):
            reasons.append("aspect")
        if f["extent"] < 0.25 or f["solidity"] < 0.45:
            reasons.append("fragmented")
        if f["rot_extent"] > 0.85:
            reasons.append("box/strip")          # lamps, windows, curbs, poles
        if f["border"] > 0.2 and a > 0.01 * H * W:
            reasons.append("border")             # large background regions
        touches_top = y <= 1
        if upright and not touches_top and f["head"] > 0.85 and f["aspect"] > 1.2:
            reasons.append("no head")            # flat-topped: posts, bins, boxes
        if upright and f["tilt"] > 35 and f["aspect"] < 1.6:
            reasons.append("tilted")             # diagonal kerbs and walls
        if f["p95"] < hot_level:
            reasons.append("not warm")           # only warm relative to dark surroundings
        if f["contrast"] < 0.06:
            reasons.append("low contrast")
        # Large blobs must show the mix of warm skin and cooler clothing that
        # people have; sun-warmed ground, walls and bonnets are uniform.
        if big_size is not None and (w > big_size or h > 2 * big_size) and f["spread"] < 0.3:
            reasons.append("uniform temperature")
        if verbose:
            print(f"      cand[{source}] box=({x},{y},{w},{h}) " +
                  " ".join(f"{k}={v:.2f}" for k, v in f.items() if k != "area") +
                  f" -> {'REJECT ' + ','.join(reasons) if reasons else 'keep'}")
        if reasons:
            continue
        shape_score = np.exp(-0.5 * (np.log(f["aspect"] / 2.5) / 0.6) ** 2) if upright else 1.0
        score = f["contrast"] * np.sqrt(a) * (0.4 + 0.6 * shape_score) * \
            (0.5 + 0.5 * f["solidity"])
        m = np.zeros((H, W), np.uint8)
        m[y:y + h, x:x + w] = comp * 255
        out.append(dict(mask=m, box=(int(x), int(y), int(w), int(h)), score=float(score),
                        contrast=f["contrast"], aspect=f["aspect"],
                        solidity=f["solidity"], source=source))
    return out


def drop_strips(mask, pose):
    """Remove long thin warm strips (kerbs, pipes, wires, window frames)
    before body parts are joined, so they cannot merge with a nearby person."""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    drop = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 30:
            continue
        comp = (lbl[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.vstack(cnts))
        elong = max(rw, rh) / (min(rw, rh) + 1e-6)
        mu = cv2.moments(comp, binaryImage=True)
        theta = 0.5 * np.degrees(np.arctan2(2 * mu["mu11"], mu["mu20"] - mu["mu02"] + 1e-9))
        tilt = 90.0 - abs(theta)
        if elong > 8 or (elong > 3.5 and tilt > 25 and pose == "upright"):
            drop[i] = True
    return mask & ~drop[lbl]


def merge_stacked(people):
    """Merge a smaller blob that sits on top of a larger one (typically a warm
    head separated from the body by a cooler collar or scarf), or that lies
    mostly inside the larger one's box, into a single person."""
    people = sorted(people, key=lambda c: -c["mask"].astype(bool).sum())
    merged = [False] * len(people)
    for i, big in enumerate(people):
        if merged[i]:
            continue
        for j in range(i + 1, len(people)):
            if merged[j]:
                continue
            sm = people[j]
            bx, by, bw, bh = big["box"]
            sx, sy, sw, sh = sm["box"]
            ox = max(0, min(bx + bw, sx + sw) - max(bx, sx))
            oy = max(0, min(by + bh, sy + sh) - max(by, sy))
            inside = ox * oy / float(sw * sh)
            on_top = (ox >= 0.5 * sw and sw <= 0.8 * bw and sy < by and
                      sy + sh >= by - 0.25 * bh)
            if inside > 0.5 or on_top:
                m = big["mask"] | sm["mask"]
                if on_top and sy + sh < by + 2:   # bridge the gap (collar)
                    m[sy + sh - 1:by + 2, max(sx, bx):min(sx + sw, bx + bw)] = 255
                x0, y0 = min(bx, sx), min(by, sy)
                x1, y1 = max(bx + bw, sx + sw), max(by + bh, sy + sh)
                big.update(mask=m, box=(x0, y0, x1 - x0, y1 - y0),
                           aspect=(y1 - y0) / float(x1 - x0),
                           score=big["score"] + sm["score"],
                           source=big["source"] + "+" + sm["source"])
                merged[j] = True
    return [p for k, p in enumerate(people) if not merged[k]]


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def find_people(heat, pose="upright", max_people=10, k_hi=4.0, k_lo=2.0,
                bg_frac=0.06, min_area_frac=0.0005, rects=None, verbose=False):
    """Return the list of candidate people (dicts) and the debug maps."""
    H, W = heat.shape
    diag = np.hypot(H, W)
    tophat, blackhat, bg, disc_d = tophat_maps(heat, bg_frac)
    anomaly = tophat - blackhat                  # signed local anomaly
    # Outlier thresholds from the top-hat's own distribution: background
    # texture forms the bulk of it, people are the outliers.
    sub = tophat[::4, ::4]
    med_w, sig = float(np.median(sub)), max(robust_sigma(sub), 0.005)
    subc = blackhat[::4, ::4]
    med_c, sig_c = float(np.median(subc)), max(robust_sigma(subc), 0.005)
    hot_level = float(np.percentile(heat, 85))
    # strong seeds must be warm both locally AND in absolute terms
    warm_hi = (tophat > med_w + k_hi * sig) & (heat >= hot_level)
    warm_lo = tophat > med_w + k_lo * sig
    warm = grow_limited(warm_hi, warm_lo, 0.25 * disc_d)
    # Cooler clothing (e.g. a jacket between a warm head and warm legs) is
    # bridged with a tall, thin vertical closing sized from the person scale.
    warm = drop_strips(warm, pose)
    local = warm.astype(np.uint8) * 255
    kv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (max(3, int(0.08 * disc_d)) | 1,
                                    max(5, int(0.5 * disc_d)) | 1))
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kv)
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, ellipse_kernel(0.0015 * diag))

    # Threshold ladder: Otsu, a second Otsu inside the warm part, and a few
    # high percentiles. The person filter then picks the best person-shaped
    # region at any level. This keeps working when a palette or scene
    # stretches the cool end (e.g. Spectra-type palettes).
    h8 = (heat * 255).astype(np.uint8)
    t1, _ = cv2.threshold(h8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upper = h8[h8 > t1]
    t2 = t1
    if upper.size > 100:
        t2, _ = cv2.threshold(upper.reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    levels = {int(t1), int(t2)} | {int(np.percentile(h8, q)) for q in (75, 85, 92)}
    glob_levels = []
    for t in sorted(levels):
        g = np.where(h8 > t, 255, 0).astype(np.uint8)
        g = cv2.morphologyEx(g, cv2.MORPH_OPEN, ellipse_kernel(0.003 * diag))
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ellipse_kernel(0.006 * diag))
        glob_levels.append(g)
    glob = glob_levels[0]

    debug = dict(background=bg, anomaly=np.clip(0.5 + anomaly / (16 * sig), 0, 1),
                 warm=warm.astype(np.float32), local=local / 255.0, glob=glob / 255.0,
                 sigma=float(sig), anomaly_raw=anomaly)
    min_area = min_area_frac * H * W

    if rects:  # user-given boxes: one person per box
        people = []
        for r in rects:
            x, y, w, h = clip_rect(r, W, H)
            # warm person-sized parts inside the box (same cue as auto mode)
            m_roi = local[y:y + h, x:x + w].copy()
            if (m_roi > 0).mean() < 0.05:        # too little: fall back to Otsu
                roi = h8[y:y + h, x:x + w]
                _, m_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m_roi = cv2.morphologyEx(m_roi, cv2.MORPH_CLOSE, ellipse_kernel(0.02 * max(w, h)))
            n, lbl, st, _ = cv2.connectedComponentsWithStats(m_roi, connectivity=8)
            if n <= 1:
                continue
            areas = st[1:, cv2.CC_STAT_AREA]
            keep = 1 + np.where(areas >= 0.1 * areas.max())[0]
            m = np.zeros((H, W), np.uint8)
            m[y:y + h, x:x + w] = np.isin(lbl, keep).astype(np.uint8) * 255
            bx, by, bw, bh = cv2.boundingRect(m)
            people.append(dict(mask=m, box=(bx, by, bw, bh), score=1.0, source="user box",
                               contrast=0.0, aspect=bh / float(bw), solidity=0.0,
                               roi=(x, y, w, h)))
        return people, debug

    # The top-hat can only isolate structures narrower than its disc, so a
    # local-cue blob much larger than the disc is background leaking through.
    max_wh = (3.5 * disc_d, 5.0 * disc_d)
    cands = candidates_from_mask(local, heat, anomaly, min_area, pose, "local", verbose,
                                 max_wh) + \
        sum((candidates_from_mask(g, heat, anomaly, min_area, pose, f"global{j + 1}", verbose,
                                  big_size=disc_d)
             for j, g in enumerate(glob_levels)), [])
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(box_iou(c["box"], k["box"]) < 0.3 for k in kept):
            kept.append(c)
    kept = merge_stacked(kept)
    kept.sort(key=lambda c: -c["score"])
    if kept:
        best = kept[0]["score"]
        kept = [c for c in kept if c["score"] >= 0.2 * best][:max_people]
    return kept, debug


# --------------------------------------------------------------------------- #
# 3. Exact boundary per person (GrabCut on a thermal feature image)
# --------------------------------------------------------------------------- #
def feature_image(heat, anomaly, sigma):
    h8 = (heat * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(h8)
    an = np.clip(128 + anomaly / (8 * sigma) * 127, 0, 255).astype(np.uint8)
    return cv2.merge([h8, clahe, an])


def refine_person(feat, heat, cand, warm_core, iters=5):
    H, W = heat.shape
    x, y, w, h = cand["box"]
    pad = max(8, int(0.3 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    sub = feat[y0:y1, x0:x1]
    cm = cand["mask"][y0:y1, x0:x1]
    # The boundary may only move within a narrow band around the detected
    # blob. This stops GrabCut from bleeding into similar-looking ground.
    band = max(3, int(0.06 * min(w, h)))
    outer = cv2.dilate(cm, ellipse_kernel(band))
    inner = cv2.erode(cm, ellipse_kernel(band))
    core = warm_core[y0:y1, x0:x1] & (cm > 0)
    core = cv2.erode(core.astype(np.uint8), ellipse_kernel(1)) > 0

    gc = np.full(cm.shape, cv2.GC_BGD, np.uint8)
    gc[outer > 0] = cv2.GC_PR_BGD
    gc[cm > 0] = cv2.GC_PR_FGD
    gc[(inner > 0) | core] = cv2.GC_FGD
    if not (gc == cv2.GC_BGD).any():
        gc[[0, -1], :] = cv2.GC_BGD
        gc[:, [0, -1]] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(sub, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        res = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        res = cm.copy()

    # keep the component(s) overlapping the candidate
    n, lbl, st, _ = cv2.connectedComponentsWithStats(res, connectivity=8)
    if n > 2:
        ov = np.bincount(lbl[cm > 0].ravel(), minlength=n)
        ov[0] = 0
        keep = np.where(ov >= 0.1 * ov.max())[0]
        res = np.isin(lbl, keep).astype(np.uint8) * 255
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = res
    return out


def fill_small_holes(mask, max_area):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask
    out = mask.copy()
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 and cv2.contourArea(c) < max_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def postprocess(mask, sigma=1.0):
    area = int((mask > 0).sum())
    if area == 0:
        return mask
    mask = fill_small_holes(mask, 0.05 * area)
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    return mask


def segment_people(heat, pose="upright", max_people=10, rects=None, refine=True, **kw):
    heat_s = denoise(heat)
    people, dbg = find_people(heat_s, pose=pose, max_people=max_people, rects=rects, **kw)
    feat = feature_image(heat_s, dbg["anomaly_raw"], dbg["sigma"])
    warm_core = dbg["anomaly_raw"] > 4.0 * dbg["sigma"]
    H, W = heat.shape
    labels = np.zeros((H, W), np.uint16)
    for i, p in enumerate(people, start=1):
        m = refine_person(feat, heat_s, p, warm_core) if refine else p["mask"]
        m = postprocess(m, sigma=1.0)
        if rects:  # stay (roughly) inside the user's box
            x, y, w, h = p["roi"]
            px, py = int(0.1 * w), int(0.1 * h)
            box = np.zeros_like(m)
            box[max(0, y - py):y + h + py, max(0, x - px):x + w + px] = 255
            m &= box
        p["final"] = m
        labels[(m > 0) & (labels == 0)] = i
    return people, labels, dbg


def extract_boundaries(mask):
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cnts):
        out.append({"type": "outer" if hier[0][i][3] == -1 else "hole",
                    "area_px": float(cv2.contourArea(c)),
                    "perimeter_px": float(cv2.arcLength(c, True)),
                    "points": c.reshape(-1, 2).tolist()})
    return out


# --------------------------------------------------------------------------- #
# SAM2 reference (deep learning; used only for comparison)
# --------------------------------------------------------------------------- #
def run_sam2(display_bgr, boxes_xyxy, model_id="facebook/sam2.1-hiera-large"):
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        sys.exit("SAM2 is not installed. Run:\n"
                 "  pip install torch torchvision\n"
                 "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                 "or pass a pre-computed mask with --sam2-mask.")
    device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    H, W = display_bgr.shape[:2]
    with torch.inference_mode():
        predictor.set_image(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(box=np.asarray(boxes_xyxy, np.float32),
                                             multimask_output=False)
    masks = np.asarray(masks).reshape(-1, H, W)
    return [(m > 0.5).astype(np.uint8) * 255 for m in masks], np.asarray(scores).ravel().tolist()


def load_mask(path, shape):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        sys.exit(f"Cannot read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 3] if m.shape[2] == 4 else m.max(axis=2)
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _boundary(m):
    m = (m > 0).astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def compare_masks(pred, ref, tol):
    p, r = pred > 0, ref > 0
    inter, union = np.logical_and(p, r).sum(), np.logical_or(p, r).sum()
    ps, rs = p.sum(), r.sum()
    out = {"iou": inter / union if union else 1.0,
           "dice": 2 * inter / (ps + rs) if (ps + rs) else 1.0,
           "pixel_precision": inter / ps if ps else 0.0,
           "pixel_recall": inter / rs if rs else 0.0,
           "area_classical_px": int(ps), "area_sam2_px": int(rs),
           "boundary_tolerance_px": float(tol)}
    bp, br = _boundary(pred), _boundary(ref)
    if bp.any() and br.any():
        dt_r = cv2.distanceTransform(np.where(br, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        dt_p = cv2.distanceTransform(np.where(bp, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        d_pr, d_rp = dt_r[bp], dt_p[br]
        bprec, brec = (d_pr <= tol).mean(), (d_rp <= tol).mean()
        out.update({"boundary_precision": bprec, "boundary_recall": brec,
                    "boundary_f1": 2 * bprec * brec / (bprec + brec + 1e-12),
                    "hausdorff_px": float(max(d_pr.max(), d_rp.max())),
                    "hd95_px": float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95))),
                    "assd_px": float((d_pr.sum() + d_rp.sum()) / (d_pr.size + d_rp.size))})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
PALETTE = [(60, 220, 60), (0, 200, 255), (255, 160, 0), (60, 60, 255), (255, 80, 200),
           (200, 255, 0), (0, 128, 255), (180, 105, 255), (255, 255, 0), (128, 0, 255)]


def draw_people(img, masks, thickness, boxes=None, colors=None):
    out = img.copy()
    for i, m in enumerate(masks):
        col = (colors or PALETTE)[i % len(colors or PALETTE)]
        tint = np.zeros_like(out)
        tint[m > 0] = col
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, col, thickness, cv2.LINE_AA)
        if boxes is not None:
            x, y, w, h = boxes[i]
            cv2.rectangle(out, (x, y), (x + w, y + h), col, max(1, thickness // 2))
            cv2.putText(out, str(i + 1), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 + thickness * 0.15, col, max(1, thickness // 2), cv2.LINE_AA)
    return out


def titled(img, title, width):
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bar = np.full((34, width, 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def comparison_figure(display, ours, sam, metrics, per_person, boxes, width=560):
    t = max(2, int(0.0025 * max(display.shape[:2])))
    GREEN, MAGENTA = (60, 220, 60), (230, 60, 230)
    a = draw_people(display, [ours], t, colors=[GREEN])
    for (x, y, w, h) in boxes:
        cv2.rectangle(a, (x, y), (x + w, y + h), (0, 200, 255), max(1, t // 2))
    b = draw_people(display, [sam], t, colors=[MAGENTA])
    c = display.copy()
    for m, col in ((sam, MAGENTA), (ours, GREEN)):
        cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c, cnts, -1, col, t, cv2.LINE_AA)
    d = np.zeros_like(display)
    p, r = ours > 0, sam > 0
    d[p & r] = (255, 255, 255)
    d[p & ~r] = GREEN
    d[~p & r] = MAGENTA
    top = np.hstack([titled(a, "Classical (OpenCV) + boxes", width),
                     titled(b, "SAM2 (reference)", width)])
    bot = np.hstack([titled(c, "green=classical  magenta=SAM2", width),
                     titled(d, "white=both  green/magenta=one only", width)])
    lines = [f"ALL PEOPLE  IoU {metrics['iou']:.3f}  Dice {metrics['dice']:.3f}  "
             f"Prec {metrics['pixel_precision']:.3f}  Rec {metrics['pixel_recall']:.3f}"]
    if "boundary_f1" in metrics:
        lines.append(f"Boundary F1@{metrics['boundary_tolerance_px']:.1f}px "
                     f"{metrics['boundary_f1']:.3f}  HD95 {metrics['hd95_px']:.1f}px  "
                     f"ASSD {metrics['assd_px']:.2f}px")
    if per_person:
        lines.append("Per person IoU: " + "  ".join(
            f"#{q['person']}={q['iou']:.3f}" for q in per_person[:8]))
    strip = np.full((20 + 28 * len(lines), top.shape[1], 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(strip, ln, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([top, bot, strip])


def debug_figure(display, heat, dbg, width=380):
    def gray(x):
        return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tiles = [titled(display, "input", width),
             titled(gray(heat), "decoded heat", width),
             titled(gray(dbg["background"]), "opening (background)", width),
             titled(gray(dbg["anomaly"]), "local anomaly", width),
             titled(gray(dbg["local"]), "warm parts + attached cool", width),
             titled(gray(dbg["glob"]), "global Otsu", width)]
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def select_rects(display):
    disp, s = resize_max(display, 1000)
    rs = cv2.selectROIs("Drag a box per person (ENTER after each), ESC when done",
                        disp, True, False)
    cv2.destroyAllWindows()
    return [[v / s for v in r] for r in rs if r[2] > 0 and r[3] > 0] or None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--outdir", default="results_thermal")
    ap.add_argument("--polarity", choices=["white", "black"], default="white",
                    help="white-hot (default): warm = bright; black-hot: warm = dark")
    ap.add_argument("--palette", default="auto",
                    help="auto | gray | luma | an OpenCV colormap name ("
                         + ", ".join(COLORMAPS) + ")")
    ap.add_argument("--pose", choices=["upright", "free"], default="upright",
                    help="upright: standing/walking people; free: any aspect ratio")
    ap.add_argument("--max-people", type=int, default=10)
    ap.add_argument("--rect", action="append",
                    help="person box x,y,w,h (repeat for several people)")
    ap.add_argument("--interactive", action="store_true", help="draw the person box(es)")
    ap.add_argument("--k-hi", type=float, default=4.0, help="strong warm threshold (sigmas)")
    ap.add_argument("--k-lo", type=float, default=2.0, help="weak warm threshold (sigmas)")
    ap.add_argument("--bg-frac", type=float, default=0.06,
                    help="top-hat disc radius as a fraction of the image side; "
                         "should exceed half a person's width (default 0.06)")
    ap.add_argument("--min-area", type=float, default=0.0005,
                    help="smallest person, as a fraction of the image area")
    ap.add_argument("--no-refine", action="store_true", help="skip the GrabCut refinement")
    ap.add_argument("--debug", action="store_true",
                    help="save the intermediate maps and print every candidate's features")
    g = ap.add_argument_group("SAM2 comparison")
    g.add_argument("--run-sam2", action="store_true")
    g.add_argument("--sam2-mask", help="pre-computed SAM2 mask (all people in one mask)")
    g.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    g.add_argument("--tol", type=float, default=None,
                   help="boundary tolerance in px (default: 0.5%% of the diagonal, min 2)")
    args = ap.parse_args()

    heat, display, how = load_heat(args.image, args.palette, args.polarity)
    H, W = heat.shape
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])
    print(f"[input] {W}x{H}  heat decoded from: {how}")

    rects = [[float(v) for v in r.split(",")] for r in args.rect] if args.rect else None
    if args.interactive and not rects:
        rects = select_rects(display)

    t0 = time.perf_counter()
    people, labels, dbg = segment_people(
        heat, pose=args.pose, max_people=args.max_people, rects=rects,
        refine=not args.no_refine, k_hi=args.k_hi, k_lo=args.k_lo,
        bg_frac=args.bg_frac, min_area_frac=args.min_area, verbose=args.debug)
    t_cls = time.perf_counter() - t0
    union = np.where(labels > 0, 255, 0).astype(np.uint8)
    masks = [np.where(labels == i + 1, 255, 0).astype(np.uint8) for i in range(len(people))]
    boxes = [p["box"] for p in people]
    print(f"[classical] {len(people)} person(s) found in {t_cls:.2f}s")
    for i, p in enumerate(people, 1):
        print(f"    #{i}: box={p['box']}  area={int((masks[i-1] > 0).sum())}px  "
              f"contrast={p['contrast']:.2f}  aspect={p['aspect']:.2f}  cue={p['source']}")

    thick = max(2, int(0.0025 * max(H, W)))
    cv2.imwrite(f"{stem}_mask.png", union)
    cv2.imwrite(f"{stem}_labels.png", labels)  # 16-bit, 0 = background, i = person i
    cutout = cv2.cvtColor(display, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = union
    cv2.imwrite(f"{stem}_cutout.png", cutout)
    cv2.imwrite(f"{stem}_overlay.png", draw_people(display, masks, thick, boxes))
    with open(f"{stem}_contours.json", "w") as f:
        json.dump({"image": args.image, "width": W, "height": H, "heat_source": how,
                   "people": [{"id": i + 1, "box_xywh": list(boxes[i]),
                               "contours": extract_boundaries(masks[i])}
                              for i in range(len(people))]}, f)
    if args.debug:
        cv2.imwrite(f"{stem}_debug.png", debug_figure(display, heat, dbg))

    if not (args.run_sam2 or args.sam2_mask):
        return
    if not people:
        sys.exit("No people found by the classical method; nothing to prompt SAM2 with. "
                 "Try --rect / --interactive.")
    pad_boxes = []
    for (x, y, w, h) in boxes:
        px, py = int(0.05 * w) + 2, int(0.05 * h) + 2
        x, y, w, h = clip_rect((x - px, y - py, w + 2 * px, h + 2 * py), W, H)
        pad_boxes.append((x, y, w, h))
    xyxy = [[x, y, x + w, y + h] for (x, y, w, h) in pad_boxes]

    t_sam, sam_scores, sam_masks = None, None, None
    if args.sam2_mask:
        sam_union = load_mask(args.sam2_mask, heat.shape)
    else:
        t0 = time.perf_counter()
        sam_masks, sam_scores = run_sam2(display, xyxy, args.sam2_model)
        t_sam = time.perf_counter() - t0
        sam_union = np.zeros_like(union)
        for m in sam_masks:
            sam_union |= m
        print(f"[sam2] {len(sam_masks)} box prompt(s), time={t_sam:.2f}s")
    cv2.imwrite(f"{stem}_sam2_mask.png", sam_union)

    tol = args.tol if args.tol is not None else max(2.0, 0.005 * np.hypot(H, W))
    metrics = compare_masks(union, sam_union, tol)
    per_person = []
    for i, m in enumerate(masks):
        if sam_masks is not None:
            q = compare_masks(m, sam_masks[i], tol)
        else:  # one combined SAM2 mask: compare both unions inside this person's box
            boxm = cv2.rectangle(np.zeros_like(union), (xyxy[i][0], xyxy[i][1]),
                                 (xyxy[i][2], xyxy[i][3]), 255, -1)
            q = compare_masks(union & boxm, sam_union & boxm, tol)
        q["person"] = i + 1
        per_person.append(q)
    metrics.update({"time_classical_s": round(t_cls, 3),
                    "time_sam2_s": None if t_sam is None else round(t_sam, 3),
                    "sam2_scores": sam_scores, "boxes_xyxy": xyxy, "per_person": per_person})
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cv2.imwrite(f"{stem}_comparison.png",
                comparison_figure(display, union, sam_union, metrics, per_person, pad_boxes))
    print("[compare] all people: " + "  ".join(
        f"{k}={metrics[k]}" for k in ("iou", "dice", "boundary_f1", "hd95_px", "assd_px")
        if k in metrics))
    for q in per_person:
        print(f"    #{q['person']}: IoU={q['iou']}  Dice={q['dice']}  "
              f"BF1={q.get('boundary_f1')}  HD95={q.get('hd95_px')}")


if __name__ == "__main__":
    main()
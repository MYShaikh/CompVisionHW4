#!/usr/bin/env python3
"""
human_boundary.py
=================
Find the exact boundary of a human in a single RGB image using ONLY classical
computer vision (OpenCV + NumPy). No deep learning and no trained / learned
models are used in the segmentation pipeline. (No HOG-SVM, no Haar cascades,
no k-means.) The result can optionally be compared against SAM2.

Classical pipeline
------------------
1. Localisation (automatic, or a user-given / mouse-selected rectangle)
   a0. Face anchor: the most face-shaped blob of rule-based skin pixels. When
      found, simple body proportions (head ellipse, neck, shoulders about 4
      face widths wide, torso down toward the frame bottom) give an upright
      body-shape prior. With --pose upright, pixels well outside this shape
      are forced to background; with --pose free they are only "probable"
      background, so spread-out arms and legs can still be recovered.
   a. Background-contrast map: the image border is split into patches, and the
      median Lab colour of each patch is taken as a background colour sample.
      Each pixel's distance to the nearest background colour gives a
      "not-background" score.
   b. Spectral-residual saliency (Hou & Zhang 2007), implemented with an FFT.
   c. Rule-based skin detection (fixed YCrCb + HSV thresholds).
   d. A centre prior, since the subject is usually near the middle.
   The weighted combination is thresholded with Otsu's method, cleaned with
   morphology, and the strongest connected blob gives the person's ROI.

2. Coarse segmentation: GrabCut (graph-cut energy minimisation) is initialised
   with a mask built from step 1. The mask marks definite background outside
   the ROI, probable foreground/background inside it, and a small definite
   foreground core. This pass runs on a down-scaled copy for speed.

3. Boundary refinement at full resolution. A narrow uncertainty band is built
   around the up-scaled coarse contour. Pixels inside the band are
   re-labelled by a second GrabCut pass whose pairwise term snaps the
   boundary to the real image edges.

4. Post-processing: keep the main connected component(s), fill small holes,
   and lightly smooth the result. The exact boundary is then extracted with
   cv2.findContours(CHAIN_APPROX_NONE), which returns every boundary pixel.

Comparison with SAM2
--------------------
SAM2 is prompted with the same box that the classical method used, so both
methods receive the same localisation information. The script reports:
IoU, Dice, pixel precision/recall, the boundary F-score at a pixel tolerance,
the Hausdorff distance, HD95 and the average symmetric surface distance (ASSD).
It also writes a side-by-side visual comparison.

SAM2 is used only as a reference, and only when it is requested. There are
two ways to use it:
  * --run-sam2 runs SAM2 locally. This needs:
        pip install torch torchvision
        pip install "git+https://github.com/facebookresearch/sam2.git"
    The checkpoint is downloaded from Hugging Face on first use.
  * --sam2-mask PATH loads a mask exported from the SAM2 web demo
    (https://sam2.metademolab.com) or from any other SAM2 run.

Usage
-----
    python human_boundary.py person.jpg                        # automatic
    python human_boundary.py person.jpg --interactive          # drag a box
    python human_boundary.py person.jpg --rect 120,40,300,620  # x,y,w,h
    python human_boundary.py action.jpg --pose free            # spread limbs
    python human_boundary.py person.jpg --run-sam2             # + compare
    python human_boundary.py person.jpg --sam2-mask sam2.png   # + compare

Outputs (written to --outdir, default ./results)
    <name>_mask.png          binary mask (255 = person)
    <name>_cutout.png        RGBA cut-out with a transparent background
    <name>_overlay.png       image with the boundary drawn on it
    <name>_contours.json     exact boundary pixel coordinates (outer + holes)
    <name>_debug.png         (--debug) intermediate maps
    <name>_sam2_mask.png     SAM2 mask         (when comparing)
    <name>_comparison.png    visual comparison (when comparing)
    <name>_metrics.json      comparison metrics (when comparing)
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def resize_max(img, max_dim):
    """Down-scale so that max(h, w) <= max_dim. Returns (image, scale)."""
    h, w = img.shape[:2]
    s = min(1.0, float(max_dim) / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img, s


def normalize01(x):
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def ellipse_kernel(radius):
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def clip_rect(rect, w, h):
    x, y, rw, rh = [int(round(v)) for v in rect]
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    return x, y, max(1, rw), max(1, rh)


# --------------------------------------------------------------------------- #
# Cue maps (all hand-designed, no learning)
# --------------------------------------------------------------------------- #
def spectral_residual_saliency(bgr, size=128):
    """Hou & Zhang (CVPR 2007) spectral-residual saliency, computed with an FFT."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8).astype(np.float32)
    phase = np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
    return normalize01(sal)


def border_color_contrast(bgr, n_per_side=8, strip_frac=0.04, use_bottom=False):
    """Distance, in Lab space, from each pixel to the nearest border-patch colour.

    By default the bottom edge is NOT sampled, because people are usually cut
    off by the bottom of the frame (webcam and portrait shots). Enable it with
    --bottom-is-background for full-body shots with floor along the bottom.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    s = max(3, int(strip_frac * min(h, w)))
    xs = np.linspace(0, w, n_per_side + 1).astype(int)
    ys = np.linspace(0, h, n_per_side + 1).astype(int)
    samples = []

    def add(region):
        if region.size:
            samples.append(np.median(region.reshape(-1, 3), axis=0))

    for i in range(n_per_side):
        add(lab[:s, xs[i]:xs[i + 1]])                     # top
        add(lab[ys[i]:ys[i + 1], :s])                     # left
        add(lab[ys[i]:ys[i + 1], w - s:])                 # right
        if use_bottom:
            add(lab[h - s:, xs[i]:xs[i + 1]])             # bottom

    d = np.full((h, w), np.inf, np.float32)
    for c in samples:
        d = np.minimum(d, np.linalg.norm(lab - c, axis=2))
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    return np.clip(d / (np.percentile(d, 95) + 1e-6), 0, 1)


def skin_map(bgr):
    """Fixed-threshold skin detector (YCrCb AND HSV rules)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135))
    m2 = cv2.inRange(hsv, (0, 15, 40), (25, 200, 255)) | \
        cv2.inRange(hsv, (165, 15, 40), (180, 200, 255))
    skin = cv2.morphologyEx(m1 & m2, cv2.MORPH_OPEN, ellipse_kernel(1))
    return cv2.GaussianBlur(skin.astype(np.float32) / 255.0, (0, 0), 2)


def find_face(skin, cp, min_area):
    """Pick the most face-like skin blob, using a compact, roughly upright
    shape near the centre. Returns (x, y, w, h, blob mask) or None."""
    m = (skin > 0.5).astype(np.uint8) * 255
    diag = np.hypot(*m.shape)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse_kernel(0.004 * diag))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    best, best_s = None, 0.0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        fill, aspect = a / float(w * h), h / float(w)
        if fill < 0.45 or not (0.7 <= aspect <= 2.2):
            continue  # rejects thin strips, pillows, scattered skin-coloured texture
        sc = a * fill * cp[int(cen[i][1]), int(cen[i][0])]
        if sc > best_s:
            best, best_s = i, sc
    if best is None:
        return None
    x, y, w, h, _ = st[best]
    blob = (lbl == best).astype(np.uint8) * 255
    # Measure the face from the upper half of the blob only, so that the neck
    # or skin-coloured objects touching the chin do not inflate its size.
    rows = blob[y:y + max(2, h // 2), x:x + w] > 0
    widths = rows.sum(axis=1)
    widths = widths[widths > 0]
    fw = float(np.median(widths)) if widths.size else float(w)
    cols = np.where(rows.any(axis=0))[0]
    cx = x + (cols.min() + cols.max()) / 2.0 if cols.size else x + w / 2.0
    fh = 1.3 * fw                              # typical face height/width ratio
    fx = cx - fw / 2.0
    core = np.zeros_like(blob)
    core[int(y):int(min(blob.shape[0], y + fh)), int(max(0, fx)):int(fx + fw)] = 255
    return fx, float(y), fw, fh, blob & core


def body_prior_from_face(face, h, w):
    """Upright-person shape from simple anthropometric ratios
    (shoulders ~ 4-5 face widths wide; body ~ 7-8 face heights tall)."""
    fx, fy, fw, fh, _ = face
    cx = fx + fw / 2.0
    shoulder_y = fy + 1.3 * fh
    bottom = min(h - 1, fy + 8.0 * fh)
    # torso: neck -> shoulders -> down to the bottom of the body / frame
    poly = np.array([
        [cx - 0.55 * fw, fy + 0.9 * fh], [cx + 0.55 * fw, fy + 0.9 * fh],
        [cx + 0.7 * fw, fy + 1.1 * fh],
        [cx + 1.9 * fw, shoulder_y], [cx + 2.3 * fw, bottom],
        [cx - 2.3 * fw, bottom], [cx - 1.9 * fw, shoulder_y],
        [cx - 0.7 * fw, fy + 1.1 * fh]], np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [poly.astype(np.int32)], 255)
    # head (face + hair) as an ellipse
    cv2.ellipse(region, (int(cx), int(fy + 0.4 * fh)),
                (int(0.75 * fw), int(0.85 * fh)), 0, 0, 360, 255, -1)
    soft = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (0, 0), 0.4 * fw)
    # small definite-foreground cores: centre of the face and centre of the chest
    torso = np.zeros((h, w), np.uint8)
    ty0, ty1 = int(fy + 1.6 * fh), int(min(h, fy + 3.2 * fh))
    tx0, tx1 = int(max(0, cx - 0.5 * fw)), int(min(w, cx + 0.5 * fw))
    if ty1 > ty0:
        torso[ty0:ty1, tx0:tx1] = 255
    # anything further than ~0.6 face widths outside the body shape is
    # treated as definite background (upright-person assumption)
    near = cv2.dilate(region, ellipse_kernel(0.6 * fw))
    return region, soft, torso, near


def center_prior(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-0.5 * (((xx - w / 2) / (0.35 * w)) ** 2 +
                          ((yy - h / 2) / (0.50 * h)) ** 2))


# --------------------------------------------------------------------------- #
# Stage 1+2: localisation and coarse GrabCut
# --------------------------------------------------------------------------- #
def coarse_segment(bgr, rect=None, iters=6, use_bottom=False, use_face=True,
                   pose="upright"):
    """Returns (binary mask 0/255, rect used, dict of debug maps)."""
    h, w = bgr.shape[:2]
    diag = np.hypot(h, w)
    smooth = cv2.bilateralFilter(bgr, 9, 50, 7)

    contrast = border_color_contrast(smooth, use_bottom=use_bottom)
    saliency = spectral_residual_saliency(smooth)
    skin = skin_map(smooth)
    cp = center_prior(h, w)

    face = find_face(skin, cp, 0.002 * h * w) if use_face else None
    if face is not None and rect is not None:
        x, y, rw, rh = rect
        fcx, fcy = face[0] + face[2] / 2, face[1] + face[3] / 2
        if not (x <= fcx <= x + rw and y <= fcy <= y + rh):
            face = None
    body_region = body_torso = body_near = None
    if face is not None:
        body_region, body_soft, body_torso, body_near = body_prior_from_face(face, h, w)
        cp = np.maximum(body_soft, 0.15)   # replaces the generic centre prior

    score = (0.50 * contrast + 0.30 * saliency + 0.20 * skin)
    if rect is None:
        score = score * (0.5 + 0.5 * cp)
    score = normalize01(score)
    score_u8 = (score * 255).astype(np.uint8)

    # Otsu threshold, computed inside the ROI if one was given
    if rect is not None:
        x, y, rw, rh = rect
        roi_vals = score_u8[y:y + rh, x:x + rw].reshape(-1, 1)
        t, _ = cv2.threshold(roi_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t, _ = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_guess = (score_u8 > t).astype(np.uint8) * 255
    if rect is not None:
        roi_mask = np.zeros_like(fg_guess)
        x, y, rw, rh = rect
        roi_mask[y:y + rh, x:x + rw] = 255
        fg_guess &= roi_mask

    k = ellipse_kernel(0.01 * diag)
    fg_guess = cv2.morphologyEx(fg_guess, cv2.MORPH_CLOSE, k)
    fg_guess = cv2.morphologyEx(fg_guess, cv2.MORPH_OPEN, ellipse_kernel(0.004 * diag))

    # Strongest blob: the largest sum of (score x centre prior)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg_guess, connectivity=8)
    if n > 1:
        sums = np.bincount(lbl.ravel(), weights=(score * cp).ravel(), minlength=n)
        sums[0] = 0
        best = int(np.argmax(sums))
        blob = (lbl == best).astype(np.uint8) * 255
    else:
        blob = np.zeros_like(fg_guess)
        cv2.ellipse(blob, (w // 2, h // 2), (w // 6, h // 3), 0, 0, 360, 255, -1)

    if rect is None and body_region is not None:
        roi_src = body_near if pose == "upright" else (body_near | blob)
        bx, by, bw, bh = cv2.boundingRect(roi_src)
        pad = 0
        rect = clip_rect((bx - pad, by - pad, bw + 2 * pad, bh + 2 * pad), w, h)
    elif rect is None:
        bx, by, bw, bh = cv2.boundingRect(blob)
        pad_x, pad_y = int(0.08 * bw) + 2, int(0.06 * bh) + 2
        rect = clip_rect((bx - pad_x, by - pad_y, bw + 2 * pad_x, bh + 2 * pad_y), w, h)
    x, y, rw, rh = rect

    # ---- GrabCut initialisation mask ----
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[y:y + rh, x:x + rw] = cv2.GC_PR_BGD
    inside = np.zeros((h, w), bool)
    inside[y:y + rh, x:x + rw] = True
    if body_region is not None:
        # inside the body shape everything starts as probable foreground; this
        # matters for dark or low-contrast clothing
        gc[inside & (body_region > 0) & (contrast > 0.10)] = cv2.GC_PR_FGD
        gc[inside & (fg_guess > 0) & (cp > 0.5)] = cv2.GC_PR_FGD
        if pose == "upright":
            # hard shape constraint: far outside the upright body = background
            gc[body_near == 0] = cv2.GC_BGD
        else:
            # soft: limbs may reach outside the body shape
            gc[inside & (body_near == 0)] = cv2.GC_PR_BGD
        # colours that match the image border and lie outside the body shape
        gc[inside & (body_region == 0) & (contrast < 0.05)] = cv2.GC_BGD
        face_core = cv2.erode(face[4], ellipse_kernel(0.15 * face[2]))
        gc[inside & (face_core > 0)] = cv2.GC_FGD
        gc[inside & (body_torso > 0)] = cv2.GC_FGD
    else:
        gc[inside & (fg_guess > 0)] = cv2.GC_PR_FGD
        # A small definite-foreground core, taken from the strongly eroded blob
        core = cv2.erode(blob, ellipse_kernel(0.03 * diag))
        if core.any():
            gc[inside & (core > 0)] = cv2.GC_FGD
    # GrabCut needs samples of both classes
    if not np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD]).any():
        cv2.ellipse(gc, (x + rw // 2, y + rh // 2), (rw // 4, rh // 3), 0, 0, 360,
                    int(cv2.GC_PR_FGD), -1)
    if not np.isin(gc, [cv2.GC_BGD, cv2.GC_PR_BGD]).any():
        gc[:2, :] = gc[-2:, :] = gc[:, :2] = gc[:, -2:] = cv2.GC_PR_BGD

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(smooth, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    debug = dict(contrast=contrast, saliency=saliency, skin=skin, score=score,
                 fg_guess=fg_guess,
                 blob=body_region if body_region is not None else blob,
                 face=None if face is None else tuple(int(v) for v in face[:4]))
    return mask, rect, debug


# --------------------------------------------------------------------------- #
# Stage 3: full-resolution boundary refinement
# --------------------------------------------------------------------------- #
def refine_boundary(bgr, mask, band, iters=3):
    """Re-segment a narrow band around the contour with GrabCut at full res."""
    h, w = mask.shape
    if not mask.any():
        return mask
    k = ellipse_kernel(band)
    inner = cv2.erode(mask, k)
    outer = cv2.dilate(mask, k)

    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[outer > 0] = cv2.GC_PR_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[inner > 0] = cv2.GC_FGD

    # Work on a crop around the band for speed. The margin keeps enough
    # background pixels for the colour model.
    bx, by, bw, bh = cv2.boundingRect(outer)
    m = 3 * band
    x0, y0 = max(0, bx - m), max(0, by - m)
    x1, y1 = min(w, bx + bw + m), min(h, by + bh + m)
    sub_img = cv2.bilateralFilter(bgr[y0:y1, x0:x1], 5, 30, 5)
    sub_gc = gc[y0:y1, x0:x1].copy()
    if not np.isin(sub_gc, [cv2.GC_BGD, cv2.GC_PR_BGD]).any():
        return mask  # the person fills the frame; nothing to refine against

    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(sub_img, sub_gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = np.where(
        (sub_gc == cv2.GC_FGD) | (sub_gc == cv2.GC_PR_FGD), 255, 0)
    return out


# --------------------------------------------------------------------------- #
# Stage 4: post-processing and boundary extraction
# --------------------------------------------------------------------------- #
def keep_main_components(mask, keep_ratio=0.15):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= keep_ratio * areas.max())[0]
    return np.isin(lbl, keep).astype(np.uint8) * 255


def fill_small_holes(mask, max_area):
    """Fill holes smaller than max_area. Larger holes, such as the gap
    between an arm and the body, are left open."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask
    out = mask.copy()
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 and cv2.contourArea(c) < max_area:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out


def postprocess(mask, sigma=1.0):
    h, w = mask.shape
    diag = np.hypot(h, w)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(0.002 * diag))
    mask = keep_main_components(mask)
    mask = fill_small_holes(mask, max_area=0.002 * h * w)
    if sigma > 0:  # light anti-staircase smoothing
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    return mask


def extract_boundaries(mask):
    """Every boundary pixel, as outer contours and hole contours."""
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    result = []
    if hier is None:
        return cnts, result
    for i, c in enumerate(cnts):
        result.append({
            "type": "outer" if hier[0][i][3] == -1 else "hole",
            "area_px": float(cv2.contourArea(c)),
            "perimeter_px": float(cv2.arcLength(c, True)),
            "points": c.reshape(-1, 2).tolist(),
        })
    return cnts, result


def segment_human(bgr, rect=None, work_size=800, refine=True, use_bottom=False,
                  use_face=True, pose="upright"):
    """Full classical pipeline. Returns (mask 0/255 at full res, rect, debug)."""
    H, W = bgr.shape[:2]
    work, s = resize_max(bgr, work_size)
    rect_w = None
    if rect is not None:
        rect_w = clip_rect([v * s for v in rect], work.shape[1], work.shape[0])

    mask_w, rect_w, debug = coarse_segment(work, rect_w, use_bottom=use_bottom,
                                           use_face=use_face, pose=pose)
    mask_w = postprocess(mask_w, sigma=0)

    # Up-sample to full resolution (soft interpolation, then threshold)
    mask = cv2.resize(mask_w, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
    rect_full = clip_rect([v / s for v in rect_w], W, H)

    if refine:
        band = max(4, int(np.ceil(2.0 / s)) + 2, int(0.006 * np.hypot(H, W)))
        mask = refine_boundary(bgr, mask, band)
    mask = postprocess(mask, sigma=1.0)
    debug["work_scale"] = s
    return mask, rect_full, debug


# --------------------------------------------------------------------------- #
# SAM2 reference (deep learning, used only for comparison)
# --------------------------------------------------------------------------- #
def run_sam2(bgr, box_xyxy, model_id="facebook/sam2.1-hiera-large", device=None):
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        sys.exit("SAM2 is not installed. Run:\n"
                 "  pip install torch torchvision\n"
                 "  pip install 'git+https://github.com/facebookresearch/sam2.git'\n"
                 "or pass a pre-computed mask with --sam2-mask.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 ("mps" if torch.backends.mps.is_available() else "cpu")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    with torch.inference_mode():
        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            box=np.asarray(box_xyxy, np.float32)[None, :], multimask_output=True)
    best = masks[int(np.argmax(scores))]
    return (best > 0.5).astype(np.uint8) * 255, float(np.max(scores))


def load_mask(path, shape):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        sys.exit(f"Cannot read mask: {path}")
    if m.ndim == 3:
        # An RGBA cut-out uses alpha as the mask; otherwise use any non-zero channel
        m = m[:, :, 3] if m.shape[2] == 4 else m.max(axis=2)
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #
def _boundary(m):
    m = (m > 0).astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def compare_masks(pred, ref, tol):
    p, r = pred > 0, ref > 0
    inter, union = np.logical_and(p, r).sum(), np.logical_or(p, r).sum()
    ps, rs = p.sum(), r.sum()
    out = {
        "iou": inter / union if union else 1.0,
        "dice": 2 * inter / (ps + rs) if (ps + rs) else 1.0,
        "pixel_precision": inter / ps if ps else 0.0,
        "pixel_recall": inter / rs if rs else 0.0,
        "area_classical_px": int(ps), "area_sam2_px": int(rs),
        "boundary_tolerance_px": float(tol),
    }
    bp, br = _boundary(pred), _boundary(ref)
    if bp.any() and br.any():
        dt_r = cv2.distanceTransform(np.where(br, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        dt_p = cv2.distanceTransform(np.where(bp, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
        d_pr, d_rp = dt_r[bp], dt_p[br]
        bprec, brec = (d_pr <= tol).mean(), (d_rp <= tol).mean()
        out.update({
            "boundary_precision": bprec, "boundary_recall": brec,
            "boundary_f1": 2 * bprec * brec / (bprec + brec + 1e-12),
            "hausdorff_px": float(max(d_pr.max(), d_rp.max())),
            "hd95_px": float(max(np.percentile(d_pr, 95), np.percentile(d_rp, 95))),
            "assd_px": float((d_pr.sum() + d_rp.sum()) / (d_pr.size + d_rp.size)),
        })
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def draw_overlay(bgr, mask, color, thickness):
    out = bgr.copy()
    tint = np.zeros_like(bgr)
    tint[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, tint, 0.30, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, thickness, cv2.LINE_AA)
    return out


def titled(img, title, width):
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    bar = np.full((34, width, 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def comparison_figure(bgr, ours, sam, metrics, rect, width=520):
    t = max(2, int(0.003 * max(bgr.shape[:2])))
    GREEN, MAGENTA = (60, 220, 60), (230, 60, 230)
    a = draw_overlay(bgr, ours, GREEN, t)
    x, y, w, h = rect
    cv2.rectangle(a, (x, y), (x + w, y + h), (0, 200, 255), t)
    b = draw_overlay(bgr, sam, MAGENTA, t)
    c = bgr.copy()
    for m, col in ((sam, MAGENTA), (ours, GREEN)):
        cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c, cnts, -1, col, t, cv2.LINE_AA)
    d = np.zeros_like(bgr)
    p, r = ours > 0, sam > 0
    d[p & r] = (255, 255, 255)
    d[p & ~r] = GREEN
    d[~p & r] = MAGENTA

    top = np.hstack([titled(a, "Classical (OpenCV) + ROI", width),
                     titled(b, "SAM2 (reference)", width)])
    bot = np.hstack([titled(c, "Boundaries: green=classical  magenta=SAM2", width),
                     titled(d, "white=both  green/magenta=only one method", width)])
    lines = [
        f"IoU {metrics['iou']:.4f}   Dice {metrics['dice']:.4f}   "
        f"Precision {metrics['pixel_precision']:.4f}   Recall {metrics['pixel_recall']:.4f}",
    ]
    if "boundary_f1" in metrics:
        lines.append(
            f"Boundary F1@{metrics['boundary_tolerance_px']:.1f}px {metrics['boundary_f1']:.4f}   "
            f"HD95 {metrics['hd95_px']:.1f}px   Hausdorff {metrics['hausdorff_px']:.1f}px   "
            f"ASSD {metrics['assd_px']:.2f}px")
    strip = np.full((20 + 28 * len(lines), top.shape[1], 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(strip, ln, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([top, bot, strip])


def debug_figure(bgr, debug, width=360):
    def gray(x):
        return cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    work = cv2.resize(bgr, (debug["score"].shape[1], debug["score"].shape[0]))
    tiles = [titled(work, "input (work scale)", width),
             titled(gray(debug["contrast"]), "border-colour contrast", width),
             titled(gray(debug["saliency"]), "spectral-residual saliency", width),
             titled(gray(debug["skin"]), "skin (rule-based)", width),
             titled(gray(debug["score"]), "combined score", width),
             titled(gray(debug["blob"] / 255.0), "person prior / blob", width)]
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def select_rect_interactively(bgr):
    disp, s = resize_max(bgr, 1000)
    r = cv2.selectROI("Drag a box around the person, then press ENTER", disp, True, False)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        return None
    return [v / s for v in r]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--rect", help="person ROI as x,y,w,h in pixels (optional)")
    ap.add_argument("--interactive", action="store_true", help="draw the ROI with the mouse")
    ap.add_argument("--work-size", type=int, default=800,
                    help="max side length for the coarse pass (default 800)")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the full-resolution boundary refinement")
    ap.add_argument("--debug", action="store_true", help="save the intermediate cue maps")
    ap.add_argument("--bottom-is-background", action="store_true",
                    help="also sample background colours from the bottom edge "
                         "(full-body shots where the floor is visible)")
    ap.add_argument("--pose", choices=["upright", "free"], default="upright",
                    help="upright (default): standing/sitting person facing the "
                         "camera, e.g. webcam or portrait; free: action poses "
                         "with arms/legs spread out")
    ap.add_argument("--no-face-prior", action="store_true",
                    help="disable the face-based upright body prior")
    g = ap.add_argument_group("SAM2 comparison")
    g.add_argument("--run-sam2", action="store_true", help="run SAM2 locally")
    g.add_argument("--sam2-mask", help="pre-computed SAM2 mask image")
    g.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    g.add_argument("--sam2-prompt", choices=["roi", "tight"], default="roi",
                   help="box given to SAM2: the classical ROI (default) or the "
                        "tight bbox of the classical mask")
    g.add_argument("--tol", type=float, default=None,
                   help="boundary tolerance in px (default: 0.5%% of the diagonal, min 2)")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        sys.exit(f"Cannot read image: {args.image}")
    H, W = bgr.shape[:2]
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    rect = None
    if args.rect:
        rect = clip_rect([float(v) for v in args.rect.split(",")], W, H)
    elif args.interactive:
        rect = select_rect_interactively(bgr)

    t0 = time.perf_counter()
    mask, rect, debug = segment_human(bgr, rect, args.work_size, refine=not args.no_refine,
                                      use_bottom=args.bottom_is_background,
                                      use_face=not args.no_face_prior,
                                      pose=args.pose)
    t_cls = time.perf_counter() - t0
    print(f"[classical] face prior: {'found' if debug.get('face') else 'not found'}")
    print(f"[classical] {W}x{H}  ROI={rect}  time={t_cls:.2f}s  "
          f"area={int((mask > 0).sum())}px")

    # ---- save the classical outputs ----
    cv2.imwrite(f"{stem}_mask.png", mask)
    cutout = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = mask
    cv2.imwrite(f"{stem}_cutout.png", cutout)
    thick = max(2, int(0.003 * max(H, W)))
    cv2.imwrite(f"{stem}_overlay.png", draw_overlay(bgr, mask, (60, 220, 60), thick))
    _, boundaries = extract_boundaries(mask)
    with open(f"{stem}_contours.json", "w") as f:
        json.dump({"image": args.image, "width": W, "height": H, "roi_xywh": list(rect),
                   "contours": boundaries}, f)
    if args.debug:
        cv2.imwrite(f"{stem}_debug.png", debug_figure(bgr, debug))
    n_out = sum(b["type"] == "outer" for b in boundaries)
    print(f"[classical] {n_out} outer contour(s), "
          f"{len(boundaries) - n_out} hole(s), "
          f"{sum(len(b['points']) for b in boundaries)} boundary pixels")

    # ---- optional SAM2 comparison ----
    if not (args.run_sam2 or args.sam2_mask):
        return
    if args.sam2_prompt == "tight" and mask.any():
        bx, by, bw, bh = cv2.boundingRect(mask)
    else:
        bx, by, bw, bh = rect
    box = [bx, by, bx + bw, by + bh]

    if args.sam2_mask:
        sam = load_mask(args.sam2_mask, bgr.shape)
        t_sam, sam_score = None, None
    else:
        t0 = time.perf_counter()
        sam, sam_score = run_sam2(bgr, box, args.sam2_model)
        t_sam = time.perf_counter() - t0
        print(f"[sam2] box={box}  score={sam_score:.3f}  time={t_sam:.2f}s")
    cv2.imwrite(f"{stem}_sam2_mask.png", sam)

    tol = args.tol if args.tol is not None else max(2.0, 0.005 * np.hypot(H, W))
    metrics = compare_masks(mask, sam, tol)
    metrics.update({"time_classical_s": round(t_cls, 3),
                    "time_sam2_s": None if t_sam is None else round(t_sam, 3),
                    "sam2_predicted_iou": sam_score, "sam2_box_xyxy": box})
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cv2.imwrite(f"{stem}_comparison.png", comparison_figure(bgr, mask, sam, metrics, rect))
    print("[compare] " + "  ".join(f"{k}={v}" for k, v in metrics.items()
                                   if k not in ("sam2_box_xyxy",)))


if __name__ == "__main__":
    main()
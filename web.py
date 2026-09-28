"""
web.py - Web application for the human boundary assignment.

How to run (in this folder):
    python -m pip install streamlit
    python -m streamlit run web.py

Tab 1 runs app.py (colour photo), tab 2 runs thermal.py (thermal image).
Both use classical OpenCV only. Upload a SAM2 mask to compare with SAM2.
"""
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

import app as rgb        # classical method for colour photos (app.py)
import thermal as th     # classical method for thermal images (thermal.py)

TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
st.set_page_config(page_title="Human boundary detection", layout="wide")
st.title("Human boundary detection with classical OpenCV")
st.write("No deep learning is used. SAM2 is only used as a comparison reference.")


def read_image(upload, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), flags)


def read_mask(upload, shape):
    m = read_image(upload, cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 3] if m.shape[2] == 4 else m.max(axis=2)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def show_metrics(m):
    names = [("iou", "IoU"), ("dice", "Dice"), ("pixel_precision", "Precision"),
             ("pixel_recall", "Recall"), ("boundary_f1", "Boundary F1"),
             ("hd95_px", "HD95 (px)"), ("assd_px", "ASSD (px)")]
    st.table({label: [m[k]] for k, label in names if k in m})


tab1, tab2 = st.tabs(["Colour photo", "Thermal image"])

# ---------------- Colour photo (app.py) ----------------
with tab1:
    up = st.file_uploader("Photo of a person", type=TYPES, key="rgb_img")
    pose = st.radio("Pose", ["upright", "free"], horizontal=True, key="rgb_pose")
    if up:
        bgr = read_image(up)
        if st.button("Segment the person", key="rgb_go"):
            t0 = time.perf_counter()
            mask, rect, _ = rgb.segment_human(bgr, pose=pose)
            st.session_state["rgb"] = ((up.name, pose), mask, rect, time.perf_counter() - t0)
        res = st.session_state.get("rgb")
        if res and res[0] == (up.name, pose):
            _, mask, rect, secs = res
            thick = max(2, int(0.003 * max(bgr.shape[:2])))
            st.image(rgb.draw_overlay(bgr, mask, (60, 220, 60), thick), channels="BGR",
                     caption=f"Classical result in {secs:.1f} s")
            sam_up = st.file_uploader("SAM2 mask to compare (e.g. myphoto_sam2_mask.png)",
                                      type=TYPES, key="rgb_sam")
            if sam_up:
                sam = read_mask(sam_up, bgr.shape)
                metrics = rgb.compare_masks(mask, sam, max(2.0, 0.005 * np.hypot(*bgr.shape[:2])))
                st.image(rgb.comparison_figure(bgr, mask, sam, metrics, rect), channels="BGR")
                show_metrics(metrics)

# ---------------- Thermal image (thermal.py) ----------------
with tab2:
    up = st.file_uploader("Thermal image", type=TYPES, key="th_img")
    c1, c2 = st.columns(2)
    polarity = c1.radio("Heat shown as", ["white", "black"], horizontal=True, key="th_pol",
                        format_func=lambda p: "white-hot" if p == "white" else "black-hot")
    palette = c2.selectbox("Colour palette", ["auto", "luma", "gray"] + list(th.COLORMAPS),
                           key="th_pal")
    if up:
        suffix = os.path.splitext(up.name)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(up.getvalue())
        try:
            heat, display, how = th.load_heat(f.name, palette, polarity)
        except SystemExit:
            st.error("This file can't be read as an image.")
            st.stop()
        finally:
            os.unlink(f.name)
        settings = (up.name, polarity, palette)
        if st.button("Find people", key="th_go"):
            t0 = time.perf_counter()
            people, labels, _ = th.segment_people(heat)
            masks = [np.where(labels == i + 1, 255, 0).astype(np.uint8)
                     for i in range(len(people))]
            st.session_state["th"] = (settings, masks, [p["box"] for p in people],
                                      time.perf_counter() - t0)
        res = st.session_state.get("th")
        if res and res[0] == settings:
            _, masks, boxes, secs = res
            if not masks:
                st.warning("No people found. Make sure this is a thermal image and check "
                           "the white-hot / black-hot setting.")
            else:
                H, W = heat.shape
                union = np.zeros((H, W), np.uint8)
                for m in masks:
                    union |= m
                thick = max(2, int(0.0025 * max(H, W)))
                st.image(th.draw_people(display, masks, thick, boxes), channels="BGR",
                         caption=f"{len(masks)} person(s) found in {secs:.1f} s")
                sam_up = st.file_uploader("SAM2 mask to compare (e.g. thermal_sam2_mask.png)",
                                          type=TYPES, key="th_sam")
                if sam_up:
                    sam = read_mask(sam_up, (H, W))
                    tol = max(2.0, 0.005 * np.hypot(H, W))
                    metrics = th.compare_masks(union, sam, tol)
                    per_person = []
                    for i, (x, y, w, h) in enumerate(boxes):
                        box = np.zeros_like(union)
                        box[y:y + h, x:x + w] = 255
                        q = th.compare_masks(union & box, sam & box, tol)
                        q["person"] = i + 1
                        per_person.append(q)
                    st.image(th.comparison_figure(display, union, sam, metrics, per_person, boxes),
                             channels="BGR")
                    show_metrics(metrics)
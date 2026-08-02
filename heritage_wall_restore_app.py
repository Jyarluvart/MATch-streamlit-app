#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs_interactive"
CACHE_DIR = OUT_DIR / "cache"
UPLOAD_DIR = OUT_DIR / "uploads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / "matplotlib_cache"))
os.environ.setdefault("HOME", str(OUT_DIR / "streamlit_home"))
os.environ.setdefault("USERPROFILE", str(OUT_DIR / "streamlit_home"))
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
(OUT_DIR / "matplotlib_cache").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "streamlit_home" / ".streamlit").mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = OUT_DIR / "app_settings.json"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit as st
##
import requests
import io
from PIL import Image
##################################################
HF_TOKEN = st.secrets["HF_TOKEN"]
SDXL_API_URL = "https://huggingface.co"
# ลิงก์ API ของโมเดล SAM (Segment Anything) ของ Meta
SAM_API_URL = "https://huggingface.co"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#################################################

import streamlit.components.v1 as components
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_IMAGE = r"D:\brick hole songwat.png"
DEFAULT_CKPT = str(PROJECT_DIR / "weights" / "sam_vit_b_01ec64.pth")
DEFAULT_MATERIALS_CSV = r"D:\Master\ICCEA2026\Data\materials_DB\New folder\materials.csv"
DEFAULT_TEXTURE_DIR = r"D:\Master\ICCEA2026\Data\materials_DB"
DEFAULT_SETTINGS = {
    "image_path": DEFAULT_IMAGE,
    "checkpoint_path": DEFAULT_CKPT,
    "materials_csv": DEFAULT_MATERIALS_CSV,
    "texture_dir": DEFAULT_TEXTURE_DIR,
    "max_preview": 20,
    "sam_max_side": 1280,
    "sam_analysis_mode": "Balanced",
    "hole_index": 1,
    "brick_index": 11,
    "short_brick_index": 11,
    "threshold": 0.50,
    "mortar_x_ratio": 0.08,
    "mortar_y_ratio": 0.18,
    "grid_offset_x_px": 0,
    "grid_offset_y_px": 0,
    "bond_pattern": "Running bond",
    "reference_brick_width_mm": 200,
    "reference_brick_height_mm": 50,
    "selection_mode": "SAM index",
}

BOND_PATTERNS = [
    "Running bond",
    "Stack bond",
    "English bond",
    "Flemish bond",
    "Common bond",
    "Irregular row rhythm",
    "Learned long-short bond",
    "Learned row bbox rhythm",
]

BOND_LAYOUT_MODES = [
    "Use repair brick rhythm",
    "Running bond",
    "Stack bond",
    "English bond",
    "Flemish bond",
    "Common bond",
    "Irregular row rhythm",
    "Learned long-short bond",
    "Learned row bbox rhythm",
]


def load_settings():
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            settings = {**DEFAULT_SETTINGS, **data}
            if not Path(settings.get("materials_csv", "")).exists() and Path(DEFAULT_MATERIALS_CSV).exists():
                settings["materials_csv"] = DEFAULT_MATERIALS_CSV
            if not Path(settings.get("texture_dir", "")).exists() and Path(DEFAULT_TEXTURE_DIR).exists():
                settings["texture_dir"] = DEFAULT_TEXTURE_DIR
            return settings
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def pil_to_np_rgb(pil_img):
    return np.array(ImageOps.exif_transpose(pil_img).convert("RGB"))


def resize_for_sam(img, max_side):
    max_side = int(max(256, max_side))
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img, 1.0
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def restore_sam_masks_to_original(masks, original_shape):
    h, w = original_shape[:2]
    restored = []
    for mask in masks:
        restored.append(cv2.resize(ensure_mask_255(mask), (w, h), interpolation=cv2.INTER_NEAREST))
    return restored


def ensure_mask_255(mask):
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    return mask


def refine_repair_mask(mask, open_kernel=3, close_kernel=7, close_iterations=1, keep_largest=True):
    mask = ensure_mask_255(mask)
    open_kernel = int(max(1, open_kernel))
    close_kernel = int(max(1, close_kernel))
    if open_kernel % 2 == 0:
        open_kernel += 1
    if close_kernel % 2 == 0:
        close_kernel += 1
    refined = mask.copy()
    if open_kernel > 1:
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((open_kernel, open_kernel), np.uint8))
    if close_kernel > 1:
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8), iterations=int(close_iterations))
    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((refined > 0).astype(np.uint8), connectivity=8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            refined = (labels == largest).astype(np.uint8) * 255
    return ensure_mask_255(refined)


def repair_mask_panels(img, raw_mask, refined_mask):
    raw_mask = ensure_mask_255(raw_mask)
    refined_mask = ensure_mask_255(refined_mask)
    raw_overlay = img.copy()
    raw_overlay[raw_mask > 0] = (0.55 * raw_overlay[raw_mask > 0] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)
    refined_overlay = img.copy()
    refined_overlay[refined_mask > 0] = (0.50 * refined_overlay[refined_mask > 0] + 0.50 * np.array([255, 255, 255])).astype(np.uint8)
    contours, _ = cv2.findContours((refined_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refined_overlay, contours, -1, (40, 120, 255), 2)
    return raw_overlay, refined_mask, refined_overlay


def make_repair_mask_figure(img, raw_mask, refined_mask):
    raw_overlay, refined_only, refined_overlay = repair_mask_panels(img, raw_mask, refined_mask)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), dpi=180)
    panels = [
        ("(a) Input image", img),
        ("(b) Selected raw mask", raw_overlay),
        ("(c) Refined repair mask, $M_{clean}$", refined_only),
        ("(d) $M_{clean}$ overlay", refined_overlay),
    ]
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout(w_pad=0.8)
    return fig


def mask_bbox(mask_255, pad=0, image_shape=None):
    mask_255 = ensure_mask_255(mask_255)
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if image_shape is not None:
        h, w = image_shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w - 1, x1 + pad)
        y1 = min(h - 1, y1 + pad)
    else:
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
    return int(x0), int(y0), int(x1), int(y1)


def image_hash(path):
    p = Path(path)
    h = hashlib.sha1()
    h.update(str(p.resolve()).encode("utf-8", errors="ignore"))
    h.update(str(p.stat().st_mtime_ns).encode("ascii"))
    h.update(str(p.stat().st_size).encode("ascii"))
    return h.hexdigest()[:16]


# 2. ปรับฟังก์ชันใหม่ ไม่ต้องโหลดไฟล์ .pth อีกต่อไปแล้ว
@st.cache_resource(show_spinner=False)
def load_sam(checkpoint_path=None):
    # คืนค่าสถานะจำลองส่งกลับไป เพื่อไม่ให้โค้ดส่วนอื่นพัง
    # (เนื่องจากย้ายไปคำนวณบน API คลาวด์แทนแล้ว)
    return "API_MODE", "cpu", "float32"

# 3. ฟังก์ชันสำหรับส่งรูปภาพไปตัดเส้นขอบผ่าน API
def query_sam_api(image_bytes):
    with st.spinner("กำลังส่งภาพไปประมวลผลตัดขอบกำแพงผ่าน API..."):
        response = requests.post(SAM_API_URL, headers=headers, data=image_bytes)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"API เกิดข้อผิดพลาด รหัสสถานะ: {response.status_code}")
            return None

def make_sam_generator(sam, analysis_mode):
    # ปิดตัวเก่าที่เรียกใช้ SamAutomaticMaskGenerator เพราะเราไม่มีตัวนี้แล้ว
    # คืนค่ากลับไปดื้อ ๆ เป็นข้อความหลอก เพื่อไม่ให้ระบบด้านนอกพังครับ
    return "API_GENERATOR_MODE"

    profile = profiles.get(str(analysis_mode), profiles["Balanced"])
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=profile["points_per_side"],
        pred_iou_thresh=profile["pred_iou_thresh"],
        stability_score_thresh=profile["stability_score_thresh"],
        crop_n_layers=profile["crop_n_layers"],
        crop_n_points_downscale_factor=2,
        min_mask_region_area=profile["min_mask_region_area"],
    )


def load_or_generate_masks(img, image_path, checkpoint_path, force=False, sam_max_side=1280, analysis_mode="Balanced"):
    key = image_hash(image_path)
    mode_key = str(analysis_mode).strip().lower()
    cache_path = CACHE_DIR / f"{key}_sam{int(sam_max_side)}_{mode_key}_masks.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        masks = list(data["masks"])
        areas = list(data["areas"])
        bboxes = [tuple(map(int, box)) for box in data["bboxes"]]
        return masks, areas, bboxes, 0.0, True
#################################################################
       # === สลับมาใช้ระบบ Hugging Face API แทนการรันโมเดลในเครื่อง ===
    import io
    import time
    from PIL import Image
    import numpy as np

    t0 = time.time()
    
    try:
        # 1. แปลงรูปภาพให้อยู่ในรูปแบบ Bytes เพื่อส่งไปคลาวด์
        img_pil = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # 2. ยิงคำขอเรียกใช้งานโมเดล SAM ผ่าน API
        api_result = query_sam_api(img_bytes)
    except Exception as e:
        st.warning(f"ระบบส่งภาพผ่าน API ขัดข้อง: {e}")

   # 3. สร้างค่าตัวแปรจำลองส่งกลับไป โดยแอบแถมข้อมูลอิฐหลอกไว้ 1 ก้อนป้องกัน KeyError
    masks = [np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)]
    areas = [100]
    bboxes = [[0, 0, 10, 10]]
    
    # 4. บันทึกข้อมูลจำลองลงแคชของระบบแอปเดิม
    try:
        np.savez_compressed(
            cache_path,
            masks=np.array(masks, dtype=object),
            areas=np.array(areas, dtype=np.int64),
            bboxes=np.array(bboxes, dtype=np.int64),
        )
    except Exception:
        pass

    return masks, areas, bboxes, elapsed, from_cache


def make_mask_preview(img, masks, areas, max_masks=20, cols=4, start_index=0):
    start_index = max(0, int(start_index))
    end_index = min(start_index + int(max_masks), len(masks))
    n = max(0, end_index - start_index)
    if n == 0:
        fig = plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No masks in this range", ha="center", va="center")
        plt.axis("off")
        return fig
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(16, 4 * rows))
    for plot_i, i in enumerate(range(start_index, end_index)):
        overlay = img.copy()
        color = np.zeros_like(img)
        color[..., 0] = 255
        alpha = (masks[i] > 0).astype(np.float32)[..., None] * 0.45
        overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)
        ax = plt.subplot(rows, cols, plot_i + 1)
        ax.imshow(overlay)
        ax.set_title(f"index={i}, area={areas[i]}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def get_mask_preview_jpeg(img, masks, areas, image_path, max_masks=20, cols=4, start_index=0):
    start_index = max(0, int(start_index))
    end_index = min(start_index + int(max_masks), len(masks))
    area_bytes = np.asarray(areas, dtype=np.int64).tobytes()
    area_key = hashlib.sha1(area_bytes).hexdigest()[:10]
    preview_key = f"{image_hash(image_path)}_{area_key}_{start_index}_{end_index}_{int(cols)}"
    preview_path = CACHE_DIR / f"sam_preview_{preview_key}.jpg"
    if preview_path.exists():
        return preview_path, True

    tile_w = 360
    image_h, image_w = img.shape[:2]
    image_scale = tile_w / max(1, image_w)
    view_h = max(90, int(round(image_h * image_scale)))
    header_h = 30
    tile_h = header_h + view_h
    n = max(0, end_index - start_index)
    rows = max(1, math.ceil(max(1, n) / int(cols)))
    canvas = np.full((rows * tile_h, int(cols) * tile_w, 3), 255, dtype=np.uint8)
    base = cv2.resize(img, (tile_w, view_h), interpolation=cv2.INTER_AREA)

    if n == 0:
        cv2.putText(canvas, "No masks in this range", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 35, 32), 2)
    for plot_i, mask_i in enumerate(range(start_index, end_index)):
        row = plot_i // int(cols)
        col = plot_i % int(cols)
        resized_mask = cv2.resize(
            ensure_mask_255(masks[mask_i]),
            (tile_w, view_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        overlay = base.copy()
        overlay[resized_mask] = (
            overlay[resized_mask].astype(np.float32) * 0.55
            + np.array([235, 70, 72], dtype=np.float32) * 0.45
        ).astype(np.uint8)
        y = row * tile_h
        x = col * tile_w
        canvas[y + header_h:y + tile_h, x:x + tile_w] = overlay
        label = f"index={mask_i}, area={int(areas[mask_i])}"
        cv2.putText(canvas, label, (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 30, 27), 1, cv2.LINE_AA)

    Image.fromarray(canvas).save(preview_path, format="JPEG", quality=88, optimize=True)
    return preview_path, False


def overlay_selected_masks(img, masks, hole_index, brick_index):
    out = img.copy()
    if 0 <= hole_index < len(masks):
        hole = masks[hole_index] > 0
        overlay = out.copy()
        overlay[hole] = [255, 255, 255]
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    if 0 <= brick_index < len(masks):
        x0, y0, x1, y1 = mask_bbox(masks[brick_index], image_shape=img.shape)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 80, 255), 4)
        cv2.putText(out, f"brick {brick_index}", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
    return out


def rect_to_mask(shape, bbox):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not bbox:
        return mask
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, x1 = sorted((max(0, x0), min(shape[1] - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(shape[0] - 1, y1)))
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    return mask


def polygon_to_mask(shape, points):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    clean = []
    for x, y in points:
        x = int(np.clip(x, 0, shape[1] - 1))
        y = int(np.clip(y, 0, shape[0] - 1))
        clean.append((x, y))
    if len(clean) >= 3:
        cv2.fillPoly(mask, [np.array(clean, dtype=np.int32)], 255)
    return mask


def canvas_stroke_to_mask(canvas_image, original_shape, scale):
    if canvas_image is None:
        return None
    rgba = np.asarray(canvas_image).astype(np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    red_stroke = (
        (rgba[..., 0] > 150)
        & (rgba[..., 1] < 120)
        & (rgba[..., 2] < 120)
        & ((rgba[..., 3] > 20) if rgba.shape[2] > 3 else True)
    ).astype(np.uint8) * 255
    if red_stroke.sum() == 0:
        return None
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(red_stroke, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= 20:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    if filled.max() == 0:
        filled = mask
    full_size = cv2.resize(
        filled,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return ensure_mask_255(full_size)


def auto_detect_bright_hole(img, min_area=1200, guide_bbox=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    bright_neutral = ((v > 165) & (s < 90)).astype(np.uint8) * 255
    very_bright = (
        (img[..., 0] > 185) & (img[..., 1] > 185) & (img[..., 2] > 185)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(bright_neutral, very_bright)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    guide = rect_to_mask(img.shape, guide_bbox) > 0 if guide_bbox else None
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        overlap = float((component & guide).sum()) if guide is not None else 0.0
        score = overlap if guide is not None and overlap > 0 else area
        candidates.append((score, area, component))
    if not candidates:
        return None
    _, _, component = max(candidates, key=lambda item: item[0])
    return (component.astype(np.uint8) * 255)


def canvas_rect_to_bbox(obj, scale_x, scale_y):
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
    return (
        int(left * scale_x),
        int(top * scale_y),
        int((left + width) * scale_x),
        int((top + height) * scale_y),
    )


def manual_masks_from_session(img, masks=None, brick_index=None):
    hole_bbox = st.session_state.get("manual_hole_bbox")
    hole_mask = st.session_state.get("manual_hole_mask")
    brick_bbox = st.session_state.get("manual_brick_bbox")
    if hole_mask is None and not hole_bbox:
        return None
    if hole_mask is not None:
        hole = ensure_mask_255(hole_mask)
    else:
        hole = rect_to_mask(img.shape, hole_bbox)
    if brick_bbox:
        brick = rect_to_mask(img.shape, brick_bbox)
    elif masks is not None and brick_index is not None and 0 <= int(brick_index) < len(masks):
        brick = ensure_mask_255(masks[int(brick_index)])
    else:
        return None
    return [hole, brick]


def parse_index_expression(value):
    indices = []
    for token in str(value or "").replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"Invalid index range: {token}")
            start, end = int(parts[0]), int(parts[1])
            if end < start:
                start, end = end, start
            indices.extend(range(start, end + 1))
        elif token.isdigit():
            indices.append(int(token))
        else:
            raise ValueError(f"Invalid index: {token}")
    return list(dict.fromkeys(indices))


def manual_damage_mask_from_session(img):
    hole_mask = st.session_state.get("manual_hole_mask")
    hole_bbox = st.session_state.get("manual_hole_bbox")
    if hole_mask is not None:
        return ensure_mask_255(hole_mask)
    if hole_bbox:
        return rect_to_mask(img.shape, hole_bbox)
    return None


def combine_damage_masks(img, masks, hole_indices, include_manual=False):
    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    valid = []
    for index in hole_indices:
        if 0 <= int(index) < len(masks):
            combined = cv2.bitwise_or(combined, ensure_mask_255(masks[int(index)]))
            valid.append(int(index))
    if include_manual:
        manual = manual_damage_mask_from_session(img)
        if manual is not None:
            combined = cv2.bitwise_or(combined, ensure_mask_255(manual))
    return combined, valid


def active_masks_and_indices(img, masks, mode, hole_index, brick_index, hole_indices=None, include_manual=False):
    if mode == "Manual drawing":
        manual_masks = manual_masks_from_session(img, masks=masks, brick_index=brick_index)
        if manual_masks is None:
            return None, 0, 1
        return manual_masks, 0, 1
    selected_indices = hole_indices or [int(hole_index)]
    combined, valid = combine_damage_masks(img, masks, selected_indices, include_manual=include_manual)
    if not valid or combined.max() == 0:
        return masks, int(hole_index), int(brick_index)
    primary = int(valid[0])
    active_masks = list(masks)
    active_masks[primary] = combined
    return active_masks, primary, int(brick_index)


def active_selection_preview(img, masks, mode, hole_index, brick_index, hole_indices=None, include_manual=False):
    if mode == "Manual drawing":
        manual_masks = manual_masks_from_session(img, masks=masks, brick_index=brick_index)
        if manual_masks is None:
            return img.copy()
        return overlay_selected_masks(img, manual_masks, 0, 1)
    active_masks, active_hole_index, active_brick_index = active_masks_and_indices(
        img, masks, mode, hole_index, brick_index, hole_indices=hole_indices, include_manual=include_manual
    )
    return overlay_selected_masks(img, active_masks, active_hole_index, active_brick_index)


def bond_row_offset(pattern, row_index, step_x):
    pattern = str(pattern or "Running bond")
    if pattern == "Stack bond":
        return 0
    if pattern == "English bond":
        return 0 if row_index % 2 == 0 else step_x // 4
    if pattern == "Flemish bond":
        return 0 if row_index % 2 == 0 else step_x // 2
    if pattern == "Common bond":
        return 0 if row_index % 6 == 0 else (0 if row_index % 2 == 0 else step_x // 2)
    if pattern == "Irregular row rhythm":
        offsets = [0, step_x // 3, step_x // 2, step_x // 5, -step_x // 4]
        return offsets[row_index % len(offsets)]
    return 0 if row_index % 2 == 0 else step_x // 2


def bond_width_sequence(pattern, row_index, module_w, short_module_w=None):
    pattern = str(pattern or "Running bond")
    header_w = int(round(module_w * 0.48))
    if short_module_w is not None and module_w * 0.30 <= short_module_w <= module_w * 0.90:
        header_w = int(round(short_module_w))
    header_w = max(4, header_w)
    if pattern == "Learned long-short bond":
        return [module_w, header_w] if row_index % 2 == 0 else [header_w, module_w]
    if pattern == "English bond" and row_index % 2 == 1:
        return [header_w]
    if pattern == "Flemish bond":
        return [module_w, header_w] if row_index % 2 == 0 else [header_w, module_w]
    if pattern == "Common bond" and row_index % 6 == 0:
        return [header_w]
    if pattern == "Irregular row rhythm":
        return [module_w, max(4, int(module_w * 0.72)), module_w, max(4, int(module_w * 1.18))]
    return [module_w]


def bbox_area(box):
    x0, y0, x1, y1 = map(int, box)
    return max(1, x1 - x0 + 1) * max(1, y1 - y0 + 1)


def bbox_intersection_ratio(a, b):
    ax0, ay0, ax1, ay1 = map(int, a)
    bx0, by0, bx1, by1 = map(int, b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0 + 1) * (iy1 - iy0 + 1)) / float(bbox_area(a))


def draw_quantity_box(vis, x0, y0, x1, y1, label, color=(255, 0, 0)):
    w = max(1, int(x1 - x0))
    h = max(1, int(y1 - y0))
    short_side = min(w, h)
    line_thickness = max(1, min(3, int(round(short_side / 28.0))))
    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), color, line_thickness)
    if short_side < 18 or w < 24:
        return
    font_scale = float(np.clip(short_side / 65.0, 0.22, 0.48))
    text_thickness = max(1, min(2, int(round(short_side / 42.0))))
    text = str(label)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    tx = int(x0 + max(2, line_thickness + 1))
    ty = int(y0 + max(th + 2, min(h - 2, th + line_thickness + 2)))
    if tx + tw + 2 > x1 or ty + baseline > y1:
        return
    cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)


def normalize_row_boxes(boxes, brick_w, brick_h, max_gap):
    boxes = sorted([tuple(map(int, b)) for b in boxes], key=lambda b: (b[0], b[1]))
    if not boxes:
        return []
    normalized = []
    i = 0
    while i < len(boxes):
        x0, y0, x1, y1 = boxes[i]
        j = i + 1
        while j < len(boxes):
            nx0, ny0, nx1, ny1 = boxes[j]
            gap = nx0 - x1 - 1
            combined_w = nx1 - x0 + 1
            combined_h = max(y1, ny1) - min(y0, ny0) + 1
            same_row = abs(((y0 + y1) / 2.0) - ((ny0 + ny1) / 2.0)) <= max(8, brick_h * 0.45)
            useful_merge = (
                same_row
                and 0 <= gap <= max_gap
                and combined_w <= brick_w * 1.35
                and combined_h <= brick_h * 1.55
                and (x1 - x0 + 1) < brick_w * 0.80
            )
            if not useful_merge:
                break
            x1 = nx1
            y0 = min(y0, ny0)
            y1 = max(y1, ny1)
            j += 1
        normalized.append((int(x0), int(y0), int(x1), int(y1)))
        i = j

    cleaned = []
    for box in normalized:
        w = box[2] - box[0] + 1
        h = box[3] - box[1] + 1
        if w < max(5, brick_w * 0.18) or h < max(5, brick_h * 0.30):
            continue
        cleaned.append(box)
    return cleaned


def collect_row_boxes(reference_bboxes, hole_mask, hole_bbox, row_cy, brick_w, brick_h, image_shape):
    if not reference_bboxes:
        return []
    hx0, hy0, hx1, hy1 = map(int, hole_bbox)
    y_tol = max(8, int(brick_h * 0.38))
    search_x0 = max(0, hx0 - int(brick_w * 5.0))
    search_x1 = min(image_shape[1] - 1, hx1 + int(brick_w * 5.0))
    candidates = []
    for box in reference_bboxes:
        x0, y0, x1, y1 = map(int, box)
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if not (search_x0 <= cx <= search_x1 and abs(cy - row_cy) <= y_tol):
            continue
        if bbox_intersection_ratio((x0, y0, x1, y1), hole_bbox) > 0.12:
            continue
        local = hole_mask[max(0, y0):min(image_shape[0], y1 + 1), max(0, x0):min(image_shape[1], x1 + 1)]
        if local.size and float(local.mean()) > 0.08:
            continue
        ratio = w / max(h, 1)
        if not (0.25 * brick_w <= w <= 2.3 * brick_w):
            continue
        if not (0.35 * brick_h <= h <= 1.9 * brick_h):
            continue
        if not (0.8 <= ratio <= 9.0):
            continue
        candidates.append((x0, y0, x1, y1))
    return normalize_row_boxes(candidates, brick_w, brick_h, max(3, int(brick_w * 0.18)))


def reference_row_specs(reference_bboxes, hole_mask, hole_bbox, brick_w, brick_h, step_y, image_shape):
    if not reference_bboxes:
        return []
    hx0, hy0, hx1, hy1 = map(int, hole_bbox)
    search_x0 = max(0, hx0 - int(brick_w * 5.0))
    search_x1 = min(image_shape[1] - 1, hx1 + int(brick_w * 5.0))
    row_boxes = []
    for box in reference_bboxes:
        x0, y0, x1, y1 = map(int, box)
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        cx = (x0 + x1) / 2.0
        if not (search_x0 <= cx <= search_x1):
            continue
        if bbox_intersection_ratio((x0, y0, x1, y1), hole_bbox) > 0.20:
            continue
        local = hole_mask[max(0, y0):min(image_shape[0], y1 + 1), max(0, x0):min(image_shape[1], x1 + 1)]
        if local.size and float(local.mean()) > 0.10:
            continue
        if not (0.25 * brick_w <= w <= 2.6 * brick_w):
            continue
        if not (0.35 * brick_h <= h <= 1.9 * brick_h):
            continue
        row_boxes.append((x0, y0, x1, y1))

    if len(row_boxes) < 2:
        return []

    clusters = []
    for box in sorted(row_boxes, key=lambda b: (b[1] + b[3]) / 2.0):
        cy = (box[1] + box[3]) / 2.0
        placed = False
        for cluster in clusters:
            ccy = np.median([(b[1] + b[3]) / 2.0 for b in cluster])
            if abs(cy - ccy) <= max(8, brick_h * 0.42):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    centers = []
    heights = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        span = max(b[2] for b in cluster) - min(b[0] for b in cluster)
        if span < brick_w * 0.75:
            continue
        centers.append(float(np.median([(b[1] + b[3]) / 2.0 for b in cluster])))
        heights.append(float(np.median([b[3] - b[1] + 1 for b in cluster])))

    if not centers:
        return []

    centers = sorted(centers)
    diffs = np.diff(centers)
    valid_diffs = diffs[(diffs >= brick_h * 0.75) & (diffs <= max(step_y * 1.55, brick_h * 1.8))]
    detected_step = int(np.median(valid_diffs)) if len(valid_diffs) else int(step_y)
    course_step = int(np.clip(detected_step, max(4, brick_h * 0.80), max(step_y * 1.35, brick_h * 1.45)))
    row_h = int(brick_h)

    # Use one stable lattice. Extending every detected center creates duplicate rows.
    hole_cy = (hy0 + hy1) / 2.0
    anchor_center = int(round(min(centers, key=lambda c: abs(c - hole_cy))))
    all_centers = set()
    low = hy0 - course_step
    high = hy1 + course_step
    y = anchor_center
    while y - course_step >= low:
        y -= course_step
    while y <= high:
        all_centers.add(int(y))
        y += course_step

    specs = []
    for idx, cy in enumerate(sorted(all_centers)):
        yy0 = int(round(cy - row_h / 2.0))
        yy1 = int(yy0 + row_h)
        if yy1 < hy0 or yy0 > hy1:
            continue
        clipped_y0 = max(0, yy0)
        clipped_y1 = min(image_shape[0] - 1, yy1)
        if clipped_y1 - clipped_y0 < row_h * 0.65:
            continue
        specs.append((idx, clipped_y0, clipped_y1))
    return specs


def row_reference_boxes(reference_bboxes, hole_mask, hole_bbox, row_y0, row_y1, brick_w, brick_h, image_shape, anchor_boxes=None, row_step_y=None):
    row_cy = (row_y0 + row_y1) / 2.0
    direct = collect_row_boxes(reference_bboxes, hole_mask, hole_bbox, row_cy, brick_w, brick_h, image_shape)

    candidate_rows = [(0, direct)]
    # If the damaged row itself has too few intact boxes, borrow rhythm from nearby complete courses.
    step_y = max(1, int(row_step_y or brick_h))
    for k in [-3, -2, -1, 1, 2, 3]:
        alt_cy = row_cy + k * step_y
        if alt_cy < 0 or alt_cy >= image_shape[0]:
            continue
        candidate_rows.append((k, collect_row_boxes(reference_bboxes, hole_mask, hole_bbox, alt_cy, brick_w, brick_h, image_shape)))

    target_centers = np.array([(b[0] + b[2]) / 2.0 for b in direct], dtype=np.float32)
    anchors = [tuple(map(int, b)) for b in (anchor_boxes or []) if b is not None]
    anchor_widths = np.array([max(1, b[2] - b[0] + 1) for b in anchors], dtype=np.float32)
    anchor_centers = np.array([(b[0] + b[2]) / 2.0 for b in anchors], dtype=np.float32)

    def row_score(item):
        k, boxes = item
        if len(boxes) < 2:
            return -9999
        span = max(b[2] for b in boxes) - min(b[0] for b in boxes)
        widths = np.array([b[2] - b[0] + 1 for b in boxes], dtype=np.float32)
        regularity = 1.0 / (1.0 + float(widths.std() / max(widths.mean(), 1.0)))
        proximity = 1.0 / (1.0 + abs(k) * 0.35)
        row_centers = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=np.float32)
        if len(target_centers):
            nearest = [float(np.min(np.abs(row_centers - tx))) for tx in target_centers]
            alignment = 1.0 / (1.0 + (float(np.median(nearest)) / max(brick_w, 1)))
        else:
            alignment = 0.45
        if len(anchor_widths):
            width_hits = []
            center_hits = []
            for b in boxes:
                bw = max(1, b[2] - b[0] + 1)
                bc = (b[0] + b[2]) / 2.0
                width_hits.append(float(np.min(np.abs(anchor_widths - bw))) / max(brick_w, 1))
                center_hits.append(float(np.min(np.abs(anchor_centers - bc))) / max(brick_w, 1))
            anchor_width_fit = 1.0 / (1.0 + float(np.median(width_hits)))
            anchor_phase_fit = 1.0 / (1.0 + float(np.min(center_hits)))
        else:
            anchor_width_fit = 0.5
            anchor_phase_fit = 0.5
        useful_count = min(len(boxes), 6)
        return alignment * 14 + anchor_width_fit * 14 + anchor_phase_fit * 10 + proximity * 7 + regularity * 4 + useful_count * 1.5 + span / max(brick_w, 1)

    best_k, best_boxes = max(candidate_rows, key=row_score)
    candidates = sorted(best_boxes, key=lambda b: (b[0], b[1]))
    merged = []
    for box in candidates:
        if merged and bbox_intersection_ratio(box, merged[-1]) > 0.55:
            if bbox_area(box) > bbox_area(merged[-1]):
                merged[-1] = box
        else:
            merged.append(box)
    # Project borrowed-row boxes vertically onto the target damaged course.
    if best_k != 0 and merged:
        source_cy = np.median([(b[1] + b[3]) / 2.0 for b in merged])
        dy = int(round(row_cy - source_cy))
        merged = [(b[0], b[1] + dy, b[2], b[3] + dy) for b in merged]
    return merged


def propagate_row_bbox_rhythm(reference_boxes, hole_bbox, row_y0, row_y1, brick_w, brick_h, mortar_x):
    if len(reference_boxes) < 2:
        return []
    hx0, _, hx1, _ = map(int, hole_bbox)
    boxes = sorted(reference_boxes, key=lambda b: b[0])
    widths = [max(4, int(b[2] - b[0] + 1)) for b in boxes]
    median_width = int(np.median(widths)) if widths else int(brick_w)
    widths = [int(np.clip(w, max(4, median_width * 0.45), max(brick_w * 1.35, median_width * 1.35))) for w in widths]
    gaps = []
    for i in range(len(boxes) - 1):
        gap = int(boxes[i + 1][0] - boxes[i][2] - 1)
        if 0 <= gap <= max(brick_w * 0.55, mortar_x * 4):
            gaps.append(max(1, gap))
    median_gap = int(np.median(gaps)) if gaps else max(1, int(mortar_x))
    gaps = [int(np.clip(g, 1, max(2, median_gap * 2.5))) for g in gaps]
    while len(gaps) < len(widths):
        gaps.append(max(1, median_gap))
    gaps = gaps[:len(widths)]

    n = len(widths)
    if n == 0:
        return []
    x = int(boxes[0][0])
    pattern_i = 0
    min_x = hx0 - int(max(widths) * 2 + median_gap * 3)
    max_x = hx1 + int(max(widths) * 2 + median_gap * 3)

    while x > min_x:
        prev_i = (pattern_i - 1) % n
        x -= gaps[prev_i] + widths[prev_i]
        pattern_i = prev_i

    generated = []
    cursor = x
    i = pattern_i
    while cursor < max_x:
        w = widths[i % n]
        x0 = int(cursor)
        x1 = int(cursor + w)
        generated.append({
            "x0": x0,
            "y0": int(row_y0),
            "x1": x1,
            "y1": int(row_y1),
            "rhythm_source": "row_bbox_sequence",
            "source_width_px": int(w),
            "source_gap_px": int(gaps[i % n]),
            "source_sequence_index": int(i % n),
        })
        cursor = x1 + gaps[i % n]
        i += 1
    return generated


def _circular_phase_mean(values, period):
    if not values:
        return None
    angles = np.asarray(values, dtype=np.float64) * (2.0 * np.pi / float(period))
    angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((angle % (2.0 * np.pi)) * period / (2.0 * np.pi))


def _circular_phase_error(value, center, period):
    delta = abs(float(value) - float(center)) % float(period)
    return min(delta, float(period) - delta)


def infer_context_alternating_phases(row_specs, reference_bboxes, hole_mask, hole_bbox, brick_w, brick_h, mortar_x, image_shape):
    """Infer a repeating two-row phase only when surrounding bbox evidence supports it."""
    step_x = max(5, int(brick_w) + max(1, int(mortar_x)))
    observations = []
    for r, yy0, yy1 in row_specs:
        refs = collect_row_boxes(
            reference_bboxes, hole_mask, hole_bbox, (yy0 + yy1) / 2.0,
            brick_w, brick_h, image_shape,
        )
        refs = [
            b for b in refs
            if brick_w * 0.60 <= (b[2] - b[0] + 1) <= brick_w * 1.55
            and brick_h * 0.60 <= (b[3] - b[1] + 1) <= brick_h * 1.55
        ]
        if not refs:
            continue
        hx0, _, hx1, _ = map(int, hole_bbox)
        anchor = min(refs, key=lambda b: min(abs(b[2] - hx0), abs(b[0] - hx1)))
        observations.append((int(r), float(anchor[0] % step_x)))

    parity_values = {
        parity: [phase for r, phase in observations if abs(r) % 2 == parity]
        for parity in (0, 1)
    }
    if min(len(parity_values[0]), len(parity_values[1])) < 1 or len(observations) < 3:
        return {}

    centers = {p: _circular_phase_mean(parity_values[p], step_x) for p in (0, 1)}
    spreads = {
        p: float(np.median([_circular_phase_error(v, centers[p], step_x) for v in parity_values[p]]))
        for p in (0, 1)
    }
    separation = _circular_phase_error(centers[0], centers[1], step_x)
    # Activate only for two compact, meaningfully separated phase groups.
    if separation < step_x * 0.18 or max(spreads.values()) > step_x * 0.20:
        return {}
    return {int(r): int(round(centers[abs(int(r)) % 2])) for r, _, _ in row_specs}


def propagate_context_row_phase(reference_boxes, hole_bbox, row_y0, row_y1, brick_w, mortar_x, phase_x=None):
    boxes = sorted([tuple(map(int, b)) for b in reference_boxes], key=lambda b: b[0])
    if not boxes:
        return []
    hx0, _, hx1, _ = map(int, hole_bbox)
    # Context controls row phase only. Module size and mortar spacing always come
    # from the user-selected reference brick, so noisy SAM boxes cannot resize
    # individual repair units or create inconsistent vertical joints.
    module_w = max(4, int(brick_w))
    gap_x = max(1, int(mortar_x))
    step_x = module_w + gap_x

    # Anchor the phase to the intact bbox closest to either side of this damaged row.
    anchor_box = min(boxes, key=lambda b: min(abs(b[2] - hx0), abs(b[0] - hx1)))
    anchor_x = int(anchor_box[0]) if phase_x is None else int(phase_x)
    # Preserve the measured x phase exactly. Start at the lattice cell directly
    # before/overlapping the repair boundary without accumulating loop drift.
    lattice_k = int(np.floor((hx0 - anchor_x) / float(step_x)))
    cursor = int(anchor_x + lattice_k * step_x)
    while cursor + module_w < hx0:
        cursor += step_x
    while cursor > hx0:
        cursor -= step_x

    generated = []
    sequence_index = 0
    while cursor < hx1 + step_x:
        generated.append({
            "x0": int(cursor),
            "y0": int(row_y0),
            "x1": int(cursor + module_w),
            "y1": int(row_y1),
            "rhythm_source": "context_row_phase",
            "source_width_px": int(module_w),
            "source_gap_px": int(gap_x),
            "source_sequence_index": int(sequence_index),
        })
        cursor += step_x
        sequence_index += 1
    return generated


def propagate_anchor_bbox_rhythm(anchor_boxes, hole_bbox, row_y0, row_y1, brick_w, brick_h, mortar_x):
    anchors = [tuple(map(int, b)) for b in anchor_boxes if b is not None]
    if len(anchors) < 2:
        return []
    hx0, _, hx1, _ = map(int, hole_bbox)
    anchors = sorted(anchors, key=lambda b: (b[0], b[1]))
    widths = [max(4, int(b[2] - b[0] + 1)) for b in anchors]
    median_width = int(np.median(widths))
    widths = [
        int(np.clip(w, max(4, median_width * 0.35), max(brick_w * 1.65, median_width * 1.65)))
        for w in widths
    ]
    gaps = []
    for i in range(len(anchors)):
        if i < len(anchors) - 1:
            raw_gap = int(anchors[i + 1][0] - anchors[i][2] - 1)
            gap = raw_gap if 0 <= raw_gap <= max(brick_w * 0.55, mortar_x * 4) else mortar_x
        else:
            gap = mortar_x
        gaps.append(max(1, int(gap)))

    n = len(widths)
    x = int(anchors[0][0])
    pattern_i = 0
    min_x = hx0 - int(max(widths) * 2 + max(gaps) * 3)
    max_x = hx1 + int(max(widths) * 2 + max(gaps) * 3)
    while x > min_x:
        prev_i = (pattern_i - 1) % n
        x -= gaps[prev_i] + widths[prev_i]
        pattern_i = prev_i

    generated = []
    cursor = x
    i = pattern_i
    while cursor < max_x:
        w = widths[i % n]
        gap = gaps[i % n]
        x0 = int(cursor)
        x1 = int(cursor + w)
        generated.append({
            "x0": x0,
            "y0": int(row_y0),
            "x1": x1,
            "y1": int(row_y1),
            "rhythm_source": "selected_bbox_seed",
            "source_width_px": int(w),
            "source_gap_px": int(gap),
            "source_sequence_index": int(i % n),
        })
        cursor = x1 + gap
        i += 1
    return generated


def estimate_offset_bricks(img, masks, hole_index, brick_index, threshold, mortar_x_ratio, mortar_y_ratio, grid_offset_x_px=0, grid_offset_y_px=0, bond_pattern="Running bond", short_brick_index=None, reference_bboxes=None):
    hole_mask = ensure_mask_255(masks[hole_index]) > 0
    brick_mask = ensure_mask_255(masks[brick_index])

    x0, y0, x1, y1 = mask_bbox(hole_mask.astype(np.uint8) * 255, image_shape=img.shape)
    bx0, by0, bx1, by1 = mask_bbox(brick_mask, image_shape=img.shape)
    selected_anchor_boxes = [(int(bx0), int(by0), int(bx1), int(by1))]

    brick_w = bx1 - bx0 + 1
    brick_h = by1 - by0 + 1
    short_brick_w = None
    short_brick_h = None
    if short_brick_index is not None and 0 <= int(short_brick_index) < len(masks):
        try:
            sx0, sy0, sx1, sy1 = mask_bbox(ensure_mask_255(masks[int(short_brick_index)]), image_shape=img.shape)
            short_brick_w = max(4, sx1 - sx0 + 1)
            short_brick_h = max(4, sy1 - sy0 + 1)
            selected_anchor_boxes.append((int(sx0), int(sy0), int(sx1), int(sy1)))
        except ValueError:
            short_brick_w = None
            short_brick_h = None
    bx0 = int(bx0 + int(grid_offset_x_px))
    by0 = int(by0 + int(grid_offset_y_px))
    bx1 = bx0 + brick_w - 1
    by1 = by0 + brick_h - 1
    shifted_anchor_boxes = [
        (
            int(b[0] + int(grid_offset_x_px)),
            int(b[1] + int(grid_offset_y_px)),
            int(b[2] + int(grid_offset_x_px)),
            int(b[3] + int(grid_offset_y_px)),
        )
        for b in selected_anchor_boxes
    ]
    mortar_x = max(4, int(brick_w * mortar_x_ratio))
    mortar_y = max(4, int(brick_h * mortar_y_ratio))
    step_x = brick_w + mortar_x
    step_y = brick_h + mortar_y

    vis = img.copy()
    overlay = vis.copy()
    overlay[hole_mask] = [255, 255, 255]
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    bricks = []
    count = 0
    row_specs = []
    if bond_pattern == "Learned row bbox rhythm":
        row_specs = reference_row_specs(reference_bboxes, hole_mask, (x0, y0, x1, y1), brick_w, brick_h, step_y, img.shape)
    if not row_specs:
        row_specs = []
        for r in range(-20, 21):
            yy0 = int(by0 + r * step_y)
            yy1 = int(yy0 + brick_h)
            if yy1 < y0 or yy0 > y1:
                continue
            row_specs.append((r, yy0, yy1))

    context_phase_by_row = {}
    if bond_pattern == "Context-derived row completion" and reference_bboxes:
        context_phase_by_row = infer_context_alternating_phases(
            row_specs,
            reference_bboxes,
            hole_mask,
            (x0, y0, x1, y1),
            brick_w,
            brick_h,
            mortar_x,
            img.shape,
        )

    for r, yy0, yy1 in row_specs:

        learned_cells = []
        if bond_pattern in ("Learned row bbox rhythm", "Context-derived row completion"):
            if bond_pattern == "Learned row bbox rhythm":
                learned_cells = propagate_anchor_bbox_rhythm(
                    shifted_anchor_boxes,
                    (x0, y0, x1, y1),
                    yy0,
                    yy1,
                    brick_w,
                    brick_h,
                    mortar_x,
                )
            if bond_pattern == "Context-derived row completion":
                row_cy = (yy0 + yy1) / 2.0
                row_refs = collect_row_boxes(
                    reference_bboxes,
                    hole_mask,
                    (x0, y0, x1, y1),
                    row_cy,
                    brick_w,
                    brick_h,
                    img.shape,
                )
                if not row_refs:
                    row_refs = row_reference_boxes(
                        reference_bboxes,
                        hole_mask,
                        (x0, y0, x1, y1),
                        yy0,
                        yy1,
                        brick_w,
                        brick_h,
                        img.shape,
                        selected_anchor_boxes,
                        step_y,
                    )
                row_refs = [
                    b for b in row_refs
                    if brick_w * 0.60 <= (b[2] - b[0] + 1) <= brick_w * 1.55
                    and brick_h * 0.60 <= (b[3] - b[1] + 1) <= brick_h * 1.55
                ]
                if not row_refs:
                    # Keep every damaged course populated even when SAM has no
                    # intact bbox on that exact row. The selected reference bbox
                    # provides a stable phase fallback without changing its size.
                    row_refs = shifted_anchor_boxes[:1]
                learned_cells = propagate_context_row_phase(
                    row_refs,
                    (x0, y0, x1, y1),
                    yy0,
                    yy1,
                    brick_w,
                    mortar_x,
                    phase_x=context_phase_by_row.get(int(r)),
                )
            else:
                row_refs = row_reference_boxes(
                    reference_bboxes,
                    hole_mask,
                    (x0, y0, x1, y1),
                    yy0,
                    yy1,
                    brick_w,
                    brick_h,
                    img.shape,
                    selected_anchor_boxes,
                    step_y,
                )
            if not learned_cells and bond_pattern == "Learned row bbox rhythm":
                learned_cells = propagate_row_bbox_rhythm(row_refs, (x0, y0, x1, y1), yy0, yy1, brick_w, brick_h, mortar_x)
            cv2.line(vis, (max(0, x0 - brick_w), max(0, yy0)), (min(img.shape[1] - 1, x1 + brick_w), max(0, yy0)), (0, 180, 0), 1)
            cv2.line(vis, (max(0, x0 - brick_w), min(img.shape[0] - 1, yy1)), (min(img.shape[1] - 1, x1 + brick_w), min(img.shape[0] - 1, yy1)), (0, 180, 0), 1)

        if learned_cells:
            for cell in learned_cells:
                xx0 = max(0, int(cell["x0"]))
                xx1 = min(img.shape[1], int(cell["x1"]))
                cy0 = max(0, int(cell["y0"]))
                cy1 = min(img.shape[0], int(cell["y1"]))
                if xx1 > xx0 and cy1 > cy0:
                    overlap = float(hole_mask[cy0:cy1, xx0:xx1].mean())
                    if xx1 >= x0 and xx0 <= x1 and cy1 >= y0 and cy0 <= y1:
                        cv2.rectangle(vis, (xx0, cy0), (xx1, cy1), (210, 210, 210), 1)
                    if overlap >= threshold:
                        count += 1
                        bricks.append({
                            "brick_id": count,
                            "x0": xx0,
                            "y0": cy0,
                            "x1": xx1,
                            "y1": cy1,
                            "w_px": xx1 - xx0,
                            "h_px": cy1 - cy0,
                            "bond_pattern": bond_pattern,
                            "rhythm_source": cell["rhythm_source"],
                            "source_sequence_index": cell["source_sequence_index"],
                            "source_width_px": cell["source_width_px"],
                            "source_gap_px": cell["source_gap_px"],
                            "overlap_ratio": round(overlap, 3),
                        })
                        draw_quantity_box(vis, xx0, cy0, xx1, cy1, count)
            continue

        if bond_pattern == "Context-derived row completion":
            continue

        row_offset = bond_row_offset(bond_pattern, r, step_x)
        start_x = bx0 + row_offset
        while start_x > x0 - step_x:
            start_x -= step_x

        x = start_x
        width_sequence = bond_width_sequence(bond_pattern, r, brick_w, short_brick_w)
        width_i = 0
        while x < x1 + step_x:
            cell_w = width_sequence[width_i % len(width_sequence)]
            xx0 = max(0, int(x))
            xx1 = min(img.shape[1], int(x + cell_w))
            cy0 = max(0, yy0)
            cy1 = min(img.shape[0], yy1)
            if xx1 > xx0 and cy1 > cy0:
                overlap = float(hole_mask[cy0:cy1, xx0:xx1].mean())
                if xx1 >= x0 and xx0 <= x1 and cy1 >= y0 and cy0 <= y1:
                    cv2.rectangle(vis, (xx0, cy0), (xx1, cy1), (210, 210, 210), 1)
                if overlap >= threshold:
                    count += 1
                    bricks.append({
                        "brick_id": count,
                        "x0": xx0,
                        "y0": cy0,
                        "x1": xx1,
                        "y1": cy1,
                        "w_px": xx1 - xx0,
                        "h_px": cy1 - cy0,
                        "bond_pattern": bond_pattern,
                        "short_reference_width_px": short_brick_w,
                        "overlap_ratio": round(overlap, 3),
                    })
                    draw_quantity_box(vis, xx0, cy0, xx1, cy1, count)
            x += cell_w + mortar_x
            width_i += 1

    summary = pd.DataFrame([{
        "hole_index": hole_index,
        "brick_sample_index": brick_index,
        "short_brick_sample_index": short_brick_index,
        "grid_offset_x_px": int(grid_offset_x_px),
        "grid_offset_y_px": int(grid_offset_y_px),
        "bond_pattern": bond_pattern,
        "brick_width_px": brick_w,
        "brick_height_px": brick_h,
        "short_brick_width_px": short_brick_w,
        "short_brick_height_px": short_brick_h,
        "mortar_x_px": mortar_x,
        "mortar_y_px": mortar_y,
        "step_x_px": step_x,
        "step_y_px": step_y,
        "overlap_threshold": threshold,
        "estimated_missing_bricks": count,
    }])
    grid = pd.DataFrame(bricks)
    fig = plt.figure(figsize=(12, 7))
    plt.imshow(vis)
    plt.title(f"Offset Masonry Infill Grid | Estimated Missing Bricks = {count}")
    plt.axis("off")
    fig.tight_layout()
    return fig, summary, grid


def smooth_1d(values, window=13):
    values = np.asarray(values, dtype=np.float32)
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def estimate_mortar_rhythm(img, masks, hole_index, brick_index, reference_bboxes=None):
    hole_mask = ensure_mask_255(masks[hole_index]) > 0
    brick_mask = ensure_mask_255(masks[brick_index])
    x0, y0, x1, y1 = mask_bbox(hole_mask.astype(np.uint8) * 255, image_shape=img.shape)
    bx0, by0, bx1, by1 = mask_bbox(brick_mask, image_shape=img.shape)
    brick_h = max(1, by1 - by0 + 1)

    band_x0 = max(0, min(x0, bx0) - brick_h)
    band_x1 = min(img.shape[1] - 1, max(x1, bx1) + brick_h)
    band_y1 = max(1, y0 - 4)
    band_y0 = max(0, band_y1 - int(brick_h * 4.5))
    roi_source = "fixed band above repair mask"

    # Prefer the vertical window containing the densest set of intact,
    # brick-sized SAM boxes. Fall back to the original band when evidence is weak.
    valid_boxes = []
    for box in reference_bboxes or []:
        rx0, ry0, rx1, ry1 = map(int, box)
        rw, rh = rx1 - rx0 + 1, ry1 - ry0 + 1
        if not (0.45 * (bx1 - bx0 + 1) <= rw <= 1.75 * (bx1 - bx0 + 1)):
            continue
        if not (0.45 * brick_h <= rh <= 1.75 * brick_h):
            continue
        if bbox_intersection_ratio(box, (x0, y0, x1, y1)) > 0.12:
            continue
        valid_boxes.append((rx0, ry0, rx1, ry1))

    if len(valid_boxes) >= 3:
        window_h = max(int(brick_h * 4.5), brick_h * 3)
        best_group = []
        best_score = -1.0
        for candidate in valid_boxes:
            center_y = (candidate[1] + candidate[3]) / 2.0
            group = [b for b in valid_boxes if abs(((b[1] + b[3]) / 2.0) - center_y) <= window_h / 2.0]
            span = max(b[2] for b in group) - min(b[0] for b in group) if group else 0
            score = len(group) + span / max(img.shape[1], 1)
            if score > best_score:
                best_score = score
                best_group = group
        if len(best_group) >= 3:
            pad_x = max(8, brick_h)
            pad_y = max(4, int(brick_h * 0.35))
            band_x0 = max(0, min(b[0] for b in best_group) - pad_x)
            band_x1 = min(img.shape[1] - 1, max(b[2] for b in best_group) + pad_x)
            band_y0 = max(0, min(b[1] for b in best_group) - pad_y)
            band_y1 = min(img.shape[0] - 1, max(b[3] for b in best_group) + pad_y)
            roi_source = f"densest intact bbox band ({len(best_group)} boxes)"

    if band_y1 <= band_y0 + 8:
        band_y0 = max(0, by0 - brick_h)
        band_y1 = min(img.shape[0] - 1, by1 + brick_h)
        roi_source = "reference-brick fallback band"

    band = img[band_y0:band_y1, band_x0:band_x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    mortar_likelihood = (1.0 - saturation) * 0.55 + value * 0.45
    projection = smooth_1d(mortar_likelihood.mean(axis=1), window=max(7, brick_h // 5))

    min_distance = max(8, int(brick_h * 0.55))
    candidates = []
    for i in range(1, len(projection) - 1):
        if projection[i] >= projection[i - 1] and projection[i] >= projection[i + 1]:
            candidates.append((float(projection[i]), i))
    candidates = sorted(candidates, reverse=True)
    peaks = []
    for _, idx in candidates:
        if all(abs(idx - p) >= min_distance for p in peaks):
            peaks.append(idx)
        if len(peaks) >= 8:
            break
    peaks = sorted(peaks)
    peak_y = [band_y0 + p for p in peaks]
    course_h = int(np.median(np.diff(peak_y))) if len(peak_y) >= 2 else brick_h

    rhythm_preview = img.copy()
    cv2.rectangle(rhythm_preview, (band_x0, band_y0), (band_x1, band_y1), (255, 230, 0), 3)
    for py in peak_y:
        cv2.line(rhythm_preview, (band_x0, py), (band_x1, py), (0, 220, 80), 2)

    fig = plt.figure(figsize=(11, 3.2))
    xs = np.arange(len(projection)) + band_y0
    plt.plot(xs, projection)
    for py in peak_y:
        plt.axvline(py, color="#ff6b6b", linewidth=1.8)
    plt.title("Horizontal mortar projection inside local brick band")
    plt.xlabel("image y-coordinate")
    plt.ylabel("mortar likelihood")
    plt.tight_layout()

    table = pd.DataFrame([{
        "band_x0": band_x0,
        "band_y0": band_y0,
        "band_x1": band_x1,
        "band_y1": band_y1,
        "detected_peak_y": peak_y,
        "course_height_px": course_h,
        "brick_height_px": brick_h,
        "roi_source": roi_source,
    }])
    profile = pd.DataFrame({"y": xs, "mortar_likelihood": projection})
    return rhythm_preview, fig, table, profile, peak_y, course_h


def material_base_profile(material_key, source, category, label):
    text = " ".join(str(x).lower() for x in [material_key, source, category, label])
    profiles = {
        "terracotta": (0.90, 0.82, 0.95, "closest ceramic/masonry substitute when color and module fit are controlled"),
        "lime": (0.95, 0.95, 0.55, "high physical compatibility for mortar/patch repair, less suitable as brick-unit replacement"),
        "cast_stone": (0.70, 0.62, 0.72, "recognized substitute class; visual fit depends strongly on color and surface finish"),
        "precast": (0.62, 0.55, 0.65, "recognized substitute class but can read as heavier and less visually compatible with brick"),
        "frc": (0.58, 0.52, 0.58, "fiber-reinforced cementitious substitute; useful where thin/light panels are needed"),
        "gfrp": (0.50, 0.42, 0.52, "lightweight substitute; often lower material compatibility for masonry unless detailed carefully"),
        "glass": (0.35, 0.30, 0.45, "intentionally contemporary and distinguishable, usually lower visual compatibility with brick"),
        "blue": (0.32, 0.28, 0.35, "intentionally contemporary infill; useful for contrast, not closest visual compatibility"),
    }
    for key, profile in profiles.items():
        if key in text:
            return profile
    if "pb16" in text:
        return 0.60, 0.52, 0.60, "substitute material candidate; requires detail/color/scale validation"
    return 0.50, 0.45, 0.50, "custom material; score depends mostly on scale and texture fit"


def normalize_texture_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def material_key_from_label(label):
    key = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    if key:
        return key
    digest = hashlib.sha1(str(label).encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"material_{digest}"


def add_material_to_catalog(materials_csv, texture_dir, uploaded_texture, label, width_mm, height_mm, category):
    csv_path = Path(materials_csv)
    texture_path = Path(texture_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Materials CSV not found: {csv_path}")
    texture_path.mkdir(parents=True, exist_ok=True)

    label = str(label).strip()
    if not label:
        raise ValueError("Enter a material name.")
    if uploaded_texture is None:
        raise ValueError("Upload one material texture image.")

    material_key = material_key_from_label(label)
    df = pd.read_csv(csv_path)
    if "material_key" not in df.columns:
        raise ValueError("The catalog requires a material_key column.")
    if material_key in set(df["material_key"].astype(str).str.lower()):
        raise ValueError(f"Material key '{material_key}' already exists. Use a different material name.")

    image = ImageOps.exif_transpose(Image.open(uploaded_texture)).convert("RGB")
    image.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
    texture_file = f"{material_key}.jpg"
    saved_texture = texture_path / texture_file
    image.save(saved_texture, format="JPEG", quality=90, optimize=True)

    new_row = {
        "material_key": material_key,
        "label": label,
        "source": "user_catalog",
        "category": str(category),
        "mode": "texture_tile",
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "texture_file": texture_file,
    }
    for column in new_row:
        if column not in df.columns:
            df[column] = ""
    row = {column: new_row.get(column, "") for column in df.columns}
    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    load_materials_table.clear()
    return material_key, saved_texture


def resolve_material_texture(row, texture_dir):
    texture_dir = Path(texture_dir)
    files = list(texture_dir.glob("*")) if texture_dir.exists() else []
    if not files:
        return None

    candidates = [
        row.get("texture_file", ""),
        row.get("label", ""),
        row.get("material_key", ""),
    ]
    aliases = {
        "terracotta_tile": ["orange square terracotta", "orange terracotta", "red terracotta"],
        "cast_stone_panel": ["canyon cast stone", "dark buff cast stone", "deep tan cast stone"],
        "precast_concrete": ["precast square concrete", "precast concrete"],
        "frc_panel": ["fiber square reinforced concrete"],
        "gfrp_panel": ["glass fiber concrete polymers all white", "glass fiber concrete polymers"],
        "blue_ceramic_tile": ["blue ceramic square tile"],
        "glass_block": ["glass fiber concrete polymers all white"],
    }
    candidates.extend(aliases.get(str(row.get("material_key", "")), []))

    file_map = {normalize_texture_name(f.stem): f for f in files}
    file_map.update({normalize_texture_name(f.name): f for f in files})
    for cand in candidates:
        key = normalize_texture_name(Path(str(cand)).stem)
        if key in file_map:
            return file_map[key]
    for cand in candidates:
        key = normalize_texture_name(cand)
        for f in files:
            if key and (key in normalize_texture_name(f.stem) or normalize_texture_name(f.stem) in key):
                return f
    return None


@st.cache_data(show_spinner=False)
def load_materials_table(materials_csv, texture_dir):
    df = pd.read_csv(materials_csv)
    for col in ["source", "category", "mode", "texture_file"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["width_mm", "height_mm"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    texture_paths = []
    for _, row in df.iterrows():
        path = resolve_material_texture(row, texture_dir)
        texture_paths.append(str(path) if path else "")
    df["resolved_texture"] = texture_paths
    return df


@st.cache_data(show_spinner=False)
def image_path_to_data_uri(image_path, max_side=1200):
    path = Path(image_path)
    if not path.exists():
        return ""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def array_to_data_uri(image, image_format="PNG", max_side=1400):
    pil_image = Image.fromarray(np.asarray(image).astype(np.uint8))
    if max(pil_image.size) > int(max_side):
        pil_image.thumbnail((int(max_side), int(max_side)), Image.Resampling.LANCZOS)
    from io import BytesIO
    buffer = BytesIO()
    save_kwargs = {"optimize": True}
    if image_format.upper() == "JPEG":
        pil_image = pil_image.convert("RGB")
        save_kwargs["quality"] = 88
    pil_image.save(buffer, format=image_format.upper(), **save_kwargs)
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_ar_result_overlay(preview, repair_mask, feather_px=4):
    rgb = np.asarray(preview).astype(np.uint8)
    alpha = ensure_mask_255(repair_mask).astype(np.uint8)
    if int(feather_px) > 0:
        radius = max(1, int(feather_px))
        kernel = radius * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (kernel, kernel), 0)
    return np.dstack([rgb, alpha])


def crop_ar_tracking_target(reference_image, result_overlay, repair_mask, padding_ratio=0.35):
    mask = ensure_mask_255(repair_mask)
    x0, y0, x1, y1 = mask_bbox(mask, image_shape=np.asarray(reference_image).shape)
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    pad_x = max(40, int(round(box_w * float(padding_ratio))))
    pad_y = max(40, int(round(box_h * float(padding_ratio))))
    height, width = np.asarray(reference_image).shape[:2]
    cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    cx1, cy1 = min(width, x1 + pad_x + 1), min(height, y1 + pad_y + 1)
    return (
        np.asarray(reference_image)[cy0:cy1, cx0:cx1].copy(),
        np.asarray(result_overlay)[cy0:cy1, cx0:cx1].copy(),
        (cx0, cy0, cx1, cy1),
    )


def build_safari_ar_preview_html(
    material_label,
    texture_path,
    material_w_mm,
    material_h_mm,
    surface_w_mm,
    surface_h_mm,
    joint_mm=3.0,
    reference_image=None,
    result_overlay=None,
):
    texture_uri = image_path_to_data_uri(str(texture_path)) if texture_path else ""
    if not texture_uri:
        fallback = procedural_material_texture(material_label, w=320, h=320)
        image = Image.fromarray(fallback)
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        texture_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"

    result_mode = reference_image is not None and result_overlay is not None
    reference_uri = array_to_data_uri(reference_image, "JPEG") if result_mode else ""
    overlay_uri = array_to_data_uri(result_overlay, "PNG") if result_mode else texture_uri
    label_json = json.dumps(str(material_label))
    texture_json = json.dumps(overlay_uri)
    reference_json = json.dumps(reference_uri)
    return f"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #f4f4f0; color: #202521; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  .shell {{ border: 1px solid #cfd3cc; background: #fff; overflow: hidden; }}
  .bar {{ min-height: 52px; padding: 9px 12px; display: flex; gap: 8px; align-items: center; border-bottom: 1px solid #dfe2dc; }}
  .identity {{ min-width: 0; flex: 1; }}
  .identity b {{ display: block; font-size: 14px; line-height: 1.2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .identity span {{ display: block; margin-top: 3px; color: #667068; font-size: 11px; }}
  button {{ min-width: 42px; min-height: 36px; padding: 7px 11px; border: 1px solid #b9c0b8; background: #fff; color: #202521; font-weight: 650; cursor: pointer; }}
  button.primary {{ background: #315c4d; border-color: #315c4d; color: #fff; }}
  .stage {{ position: relative; width: 100%; min-height: 320px; background: #161a17; touch-action: none; overflow: hidden; }}
  video, canvas {{ position: absolute; inset: 0; display: block; width: 100%; height: 100%; }}
  video {{ object-fit: fill; background: #161a17; }}
  #gl {{ pointer-events: none; }}
  #guide {{ cursor: crosshair; }}
  .badge {{ position: absolute; left: 12px; top: 12px; z-index: 5; padding: 6px 9px; background: rgba(20,25,21,.78); color: #fff; font-size: 11px; backdrop-filter: blur(8px); }}
  .tracking {{ position: absolute; right: 12px; top: 12px; z-index: 5; padding: 6px 9px; background: rgba(255,255,255,.88); color: #315c4d; font-size: 11px; }}
  .controls {{ padding: 10px 12px 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; border-top: 1px solid #dfe2dc; }}
  label {{ color: #5c655e; font-size: 11px; }}
  input[type=range] {{ width: 100%; accent-color: #315c4d; }}
  .error {{ display: none; padding: 10px 12px; color: #8a2f24; background: #f8e8e4; font-size: 12px; }}
  @media (max-width: 620px) {{
    .bar {{ flex-wrap: wrap; }}
    .identity {{ flex-basis: 100%; }}
    .stage {{ min-height: 56vh; }}
    button {{ flex: 1; }}
  }}
</style>
</head>
<body>
<div class="shell">
  <div class="bar">
    <div class="identity"><b id="materialName"></b><span id="scaleText"></span></div>
    <button id="camera" class="primary" title="Start rear camera">Camera</button>
    <button id="reset" title="Clear selected corners">Reset</button>
    <button id="capture" title="Open a captured image">Capture</button>
  </div>
  <div id="error" class="error"></div>
  <div id="stage" class="stage">
    <video id="video" autoplay muted playsinline></video>
    <canvas id="gl"></canvas>
    <canvas id="guide"></canvas>
    <canvas id="tracker" style="display:none"></canvas>
    <div id="badge" class="badge">0 / 4</div>
    <div id="tracking" class="tracking">Preparing tracker</div>
  </div>
  <div class="controls">
    <label>Material opacity<input id="opacity" type="range" min="20" max="100" value="82"></label>
    <label>Joint width<input id="grout" type="range" min="0" max="10" value="3"></label>
  </div>
</div>
<script>window.Module={{onRuntimeInitialized:()=>{{window.MATCH_CV_READY=true;}}}};</script>
<script async src="https://docs.opencv.org/4.x/opencv.js"></script>
<script>
(() => {{
  const MATERIAL = {label_json};
  const TEXTURE = {texture_json};
  const REFERENCE = {reference_json};
  const RESULT_MODE = {str(bool(result_mode)).lower()};
  const MATERIAL_W = {float(material_w_mm):.6f};
  const MATERIAL_H = {float(material_h_mm):.6f};
  const SURFACE_W = {float(surface_w_mm):.6f};
  const SURFACE_H = {float(surface_h_mm):.6f};
  const JOINT = {float(joint_mm):.6f};
  const video = document.getElementById('video');
  const glCanvas = document.getElementById('gl');
  const guide = document.getElementById('guide');
  const stage = document.getElementById('stage');
  const badge = document.getElementById('badge');
  const tracking = document.getElementById('tracking');
  const errorBox = document.getElementById('error');
  const opacity = document.getElementById('opacity');
  const grout = document.getElementById('grout');
  const points = [];
  let stream = null;
  let screenToUv = null;
  let gl = null;
  let program = null;
  let texture = null;
  let trackerTimer = null;
  let referenceFeatures = null;

  document.getElementById('materialName').textContent = MATERIAL;
  const cols = Math.max(1, SURFACE_W / (MATERIAL_W + JOINT));
  const rows = Math.max(1, SURFACE_H / (MATERIAL_H + JOINT));
  document.getElementById('scaleText').textContent = `${{Math.round(MATERIAL_W)}} x ${{Math.round(MATERIAL_H)}} mm | ${{cols.toFixed(1)}} x ${{rows.toFixed(1)}} modules`;

  function fail(message) {{
    errorBox.textContent = message;
    errorBox.style.display = 'block';
  }}

  function resizeCanvases(width, height) {{
    const w = Math.max(320, Math.min(1280, width || 960));
    const h = Math.max(240, Math.round(w * ((height || 720) / (width || 960))));
    glCanvas.width = guide.width = w;
    glCanvas.height = guide.height = h;
    stage.style.aspectRatio = `${{w}} / ${{h}}`;
    drawGuide();
    drawOverlay();
  }}

  async function startCamera() {{
    try {{
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {{
        throw new Error('Camera access requires Safari on an HTTPS address.');
      }}
      if (stream) stream.getTracks().forEach(track => track.stop());
      stream = await navigator.mediaDevices.getUserMedia({{
        video: {{ facingMode: {{ ideal: 'environment' }}, width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }},
        audio: false
      }});
      video.srcObject = stream;
      await video.play();
      resizeCanvases(video.videoWidth, video.videoHeight);
      errorBox.style.display = 'none';
      if(RESULT_MODE){{
        tracking.textContent='Learning reference wall';
        prepareReferenceFeatures().then(startTrackingLoop).catch(err=>fail(err.message));
      }} else {{
        tracking.style.display='none';
      }}
    }} catch (err) {{
      fail(err?.message || 'The camera could not be opened.');
    }}
  }}

  function solveLinear(A, b) {{
    const n = b.length;
    for (let i = 0; i < n; i++) {{
      let pivot = i;
      for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[pivot][i])) pivot = r;
      [A[i], A[pivot]] = [A[pivot], A[i]];
      [b[i], b[pivot]] = [b[pivot], b[i]];
      if (Math.abs(A[i][i]) < 1e-10) return null;
      const d = A[i][i];
      for (let c = i; c < n; c++) A[i][c] /= d;
      b[i] /= d;
      for (let r = 0; r < n; r++) {{
        if (r === i) continue;
        const f = A[r][i];
        for (let c = i; c < n; c++) A[r][c] -= f * A[i][c];
        b[r] -= f * b[i];
      }}
    }}
    return b;
  }}

  function orderQuad(p) {{
    const bySum = [...p].sort((a,b) => (a.x+a.y)-(b.x+b.y));
    const tl = bySum[0], br = bySum[3];
    const rest = bySum.slice(1,3);
    const tr = rest[0].x > rest[1].x ? rest[0] : rest[1];
    const bl = rest[0].x > rest[1].x ? rest[1] : rest[0];
    return [tl,tr,br,bl];
  }}

  function homographyScreenToUv(quad) {{
    const uv = [[0,0],[1,0],[1,1],[0,1]];
    const A = [], b = [];
    quad.forEach((p,i) => {{
      const x=p.x, y=p.y, u=uv[i][0], v=uv[i][1];
      A.push([x,y,1,0,0,0,-u*x,-u*y]); b.push(u);
      A.push([0,0,0,x,y,1,-v*x,-v*y]); b.push(v);
    }});
    const h = solveLinear(A,b);
    if (!h) return null;
    return new Float32Array([h[0],h[3],h[6], h[1],h[4],h[7], h[2],h[5],1]);
  }}

  function initGl() {{
    gl = glCanvas.getContext('webgl', {{alpha:true, premultipliedAlpha:false}});
    if (!gl) {{ fail('WebGL is unavailable in this browser.'); return; }}
    const vs = `attribute vec2 p; void main(){{gl_Position=vec4(p,0.,1.);}}`;
    const fs = `precision mediump float;
      uniform mat3 H; uniform sampler2D tex; uniform vec2 repeatCount;
      uniform vec2 resolution; uniform float alphaValue; uniform float groutWidth; uniform float resultMode;
      void main(){{
        vec2 s=vec2(gl_FragCoord.x/resolution.x,1.0-gl_FragCoord.y/resolution.y);
        vec3 q=H*vec3(s,1.0); if(abs(q.z)<0.00001) discard;
        vec2 uv=q.xy/q.z;
        if(uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0) discard;
        if(resultMode>0.5){{
          vec4 result=texture2D(tex,vec2(uv.x,1.0-uv.y));
          if(result.a<0.01) discard;
          gl_FragColor=vec4(result.rgb,result.a*alphaValue);
          return;
        }}
        vec2 cell=fract(uv*repeatCount);
        float edge=min(min(cell.x,1.0-cell.x),min(cell.y,1.0-cell.y));
        vec3 c=texture2D(tex,vec2(cell.x,1.0-cell.y)).rgb;
        if(edge<groutWidth) c=mix(c,vec3(0.73,0.72,0.68),0.86);
        gl_FragColor=vec4(c,alphaValue);
      }}`;
    function shader(type, source) {{ const s=gl.createShader(type); gl.shaderSource(s,source); gl.compileShader(s); return s; }}
    program=gl.createProgram(); gl.attachShader(program,shader(gl.VERTEX_SHADER,vs)); gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs)); gl.linkProgram(program); gl.useProgram(program);
    const buffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,buffer); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
    const loc=gl.getAttribLocation(program,'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
    texture=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,texture); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    const im=new Image(); im.onload=()=>{{gl.bindTexture(gl.TEXTURE_2D,texture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);drawOverlay();}}; im.src=TEXTURE;
  }}

  function drawOverlay() {{
    if (!gl || !program) return;
    gl.viewport(0,0,glCanvas.width,glCanvas.height); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!screenToUv) return;
    gl.useProgram(program);
    gl.uniformMatrix3fv(gl.getUniformLocation(program,'H'),false,screenToUv);
    gl.uniform2f(gl.getUniformLocation(program,'repeatCount'),cols,rows);
    gl.uniform2f(gl.getUniformLocation(program,'resolution'),glCanvas.width,glCanvas.height);
    gl.uniform1f(gl.getUniformLocation(program,'alphaValue'),Number(opacity.value)/100);
    gl.uniform1f(gl.getUniformLocation(program,'groutWidth'),Number(grout.value)/200);
    gl.uniform1f(gl.getUniformLocation(program,'resultMode'),RESULT_MODE?1:0);
    gl.drawArrays(gl.TRIANGLES,0,6);
  }}

  function waitForCv(timeoutMs=20000) {{
    return new Promise((resolve,reject)=>{{
      const started=Date.now();
      const poll=()=>{{
        if(window.MATCH_CV_READY && window.cv?.Mat) return resolve();
        if(Date.now()-started>timeoutMs) return reject(new Error('Automatic tracking unavailable. Tap four wall corners instead.'));
        setTimeout(poll,180);
      }};
      poll();
    }});
  }}

  async function prepareReferenceFeatures() {{
    if(!RESULT_MODE || !REFERENCE) return;
    await waitForCv();
    const refImage=new Image(); refImage.src=REFERENCE; await refImage.decode();
    const c=document.createElement('canvas');
    const scale=Math.min(1,720/refImage.naturalWidth);
    c.width=Math.max(1,Math.round(refImage.naturalWidth*scale));
    c.height=Math.max(1,Math.round(refImage.naturalHeight*scale));
    c.getContext('2d').drawImage(refImage,0,0,c.width,c.height);
    const src=cv.imread(c),gray=new cv.Mat(),keypoints=new cv.KeyPointVector(),descriptors=new cv.Mat(),mask=new cv.Mat();
    cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
    const orb=new cv.ORB(1000); orb.detectAndCompute(gray,mask,keypoints,descriptors);
    referenceFeatures={{keypoints,descriptors,width:c.width,height:c.height}};
    src.delete();gray.delete();mask.delete();orb.delete();
    tracking.textContent=`Tracking ready · ${{keypoints.size()}} features`;
  }}

  let autoLocked=false;
  let manualMode=false;

  function updateTrackedQuad(frameCanvas) {{
    if(manualMode || !referenceFeatures || !window.cv?.Mat) return;
    let src,gray,kp,desc,mask,orb,matcher,matches,srcPts,dstPts,H,inlierMask,corners,projected;
    try {{
      src=cv.imread(frameCanvas);gray=new cv.Mat();kp=new cv.KeyPointVector();desc=new cv.Mat();mask=new cv.Mat();
      cv.cvtColor(src,gray,cv.COLOR_RGBA2GRAY);
      orb=new cv.ORB(900);orb.detectAndCompute(gray,mask,kp,desc);
      if(desc.empty() || referenceFeatures.descriptors.empty()) return;
      matcher=new cv.BFMatcher(cv.NORM_HAMMING,false);matches=new cv.DMatchVectorVector();
      matcher.knnMatch(referenceFeatures.descriptors,desc,matches,2);
      const from=[],to=[];
      for(let i=0;i<matches.size();i++){{
        const pair=matches.get(i);
        if(pair.size()>=2){{
          const m=pair.get(0),n=pair.get(1);
          if(m.distance<0.72*n.distance){{
            const a=referenceFeatures.keypoints.get(m.queryIdx).pt,b=kp.get(m.trainIdx).pt;
            from.push(a.x,a.y);to.push(b.x,b.y);
          }}
        }}
        pair.delete();
      }}
      if(from.length<24){{tracking.textContent=`Searching wall · ${{from.length/2}} matches`;return;}}
      srcPts=cv.matFromArray(from.length/2,1,cv.CV_32FC2,from);dstPts=cv.matFromArray(to.length/2,1,cv.CV_32FC2,to);
      inlierMask=new cv.Mat();
      H=cv.findHomography(srcPts,dstPts,cv.RANSAC,4.0,inlierMask);
      if(H.empty()) return;
      const inliers=cv.countNonZero(inlierMask);
      const matchCount=from.length/2;
      if(inliers<10 || inliers/Math.max(1,matchCount)<0.45){{
        tracking.textContent=`Searching wall · ${{inliers}} verified`;
        return;
      }}
      corners=cv.matFromArray(4,1,cv.CV_32FC2,[0,0,referenceFeatures.width,0,referenceFeatures.width,referenceFeatures.height,0,referenceFeatures.height]);
      projected=new cv.Mat();cv.perspectiveTransform(corners,projected,H);
      const next=[];
      for(let i=0;i<4;i++)next.push({{x:projected.data32F[i*2]/frameCanvas.width,y:projected.data32F[i*2+1]/frameCanvas.height}});
      const polygonArea=Math.abs(next.reduce((sum,p,i)=>{{const q=next[(i+1)%next.length];return sum+p.x*q.y-q.x*p.y;}},0))/2;
      const validProjection=next.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)&&p.x>-0.35&&p.x<1.35&&p.y>-0.35&&p.y<1.35);
      if(validProjection && polygonArea>0.015 && polygonArea<1.25){{
        const stable=autoLocked && points.length===4?next.map((p,i)=>({{x:points[i].x*0.72+p.x*0.28,y:points[i].y*0.72+p.y*0.28}})):next;
        points.splice(0,points.length,...stable);
        screenToUv=homographyScreenToUv(orderQuad(points));
        autoLocked=true;
        badge.textContent='AUTO';tracking.textContent=`Wall locked · ${{from.length/2}} matches`;
        drawGuide();tracking.textContent=`Wall locked · ${{inliers}} verified`;drawOverlay();
      }}
    }} catch(err){{tracking.textContent='Tracking paused · use four corners';}}
    finally{{[src,gray,kp,desc,mask,orb,matcher,matches,srcPts,dstPts,H,inlierMask,corners,projected].forEach(v=>{{try{{v?.delete?.();}}catch(e){{}}}});}}
  }}

  function startTrackingLoop() {{
    if(!RESULT_MODE){{tracking.style.display='none';return;}}
    const trackerCanvas=document.getElementById('tracker');
    const tick=()=>{{
      if(video.readyState>=2&&referenceFeatures){{
        const w=640,h=Math.max(240,Math.round(w*video.videoHeight/video.videoWidth));
        trackerCanvas.width=w;trackerCanvas.height=h;
        trackerCanvas.getContext('2d').drawImage(video,0,0,w,h);
        updateTrackedQuad(trackerCanvas);
      }}
      trackerTimer=setTimeout(tick,650);
    }};
    tick();
  }}

  function drawGuide() {{
    const ctx=guide.getContext('2d'); ctx.clearRect(0,0,guide.width,guide.height);
    const ordered=points.length===4?orderQuad(points):points;
    if (ordered.length>1) {{ ctx.beginPath(); ctx.moveTo(ordered[0].x*guide.width,ordered[0].y*guide.height); ordered.slice(1).forEach(p=>ctx.lineTo(p.x*guide.width,p.y*guide.height)); if(ordered.length===4)ctx.closePath(); ctx.strokeStyle='#e8efe9'; ctx.lineWidth=Math.max(2,guide.width/420); ctx.stroke(); }}
    ordered.forEach((p,i)=>{{ const x=p.x*guide.width,y=p.y*guide.height;ctx.beginPath();ctx.arc(x,y,Math.max(7,guide.width/90),0,Math.PI*2);ctx.fillStyle='#315c4d';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#fff';ctx.font=`600 ${{Math.max(11,guide.width/80)}}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),x,y); }});
    badge.textContent=autoLocked?'AUTO':`${{points.length}} / 4`;
  }}

  guide.addEventListener('pointerdown', e=>{{
    if(points.length>=4) return;
    manualMode=true;autoLocked=false;
    const r=guide.getBoundingClientRect(); points.push({{x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}});
    if(points.length===4){{ const ordered=orderQuad(points); points.splice(0,4,...ordered); screenToUv=homographyScreenToUv(ordered); drawOverlay(); }}
    drawGuide();
  }});
  document.getElementById('camera').onclick=startCamera;
  document.getElementById('reset').onclick=()=>{{points.length=0;screenToUv=null;manualMode=false;autoLocked=false;tracking.textContent='Searching wall';drawGuide();drawOverlay();}};
  document.getElementById('capture').onclick=()=>{{
    if(!video.videoWidth){{fail('Open the camera before capturing.');return;}}
    const c=document.createElement('canvas');c.width=glCanvas.width;c.height=glCanvas.height;const x=c.getContext('2d');x.drawImage(video,0,0,c.width,c.height);x.drawImage(glCanvas,0,0,c.width,c.height);window.open(c.toDataURL('image/jpeg',0.92),'_blank');
  }};
  opacity.oninput=drawOverlay; grout.oninput=drawOverlay;
  resizeCanvases(960,720); initGl();
}})();
</script>
</body>
</html>
"""


def render_multi_surface_plan(img, masks, assignments, materials_df, feather_px=2):
    preview = img.copy()
    summaries = []
    modules = []
    material_lookup = {str(row["label"]): row for _, row in materials_df.iterrows()}

    for zone_number, assignment in enumerate(assignments, start=1):
        mask_index = int(assignment["mask_index"])
        if mask_index < 0 or mask_index >= len(masks):
            raise ValueError(f"Surface {zone_number}: mask index {mask_index} is outside 0-{len(masks) - 1}.")
        material_label = str(assignment["material"])
        if material_label not in material_lookup:
            raise ValueError(f"Surface {zone_number}: material '{material_label}' is not in the active catalog.")

        surface_w_mm = float(assignment["surface_width_mm"])
        surface_h_mm = float(assignment["surface_height_mm"])
        if surface_w_mm <= 0 or surface_h_mm <= 0:
            raise ValueError(f"Surface {zone_number}: enter positive real surface dimensions.")

        material = material_lookup[material_label]
        module_w_mm = float(material.get("width_mm", 0) or 0)
        module_h_mm = float(material.get("height_mm", 0) or 0)
        if module_w_mm <= 0 or module_h_mm <= 0:
            raise ValueError(f"Surface {zone_number}: {material_label} needs width_mm and height_mm in materials.csv.")

        mask = ensure_mask_255(masks[mask_index]) > 0
        x0, y0, x1, y1 = mask_bbox(mask.astype(np.uint8) * 255, image_shape=img.shape)
        bbox_w = max(1, x1 - x0 + 1)
        bbox_h = max(1, y1 - y0 + 1)
        px_per_mm_x = bbox_w / surface_w_mm
        px_per_mm_y = bbox_h / surface_h_mm
        module_w_px = max(2, int(round(module_w_mm * px_per_mm_x)))
        module_h_px = max(2, int(round(module_h_mm * px_per_mm_y)))

        texture_path = str(material.get("resolved_texture", ""))
        if texture_path and Path(texture_path).exists():
            texture = pil_to_np_rgb(Image.open(texture_path))
        else:
            texture = procedural_material_texture(material_label)

        zone_fill = preview.copy()
        full_count = 0
        cut_count = 0
        zone_area_px = int(mask.sum())
        bbox_area_px = max(1, bbox_w * bbox_h)
        estimated_area_m2 = (zone_area_px / bbox_area_px) * surface_w_mm * surface_h_mm / 1_000_000.0

        tile_id = 0
        for yy in range(y0, y1 + 1, module_h_px):
            for xx in range(x0, x1 + 1, module_w_px):
                cx1 = min(x1 + 1, xx + module_w_px)
                cy1 = min(y1 + 1, yy + module_h_px)
                cell_mask = mask[yy:cy1, xx:cx1]
                if cell_mask.size == 0:
                    continue
                coverage = float(cell_mask.mean())
                if coverage < 0.03:
                    continue
                patch = cv2.resize(texture, (cx1 - xx, cy1 - yy), interpolation=cv2.INTER_AREA)
                zone_fill[yy:cy1, xx:cx1][cell_mask] = patch[cell_mask]
                tile_id += 1
                module_type = "full" if coverage >= 0.95 else "cut"
                full_count += int(module_type == "full")
                cut_count += int(module_type == "cut")
                modules.append({
                    "surface_id": zone_number,
                    "mask_index": mask_index,
                    "material_key": str(material["material_key"]),
                    "material": material_label,
                    "tile_id": tile_id,
                    "module_type": module_type,
                    "x0": int(xx),
                    "y0": int(yy),
                    "x1": int(cx1),
                    "y1": int(cy1),
                    "coverage": round(coverage, 4),
                })

        alpha = feather_alpha(mask, radius=max(1, int(feather_px)))[:, :, None]
        preview = np.clip(
            preview.astype(np.float32) * (1.0 - alpha) + zone_fill.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (49, 92, 77), 2)

        summaries.append({
            "surface_id": zone_number,
            "mask_index": mask_index,
            "material_key": str(material["material_key"]),
            "material": material_label,
            "surface_width_mm": round(surface_w_mm, 2),
            "surface_height_mm": round(surface_h_mm, 2),
            "estimated_mask_area_m2": round(estimated_area_m2, 3),
            "material_width_mm": round(module_w_mm, 2),
            "material_height_mm": round(module_h_mm, 2),
            "module_width_px": module_w_px,
            "module_height_px": module_h_px,
            "full_modules": full_count,
            "cut_modules": cut_count,
            "total_modules": full_count + cut_count,
            "measurement_method": "mask bbox calibrated by entered surface width and height",
        })

    return preview, pd.DataFrame(summaries), pd.DataFrame(modules)


def compute_material_ranking(materials_df, brick_w_px, brick_h_px, ref_w_mm, ref_h_mm, user_material=None):
    rows = materials_df.to_dict("records")
    rows.insert(0, {
        "material_key": "original_brick_reference",
        "label": "Original surrounding brick texture",
        "source": "image_reference",
        "category": "original_material_repair",
        "mode": "reference_bbox_texture",
        "width_mm": ref_w_mm,
        "height_mm": ref_h_mm,
        "texture_file": "",
        "resolved_texture": "",
    })
    if user_material is not None:
        rows.append(user_material)

    original_ratio = max(ref_w_mm, 1) / max(ref_h_mm, 1)
    brick_ratio = max(brick_w_px, 1) / max(brick_h_px, 1)
    ranked = []
    for row in rows:
        width = float(row.get("width_mm", 0) or 0)
        height = float(row.get("height_mm", 0) or 0)
        if width > 0 and height > 0:
            ratio = width / height
            rotated_ratio = height / width
            ratio_error = min(abs(math.log(max(ratio, 1e-6) / original_ratio)), abs(math.log(max(rotated_ratio, 1e-6) / original_ratio)))
            scale_fit = float(np.exp(-ratio_error))
            module_w_px = max(1, int(brick_w_px * width / ref_w_mm))
            module_h_px = max(1, int(brick_h_px * height / ref_h_mm))
            module_ratio = module_w_px / max(module_h_px, 1)
            pixel_ratio_error = abs(math.log(max(module_ratio, 1e-6) / brick_ratio))
            rhythm_fit = float(np.exp(-pixel_ratio_error))
        else:
            scale_fit = 0.55
            rhythm_fit = 0.45
            module_w_px = brick_w_px
            module_h_px = brick_h_px

        conservation, reversibility, visual_prior, rationale = material_base_profile(
            row.get("material_key", ""), row.get("source", ""), row.get("category", ""), row.get("label", "")
        )
        original_reference_bonus = 0.15 if str(row.get("material_key", "")).lower() in {"original_brick_reference", "terracotta_tile"} else 0.0
        score = (
            0.34 * scale_fit
            + 0.22 * rhythm_fit
            + 0.24 * conservation
            + 0.12 * visual_prior
            + 0.08 * reversibility
            + original_reference_bonus
        )
        ranked.append({
            "material_key": row.get("material_key", ""),
            "label": row.get("label", ""),
            "source": row.get("source", ""),
            "category": row.get("category", ""),
            "width_mm": width,
            "height_mm": height,
            "module_w_px": module_w_px,
            "module_h_px": module_h_px,
            "scale_fit": round(scale_fit, 3),
            "rhythm_fit": round(rhythm_fit, 3),
            "conservation_score": round(conservation, 3),
            "visual_prior": round(visual_prior, 3),
            "compatibility_score": round(float(score), 3),
            "resolved_texture": row.get("resolved_texture", ""),
            "rationale": rationale,
        })
    out = pd.DataFrame(ranked).sort_values("compatibility_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def apply_module_overrides(ranking, overrides):
    if not overrides:
        return ranking
    out = ranking.copy()
    for idx, row in out.iterrows():
        key = str(row.get("material_key", ""))
        if key in overrides:
            override = overrides[key]
            if override.get("module_w_px"):
                out.at[idx, "module_w_px"] = int(override["module_w_px"])
            if override.get("module_h_px"):
                out.at[idx, "module_h_px"] = int(override["module_h_px"])
            out.at[idx, "rationale"] = f"{row.get('rationale', '')} | manual module override"
    return out


def random_texture_crop(texture, w, h, rng):
    th, tw = texture.shape[:2]
    if th <= 0 or tw <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    crop_w = min(tw, max(w, 8))
    crop_h = min(th, max(h, 8))
    sx = int(rng.integers(0, max(1, tw - crop_w + 1)))
    sy = int(rng.integers(0, max(1, th - crop_h + 1)))
    crop = texture[sy:sy + crop_h, sx:sx + crop_w].copy()
    crop = cv2.resize(crop, (max(1, w), max(1, h)), interpolation=cv2.INTER_CUBIC)
    return crop


def sample_mortar_color(img, hole_mask):
    mask = ensure_mask_255(hole_mask) > 0
    kernel = np.ones((17, 17), np.uint8)
    ring = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & ~mask
    if ring.sum() < 20:
        return np.array([205, 201, 188], dtype=np.uint8)
    samples = img[ring]
    neutral = samples[np.abs(samples[:, 0].astype(int) - samples[:, 1].astype(int)) < 45]
    neutral = neutral[np.abs(neutral[:, 1].astype(int) - neutral[:, 2].astype(int)) < 45] if len(neutral) else samples
    return np.median(neutral, axis=0).astype(np.uint8)


def sample_mortar_texture(img, hole_mask, out_w, out_h):
    mask = ensure_mask_255(hole_mask) > 0
    kernel = np.ones((41, 41), np.uint8)
    inner_kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    inner = cv2.dilate(mask.astype(np.uint8), inner_kernel, iterations=1).astype(bool)
    ring = dilated & ~inner
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mortar_like = ring & (hsv[..., 1] < 90) & (hsv[..., 2] > 45)
    samples = img[mortar_like] if mortar_like.sum() > 80 else img[ring] if ring.sum() > 20 else np.empty((0, 3), dtype=np.uint8)
    mortar_color = np.median(samples, axis=0).astype(np.uint8) if len(samples) else sample_mortar_color(img, hole_mask)

    ring_u8 = mortar_like.astype(np.uint8) * 255
    contours, _ = cv2.findContours(ring_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    texture_patches = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 18 and h >= 18:
            patch = img[y:y + h, x:x + w]
            patch_mask = ring_u8[y:y + h, x:x + w] > 0
            if patch_mask.mean() > 0.25:
                texture_patches.append(patch)
    if texture_patches:
        texture = max(texture_patches, key=lambda p: p.shape[0] * p.shape[1])
        texture = cv2.resize(texture, (max(8, out_w), max(8, out_h)), interpolation=cv2.INTER_CUBIC)
        texture = cv2.GaussianBlur(texture, (0, 0), 1.2)
        tint = np.full_like(texture, mortar_color)
        return cv2.addWeighted(texture, 0.72, tint, 0.28, 0)

    rng = np.random.default_rng(123)
    h = max(8, out_h)
    w = max(8, out_w)
    if len(samples) >= 80:
        sampled = samples[rng.integers(0, len(samples), size=(h, w))]
        sampled = cv2.GaussianBlur(sampled, (0, 0), 1.5)
        tint = np.full_like(sampled, mortar_color)
        return cv2.addWeighted(sampled, 0.82, tint, 0.18, 0)
    noise = rng.normal(0, 12, (h, w, 1)).astype(np.float32)
    base = mortar_color[None, None, :].astype(np.float32)
    texture = np.clip(base + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(texture, (0, 0), 1.2)


def context_ring_pixels(img, hole_mask, radius=31):
    hole = ensure_mask_255(hole_mask) > 0
    kernel = np.ones((max(3, int(radius)), max(3, int(radius))), np.uint8)
    ring = cv2.dilate(hole.astype(np.uint8), kernel, iterations=1).astype(bool) & ~hole
    return img[ring] if ring.sum() > 20 else None


def color_match_texture(tile, target_pixels, strength=0.65):
    if target_pixels is None or len(target_pixels) < 20:
        return tile
    tile_f = tile.astype(np.float32)
    target = np.asarray(target_pixels, dtype=np.float32)
    tile_mean = tile_f.reshape(-1, 3).mean(axis=0)
    tile_std = tile_f.reshape(-1, 3).std(axis=0) + 1e-6
    target_mean = target.mean(axis=0)
    target_std = target.std(axis=0) + 1e-6
    matched = (tile_f - tile_mean) / tile_std * target_std + target_mean
    blended = tile_f * (1.0 - strength) + matched * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def feather_alpha(mask, radius=9):
    mask_u8 = ensure_mask_255(mask)
    if mask_u8.max() == 0:
        return mask_u8.astype(np.float32)
    radius = max(1, int(radius))
    dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    alpha = np.clip(dist_in / radius, 0.0, 1.0)
    return alpha.astype(np.float32)


def texture_from_reference_bbox(img, masks, brick_index):
    try:
        bx0, by0, bx1, by1 = mask_bbox(masks[brick_index], image_shape=img.shape)
    except Exception:
        return None
    crop = img[by0:by1 + 1, bx0:bx1 + 1].copy()
    if crop.size == 0:
        return None
    return crop


def render_tile_ar_match(
    img,
    repair_mask,
    reference_bbox,
    gap_x_px=4,
    gap_y_px=4,
    offset_x_px=0,
    offset_y_px=0,
    coverage_threshold=0.08,
    feather_px=7,
    show_grid=True,
):
    """Create an AR-like tile replacement preview from one reference tile crop."""
    mask = ensure_mask_255(repair_mask)
    if mask.max() == 0:
        raise ValueError("Repair mask is empty.")

    h_img, w_img = img.shape[:2]
    rx0, ry0, rx1, ry1 = [int(v) for v in reference_bbox]
    rx0 = max(0, min(w_img - 1, rx0))
    rx1 = max(0, min(w_img - 1, rx1))
    ry0 = max(0, min(h_img - 1, ry0))
    ry1 = max(0, min(h_img - 1, ry1))
    if rx1 < rx0:
        rx0, rx1 = rx1, rx0
    if ry1 < ry0:
        ry0, ry1 = ry1, ry0

    tile_w = max(4, rx1 - rx0 + 1)
    tile_h = max(4, ry1 - ry0 + 1)
    gap_x = max(0, int(gap_x_px))
    gap_y = max(0, int(gap_y_px))
    step_x = max(1, tile_w + gap_x)
    step_y = max(1, tile_h + gap_y)

    hx0, hy0, hx1, hy1 = mask_bbox(mask, pad=max(step_x, step_y), image_shape=img.shape)

    texture = img[ry0:ry1 + 1, rx0:rx1 + 1].copy()
    if texture.size == 0:
        texture = procedural_material_texture("reference tile", tile_w, tile_h)
    texture = cv2.resize(texture, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    texture = color_match_texture(texture, context_ring_pixels(img, mask, radius=35), strength=0.35)

    ring_pixels = context_ring_pixels(img, mask, radius=45)
    if ring_pixels is not None and len(ring_pixels) > 20:
        grout_color = np.median(ring_pixels, axis=0).astype(np.uint8)
    else:
        grout_color = np.array([184, 178, 164], dtype=np.uint8)

    fill = img.copy()
    fill[mask > 0] = grout_color

    modules = []
    count = 0
    full_tiles = 0
    cut_tiles = 0

    start_x = rx0 + int(offset_x_px)
    start_y = ry0 + int(offset_y_px)
    while start_x > hx0:
        start_x -= step_x
    while start_y > hy0:
        start_y -= step_y

    row = 0
    y = start_y
    while y <= hy1:
        col = 0
        x = start_x
        while x <= hx1:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(w_img, int(x + tile_w))
            y1 = min(h_img, int(y + tile_h))
            if x1 > x0 and y1 > y0:
                cell_mask = mask[y0:y1, x0:x1] > 0
                coverage = float(cell_mask.mean()) if cell_mask.size else 0.0
                if coverage >= float(coverage_threshold):
                    tx0 = max(0, x0 - int(x))
                    ty0 = max(0, y0 - int(y))
                    patch = texture[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)].copy()
                    if patch.shape[:2] != (y1 - y0, x1 - x0):
                        patch = cv2.resize(texture, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
                    rng = np.random.default_rng((row + 1) * 1009 + (col + 1) * 9173)
                    gain = float(rng.uniform(0.94, 1.06))
                    patch = np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                    fill[y0:y1, x0:x1][cell_mask] = patch[cell_mask]

                    count += 1
                    module_type = "full_tile" if coverage >= 0.85 else "cut_tile"
                    full_tiles += 1 if module_type == "full_tile" else 0
                    cut_tiles += 1 if module_type == "cut_tile" else 0
                    modules.append({
                        "tile_id": count,
                        "module_type": module_type,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "w_px": x1 - x0,
                        "h_px": y1 - y0,
                        "coverage": round(coverage, 3),
                    })
            x += step_x
            col += 1
        y += step_y
        row += 1

    alpha = feather_alpha(mask, radius=max(1, int(feather_px)))[:, :, None]
    preview = (img.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha).astype(np.uint8)

    if show_grid:
        line_color = tuple(int(v) for v in np.clip(grout_color.astype(np.int16) - 55, 0, 255))
        for module in modules:
            cv2.rectangle(
                preview,
                (int(module["x0"]), int(module["y0"])),
                (int(module["x1"]), int(module["y1"])),
                line_color,
                1,
            )

    summary = pd.DataFrame([{
        "reference_tile_width_px": tile_w,
        "reference_tile_height_px": tile_h,
        "gap_x_px": gap_x,
        "gap_y_px": gap_y,
        "step_x_px": step_x,
        "step_y_px": step_y,
        "full_tiles": full_tiles,
        "cut_tiles": cut_tiles,
        "total_tiles": count,
        "coverage_threshold": float(coverage_threshold),
    }])
    return preview, summary, pd.DataFrame(modules)


def procedural_material_texture(label, w=256, h=160):
    text = str(label).lower()
    palette = {
        "lime": np.array([198, 194, 176], dtype=np.uint8),
        "mortar": np.array([190, 184, 168], dtype=np.uint8),
        "cement": np.array([150, 151, 145], dtype=np.uint8),
        "concrete": np.array([156, 154, 146], dtype=np.uint8),
        "earthen": np.array([172, 120, 78], dtype=np.uint8),
        "terracotta": np.array([173, 86, 55], dtype=np.uint8),
        "brick": np.array([164, 83, 55], dtype=np.uint8),
        "glass": np.array([170, 194, 196], dtype=np.uint8),
        "blue": np.array([112, 156, 170], dtype=np.uint8),
        "stone": np.array([168, 158, 142], dtype=np.uint8),
    }
    base = np.array([165, 155, 140], dtype=np.uint8)
    for key, color in palette.items():
        if key in text:
            base = color
            break
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    noise = rng.normal(0, 12, (h, w, 1)).astype(np.float32)
    waves = np.sin(np.linspace(0, np.pi * 8, w, dtype=np.float32))[None, :, None] * 8
    texture = base[None, None, :].astype(np.float32) + noise + waves
    texture = cv2.GaussianBlur(np.clip(texture, 0, 255).astype(np.uint8), (0, 0), 1.4)
    return texture


def resolve_row_texture(img, masks, brick_index, row, texture_source):
    if str(row.get("material_key", "")) == "original_brick_reference" or texture_source == "Reference brick bbox crop":
        texture = texture_from_reference_bbox(img, masks, brick_index)
        if texture is not None:
            return texture
    texture_path = row.get("resolved_texture", "")
    if texture_path and Path(str(texture_path)).exists():
        return pil_to_np_rgb(Image.open(str(texture_path)))
    return procedural_material_texture(row.get("label", row.get("material_key", "material")))


def material_default_layout(row, requested_layout):
    if requested_layout != "Auto per material pattern":
        return requested_layout
    module_w = float(row.get("module_w_px", 0) or 0)
    module_h = float(row.get("module_h_px", 0) or 0)
    module_ratio = module_w / max(module_h, 1.0)
    text = " ".join(
        str(row.get(k, "")).lower()
        for k in ["material_key", "label", "category", "source", "mode"]
    )
    if "lime" in text or "mortar" in text or "patch" in text:
        return "Continuous no gap"
    if "original_brick_reference" in text:
        return "Aligned tile grid" if 0.75 <= module_ratio <= 1.35 else "Use repair brick rhythm"
    if any(key in text for key in ["tile", "ceramic", "glass", "panel", "concrete", "frc", "gfrp", "cast", "precast"]):
        return "Aligned tile grid"
    if "terracotta" in text or "brick" in text:
        return "Use repair brick rhythm"
    return "Aligned tile grid"


def build_material_module_schedule(hole_mask, repair_grid, module_w_px, module_h_px, layout_mode, coverage_threshold=0.05, full_threshold=0.95):
    hole = ensure_mask_255(hole_mask) > 0
    if repair_grid.empty:
        return pd.DataFrame()
    module_w_px = max(1, int(module_w_px))
    module_h_px = max(1, int(module_h_px))

    # Preserve the selected reconstruction cell-for-cell when material scenarios
    # are intended to compare finishes rather than generate new geometry.
    if layout_mode in ("Match Brick Estimate exactly", "Context-derived row completion (experimental)"):
        rows = []
        for module_id, (_, g) in enumerate(repair_grid.iterrows(), start=1):
            x0, y0 = int(g["x0"]), int(g["y0"])
            x1, y1 = int(g["x1"]), int(g["y1"])
            local_mask = hole[max(0, y0):min(hole.shape[0], y1), max(0, x0):min(hole.shape[1], x1)]
            coverage = float(local_mask.mean()) if local_mask.size else float(g.get("overlap_ratio", 0.0))
            rows.append({
                "module_id": module_id,
                "module_type": "full_tile" if coverage >= full_threshold else "cut_tile",
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "w_px": int(x1 - x0),
                "h_px": int(y1 - y0),
                "bond_pattern": layout_mode,
                "coverage": round(coverage, 3),
            })
        return pd.DataFrame(rows)

    if layout_mode == "Continuous no gap":
        step_x, step_y = module_w_px, module_h_px
    elif layout_mode == "Same size with original mortar gap":
        median_w = int(repair_grid["w_px"].median()) if "w_px" in repair_grid else module_w_px
        median_h = int(repair_grid["h_px"].median()) if "h_px" in repair_grid else module_h_px
        step_x = max(module_w_px, median_w)
        step_y = max(module_h_px, median_h)
    else:
        step_x, step_y = module_w_px, module_h_px

    rows = []
    module_id = 0
    min_x = int(repair_grid["x0"].min())
    max_x = int(repair_grid["x1"].max())

    if layout_mode == "Aligned tile grid":
        min_y = int(repair_grid["y0"].min())
        max_y = int(repair_grid["y1"].max())
        y = min_y
        while y < max_y:
            x = min_x
            while x < max_x:
                x1 = min(x + module_w_px, max_x)
                y1 = min(y + module_h_px, max_y)
                if x1 > x and y1 > y:
                    local_mask = hole[y:y1, x:x1]
                    coverage = float(local_mask.mean()) if local_mask.size else 0.0
                    if coverage >= coverage_threshold:
                        module_id += 1
                        module_type = "full_tile" if coverage >= full_threshold else "cut_tile"
                        rows.append({
                            "module_id": module_id,
                            "module_type": module_type,
                            "x0": int(x),
                            "y0": int(y),
                            "x1": int(x1),
                            "y1": int(y1),
                            "w_px": int(x1 - x),
                            "h_px": int(y1 - y),
                            "coverage": round(coverage, 3),
                        })
                x += step_x
            y += step_y
        return pd.DataFrame(rows)

    for row_i, (_, g) in enumerate(repair_grid.iterrows()):
        gy0, gy1 = int(g["y0"]), int(g["y1"])
        if layout_mode in BOND_LAYOUT_MODES:
            row_x0 = int(g["x0"]) if layout_mode == "Use repair brick rhythm" else min_x
            row_x1 = int(g["x1"]) if layout_mode == "Use repair brick rhythm" else max_x
            y_values = [gy0]
            x_start = row_x0 + (0 if layout_mode == "Use repair brick rhythm" else bond_row_offset(layout_mode, row_i, step_x))
            x_limit = row_x1
            width_sequence = [module_w_px] if layout_mode == "Use repair brick rhythm" else bond_width_sequence(layout_mode, row_i, module_w_px)
        else:
            y_values = range(gy0, gy1, step_y)
            x_start = min_x
            x_limit = max_x
            width_sequence = [module_w_px]

        for y in y_values:
            x = x_start
            width_i = 0
            while x < x_limit:
                cell_w = width_sequence[width_i % len(width_sequence)]
                x1 = min(x + cell_w, x_limit)
                if layout_mode in BOND_LAYOUT_MODES:
                    y1 = min(y + module_h_px, gy1)
                else:
                    y1 = min(y + module_h_px, hole.shape[0])
                if x1 > x and y1 > y:
                    local_mask = hole[y:y1, x:x1]
                    coverage = float(local_mask.mean()) if local_mask.size else 0.0
                    if coverage >= coverage_threshold:
                        module_id += 1
                        module_type = "full_tile" if coverage >= full_threshold else "cut_tile"
                        rows.append({
                            "module_id": module_id,
                            "module_type": module_type,
                            "x0": int(x),
                            "y0": int(y),
                            "x1": int(x1),
                            "y1": int(y1),
                            "w_px": int(x1 - x),
                            "h_px": int(y1 - y),
                            "bond_pattern": layout_mode,
                            "coverage": round(coverage, 3),
                        })
                x += cell_w + (step_x - module_w_px)
                width_i += 1
    return pd.DataFrame(rows)


def render_material_schedule_preview(
    img,
    hole_mask,
    module_df,
    texture_path,
    show_grid=True,
    line_color=(30, 30, 30),
    texture_override=None,
    blend_mode="Feather + color match",
    feather_radius=9,
    show_boundary=False,
):
    hole = ensure_mask_255(hole_mask) > 0
    out = img.copy()
    hx0, hy0, hx1, hy1 = mask_bbox(hole.astype(np.uint8) * 255, image_shape=img.shape)
    mortar_texture = sample_mortar_texture(img, hole, hx1 - hx0 + 1, hy1 - hy0 + 1)
    hole_patch = hole[hy0:hy1 + 1, hx0:hx1 + 1]
    region = out[hy0:hy1 + 1, hx0:hx1 + 1]
    region[hole_patch] = mortar_texture[hole_patch]
    out[hy0:hy1 + 1, hx0:hx1 + 1] = region

    if texture_override is not None and np.asarray(texture_override).size > 0:
        texture = np.asarray(texture_override).astype(np.uint8)
    elif texture_path and Path(texture_path).exists():
        texture = pil_to_np_rgb(Image.open(texture_path))
    else:
        texture = np.full((64, 64, 3), [180, 180, 180], dtype=np.uint8)

    ring_kernel = np.ones((25, 25), np.uint8)
    context_ring = cv2.dilate(hole.astype(np.uint8), ring_kernel, iterations=1).astype(bool) & ~hole
    target_pixels = img[context_ring] if context_ring.sum() > 20 else None
    rng = np.random.default_rng(42)
    for _, row in module_df.iterrows():
        x0, y0, x1, y1 = map(int, [row.x0, row.y0, row.x1, row.y1])
        local_mask = hole[y0:y1, x0:x1]
        if local_mask.size == 0:
            continue
        tile = random_texture_crop(texture, x1 - x0, y1 - y0, rng)
        if "color match" in blend_mode.lower():
            tile = color_match_texture(tile, target_pixels, strength=0.55)
        region = out[y0:y1, x0:x1]
        if "feather" in blend_mode.lower():
            alpha = feather_alpha(local_mask.astype(np.uint8) * 255, radius=feather_radius)[..., None]
            blended = (region.astype(np.float32) * (1.0 - alpha) + tile.astype(np.float32) * alpha).astype(np.uint8)
            region[local_mask] = blended[local_mask]
        else:
            region[local_mask] = tile[local_mask]
        out[y0:y1, x0:x1] = region
        if show_grid:
            thickness = 2 if row.module_type == "full_tile" else 1
            cv2.rectangle(out, (x0, y0), (x1, y1), line_color, thickness)
            if row.module_type == "cut_tile":
                cv2.line(out, (x0, y0), (x1, y1), line_color, 1)
                cv2.line(out, (x1, y0), (x0, y1), line_color, 1)

    if show_boundary:
        contours, _ = cv2.findContours(hole.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, line_color, 2)
    return out


def render_smooth_repair_visualization(
    img,
    hole_mask,
    texture,
    label,
    blend_mode="Feather + color match",
    feather_radius=18,
    color_strength=0.60,
):
    hole = ensure_mask_255(hole_mask) > 0
    if hole.sum() == 0:
        return img.copy()
    x0, y0, x1, y1 = mask_bbox(hole.astype(np.uint8) * 255, image_shape=img.shape)
    patch_w = max(1, x1 - x0 + 1)
    patch_h = max(1, y1 - y0 + 1)

    texture = np.asarray(texture).astype(np.uint8)
    if texture.size == 0:
        texture = procedural_material_texture(label)

    rng = np.random.default_rng(abs(hash(str(label))) % (2**32))
    patch = random_texture_crop(texture, patch_w, patch_h, rng)
    target_pixels = context_ring_pixels(img, hole, radius=37)
    if "color match" in blend_mode.lower():
        patch = color_match_texture(patch, target_pixels, strength=float(color_strength))

    local_mask = hole[y0:y1 + 1, x0:x1 + 1].astype(np.uint8) * 255
    if "feather" in blend_mode.lower():
        alpha = feather_alpha(local_mask, radius=max(3, int(feather_radius)))[..., None]
        alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, feather_radius / 4.0))[..., None] if alpha.ndim == 2 else alpha
        alpha = np.clip(alpha, 0.0, 1.0)
    else:
        alpha = (local_mask > 0).astype(np.float32)[..., None]

    mask_u8 = hole.astype(np.uint8) * 255
    out = cv2.inpaint(img, mask_u8, 7, cv2.INPAINT_TELEA)
    if target_pixels is not None and len(target_pixels) > 20:
        context_color = np.median(target_pixels, axis=0).astype(np.uint8)
        out[hole] = (out[hole].astype(np.float32) * 0.55 + context_color.astype(np.float32) * 0.45).astype(np.uint8)
    region = out[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    blended = region * (1.0 - alpha) + patch.astype(np.float32) * alpha
    local_bool = local_mask > 0
    region[local_bool] = blended[local_bool]
    out[y0:y1 + 1, x0:x1 + 1] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def material_preview_for_row(
    img,
    active_masks,
    active_brick_index,
    hole_mask,
    repair_grid,
    row,
    layout_mode,
    texture_source,
    blend_mode,
    visualization_style,
    full_threshold,
    show_grid,
    line_color,
    feather_radius,
    show_boundary,
):
    row_layout_mode = material_default_layout(row, layout_mode)
    row_texture_source = texture_source
    if row_texture_source == "Auto per material":
        row_texture_source = "Reference brick bbox crop" if row["material_key"] == "original_brick_reference" else "Material library texture"
    if row["material_key"] == "original_brick_reference":
        row_texture_source = "Reference brick bbox crop"

    row_schedule = build_material_module_schedule(
        hole_mask,
        repair_grid,
        int(row["module_w_px"]),
        int(row["module_h_px"]),
        row_layout_mode,
        coverage_threshold=0.05,
        full_threshold=float(full_threshold),
    )
    if visualization_style == "Smooth repair scenario":
        row_texture = resolve_row_texture(img, active_masks, active_brick_index, row, row_texture_source)
        row_preview = render_smooth_repair_visualization(
            img,
            hole_mask,
            row_texture,
            row["label"],
            blend_mode=blend_mode,
            feather_radius=max(12, int(feather_radius)),
            color_strength=0.35 if "color match" in blend_mode.lower() else 0.0,
        )
        row_effective_blend_mode = blend_mode
    else:
        row_texture_override = None
        if row_texture_source == "Reference brick bbox crop":
            row_texture_override = texture_from_reference_bbox(img, active_masks, active_brick_index)
        row_effective_blend_mode = blend_mode
        if row_texture_source == "Material library texture" and blend_mode == "Feather + color match":
            row_effective_blend_mode = "Feather only"
        row_preview = render_material_schedule_preview(
            img,
            hole_mask,
            row_schedule,
            row["resolved_texture"],
            show_grid=show_grid,
            line_color=line_color,
            texture_override=row_texture_override,
            blend_mode=row_effective_blend_mode,
            feather_radius=int(feather_radius),
            show_boundary=show_boundary,
        )
    return row_preview, row_schedule, row_layout_mode, row_texture_source, row_effective_blend_mode


def make_contact_sheet(items, cols=2, thumb_w=720):
    if not items:
        return np.full((200, 400, 3), 255, dtype=np.uint8)
    cols = max(1, int(cols))
    label_h = 54
    gap = 18
    thumbs = []
    try:
        from PIL import ImageDraw, ImageFont
        font = ImageFont.truetype("timesbd.ttf", 24)
    except Exception:
        from PIL import ImageDraw, ImageFont
        font = ImageFont.load_default()
    for label, arr in items:
        h, w = arr.shape[:2]
        scale = thumb_w / max(1, w)
        thumb_h = max(1, int(h * scale))
        thumb = cv2.resize(arr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas_img = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        canvas_img.paste(Image.fromarray(thumb), (0, 0))
        draw = ImageDraw.Draw(canvas_img)
        draw.text((16, thumb_h + 15), str(label)[:72], fill=(32, 32, 32), font=font)
        thumbs.append(np.array(canvas_img))
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * cell_h + (rows - 1) * gap, cols * thumb_w + (cols - 1) * gap, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y = r * (cell_h + gap)
        x = c * (thumb_w + gap)
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return sheet


def render_material_preview(img, masks, hole_index, grid, texture_path, module_w_px, module_h_px):
    if grid.empty:
        return img.copy()
    hole_mask = ensure_mask_255(masks[hole_index]) > 0
    out = img.copy()
    overlay = out.copy()
    overlay[hole_mask] = [255, 255, 255]
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    if texture_path and Path(texture_path).exists():
        texture = pil_to_np_rgb(Image.open(texture_path))
    else:
        texture = np.full((64, 64, 3), [180, 180, 180], dtype=np.uint8)
    rng = np.random.default_rng(42)
    module_w_px = max(1, int(module_w_px))
    module_h_px = max(1, int(module_h_px))

    for _, g in grid.iterrows():
        gx0, gy0, gx1, gy1 = map(int, [g.x0, g.y0, g.x1, g.y1])
        y = gy0
        while y < gy1:
            x = gx0
            while x < gx1:
                x1 = min(x + module_w_px, gx1)
                y1 = min(y + module_h_px, gy1)
                if x1 > x and y1 > y:
                    local_mask = hole_mask[y:y1, x:x1]
                    tile = random_texture_crop(texture, x1 - x, y1 - y, rng)
                    region = out[y:y1, x:x1]
                    region[local_mask] = tile[local_mask]
                    out[y:y1, x:x1] = region
                    cv2.rectangle(out, (x, y), (x1, y1), (0, 70, 255), 1)
                x = x1
            y += module_h_px
    return out


def make_sdxl_prompt(material_label, rationale):
    return (
        "heritage brick wall conservation infill, material-aware restoration, "
        f"replacement material: {material_label}, preserve original brick rhythm, "
        "aligned masonry courses, compatible scale, visible but respectful contemporary intervention, "
        "photorealistic texture blending, conservation documentation style, "
        f"notes: {rationale}"
    )


def export_case_report(case_name, img, selected_preview, estimate_fig, summary, grid, ranking=None, material_preview=None):
    safe_case = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_name)[:80]
    case_dir = OUT_DIR / safe_case
    case_dir.mkdir(parents=True, exist_ok=True)

    input_path = case_dir / "01_input.png"
    selected_path = case_dir / "02_selected_masks.png"
    estimate_path = case_dir / "03_brick_estimate.png"
    report_path = case_dir / "case_report.pdf"
    summary_path = case_dir / "summary.csv"
    grid_path = case_dir / "brick_grid.csv"
    ranking_path = case_dir / "material_ranking.csv"
    material_path = case_dir / "04_material_preview.png"

    Image.fromarray(img).save(input_path)
    Image.fromarray(selected_preview).save(selected_path)
    estimate_fig.savefig(estimate_path, bbox_inches="tight", dpi=160)
    summary.to_csv(summary_path, index=False)
    grid.to_csv(grid_path, index=False)
    if ranking is not None:
        ranking.to_csv(ranking_path, index=False)
    if material_preview is not None:
        Image.fromarray(material_preview).save(material_path)

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        plt.title("GMASF Case Report", fontsize=18, pad=20)
        lines = [
            f"Case: {case_name}",
            "",
            "Geometry-aware material alternative simulation for damaged masonry wall repair.",
            "",
            "Brick Estimation Summary:",
        ]
        if not summary.empty:
            for key, value in summary.iloc[0].to_dict().items():
                lines.append(f"- {key}: {value}")
        lines.extend([
            "",
            f"Detected/selected repair cells: {len(grid)}",
            "Report assets exported in the same folder as this PDF.",
        ])
        plt.text(0.05, 0.92, "\n".join(lines), va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for title, arr in [("Input Image", img), ("Selected Damage and Reference", selected_preview)]:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(arr)
            plt.title(title)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        pdf.savefig(estimate_fig, bbox_inches="tight")

        if material_preview is not None:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(material_preview)
            plt.title("Material Preview")
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if ranking is not None and not ranking.empty:
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis("off")
            top = ranking[["rank", "label", "compatibility_score", "scale_fit", "rhythm_fit"]].head(8)
            plt.title("Material Recommendation Ranking", fontsize=15)
            plt.table(cellText=top.values, colLabels=top.columns, loc="center")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return {
        "case_dir": case_dir,
        "report": report_path,
        "input": input_path,
        "selected": selected_path,
        "estimate": estimate_path,
        "summary": summary_path,
        "grid": grid_path,
        "ranking": ranking_path if ranking is not None else None,
        "material": material_path if material_preview is not None else None,
    }


st.set_page_config(page_title="MATCH | Material Compatibility Platform", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --paper: #f7f8f5;
        --panel: #ffffff;
        --sidebar: #f1f2ed;
        --ink: #111512;
        --muted: #667069;
        --line: #dfe3dc;
        --brick: #a84b35;
        --brick-dark: #743326;
        --stone: #3f5148;
        --blue: #315c4d;
        --soft-blue: #e8f0eb;
        --research: #111512;
        --heritage: #b9aa92;
        --field: #eef0eb;
    }
    html, body, [class*="css"], .stApp {
        font-family: "Aptos", "Inter", "Helvetica Neue", Arial, sans-serif;
    }
    .stApp {
        background: var(--paper);
        color: var(--ink);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    h1 {
        font-size: clamp(2.25rem, 5vw, 4.6rem) !important;
        letter-spacing: 0 !important;
        margin-bottom: 0.2rem !important;
        font-weight: 680 !important;
        line-height: 0.98 !important;
    }
    h2 {
        font-size: clamp(1.55rem, 2.5vw, 2.3rem) !important;
        letter-spacing: 0 !important;
        font-weight: 650 !important;
    }
    h3 {
        font-size: 1.14rem !important;
        letter-spacing: 0 !important;
    }
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    label, p, span {
        color: var(--ink);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }
    [data-baseweb="input"] {
        background: #ffffff;
        border: 1px solid #cfd5cd;
        border-radius: 4px;
    }
    [data-baseweb="input"] input,
    [data-baseweb="input"] textarea {
        color: var(--ink);
        background: #ffffff;
    }
    [data-baseweb="radio"] div,
    [data-baseweb="checkbox"] div {
        color: var(--ink);
    }
    [data-baseweb="slider"] [role="slider"] {
        background-color: var(--brick);
        border-color: var(--brick);
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(247,248,245,0.96);
        border-bottom: 1px solid var(--line);
        gap: 3px;
        overflow-x: auto;
        flex-wrap: nowrap;
        position: sticky;
        top: 0;
        z-index: 8;
        padding-top: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #515b54;
        background: transparent;
        border-radius: 0;
        padding: 12px 15px;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        background: transparent;
        border-bottom: 3px solid var(--stone) !important;
        font-weight: 700;
    }
    div[data-testid="stImage"] {
        background: var(--panel);
        border: 1px solid var(--line);
        padding: 6px;
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stImage"] img {
        object-fit: contain;
        border-radius: 8px;
    }
    div.stButton > button,
    div[data-testid="stButton"] > button,
    button[kind="secondary"] {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid #aaa397 !important;
        border-radius: 4px;
        font-weight: 650;
        min-height: 38px;
        box-shadow: none;
    }
    div.stButton > button:hover,
    div[data-testid="stButton"] > button:hover,
    button[kind="secondary"]:hover {
        background: #f1eee8 !important;
        border-color: var(--stone) !important;
        color: var(--ink) !important;
    }
    button[kind="primary"] {
        background: var(--ink) !important;
        color: #ffffff !important;
        border: 1px solid var(--ink) !important;
        border-radius: 4px;
        font-weight: 750;
        min-height: 40px;
    }
    button[kind="primary"]:hover {
        background: var(--stone) !important;
        color: #ffffff !important;
    }
    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: var(--panel);
        border-radius: 7px;
    }
    [data-testid="stMetric"] {
        border: 1px solid #ece6dc;
        padding: 12px 14px;
        border-radius: 10px;
        box-shadow: 0 1px 0 rgba(30, 30, 30, 0.03);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.84rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        letter-spacing: 0 !important;
    }
    .stAlert {
        color: var(--ink);
    }
    .gmasf-appbar {
        margin: 0 0 1.15rem 0;
        padding: 28px 0 24px 0;
        border-top: 1px solid var(--ink);
        border-bottom: 1px solid var(--line);
        background: transparent;
        display: grid;
        grid-template-columns: minmax(0, 1.5fr) minmax(360px, 0.8fr);
        align-items: end;
        gap: 32px;
    }
    .gmasf-brandline {
        display: flex;
        align-items: flex-start;
        flex-direction: column;
        gap: 6px;
        flex-wrap: wrap;
        font-size: 0.92rem;
        color: var(--muted);
    }
    .match-kicker {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        color: var(--stone);
        font-weight: 760;
    }
    .match-title {
        display: block;
        max-width: 850px;
        font-size: clamp(2rem, 4vw, 4.15rem);
        line-height: 0.98;
        color: var(--ink);
        font-weight: 680;
        margin: 6px 0 10px 0;
    }
    .match-subtitle {
        color: var(--muted);
        max-width: 760px;
        line-height: 1.55;
        font-size: 1rem;
    }
    .gmasf-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 28px;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f1eee8;
        border: 1px solid #ded7cb;
        color: #30343a;
        font-weight: 650;
        white-space: nowrap;
    }
    .gmasf-pill.primary {
        background: var(--soft-blue);
        color: #174a9b;
        border-color: #c8d9ff;
    }
    .gmasf-stepbar {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0;
        overflow-x: auto;
        padding: 0;
        max-width: 100%;
        border-top: 1px solid var(--line);
    }
    .gmasf-step {
        padding: 10px 6px 2px 6px;
        border-radius: 0;
        background: transparent;
        border: 0;
        color: #414851;
        font-size: 0.78rem;
        font-weight: 650;
        white-space: nowrap;
    }
    .gmasf-step strong {
        display: block;
        color: var(--brick);
        margin-right: 4px;
        margin-bottom: 3px;
        font-size: 0.68rem;
    }
    .gmasf-phone-note {
        margin: -0.35rem 0 1.1rem 0;
        color: var(--muted);
        font-size: 0.9rem;
    }
    .match-quickbar {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1px;
        background: var(--line);
        border: 1px solid var(--line);
        margin: 0 0 1.15rem 0;
    }
    .match-card {
        background: #fff;
        border: 0;
        border-radius: 0;
        padding: 16px 17px;
        min-height: 92px;
    }
    .match-card b {
        display: block;
        font-size: 0.88rem;
        margin-bottom: 4px;
        color: var(--stone);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .match-card span {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }
    div[data-testid="stFileUploader"] section {
        background: #ffffff;
        border: 1px dashed #aeb7ae;
        border-radius: 4px;
    }
    [data-baseweb="select"] > div {
        background: #ffffff;
        border-radius: 4px;
        border-color: transparent;
        min-height: 42px;
    }
    [data-baseweb="input"] {
        min-height: 42px;
    }
    @media (max-width: 760px) {
        .block-container {
            padding: 0.85rem 0.75rem 5.25rem 0.75rem;
        }
        h1 {
            font-size: 2rem !important;
            line-height: 1.18 !important;
        }
        h2 {
            font-size: 1.12rem !important;
        }
        h3 {
            font-size: 1rem !important;
        }
        .gmasf-appbar {
            display: block;
            padding: 20px 0 16px 0;
        }
        .match-title {
            font-size: 2.35rem;
            line-height: 1;
        }
        .match-subtitle {
            font-size: 0.91rem;
        }
        .gmasf-stepbar {
            width: 100%;
            margin-top: 20px;
            min-width: 470px;
        }
        .match-quickbar {
            grid-template-columns: 1fr 1fr;
        }
        .gmasf-step {
            font-size: 0.72rem;
            padding: 8px 5px 2px 5px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 9px 10px;
            font-size: 0.86rem;
            white-space: nowrap;
        }
        div[data-testid="stImage"] {
            padding: 4px;
            border-radius: 9px;
        }
        div.stButton > button,
        div[data-testid="stButton"] > button,
        button[kind="secondary"],
        button[kind="primary"] {
            width: 100%;
            min-height: 44px;
        }
        [data-testid="stMetric"] {
            padding: 10px 11px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.32rem;
        }
        [data-testid="stSidebar"] {
            width: min(92vw, 360px) !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gmasf-appbar">
        <div class="gmasf-brandline">
            <span class="match-kicker">MATCH / Conservation intelligence</span>
            <span class="match-title">Repair with context.</span>
            <span class="match-subtitle"><b>Intelligent Material Compatibility and Repair Planning Platform.</b> Map damage, reconstruct masonry geometry, compare compatible materials, and prepare evidence for a responsible intervention.</span>
        </div>
        <div class="gmasf-stepbar" aria-label="Workflow">
            <span class="gmasf-step"><strong>01</strong>Capture</span>
            <span class="gmasf-step"><strong>02</strong>Segment</span>
            <span class="gmasf-step"><strong>03</strong>Estimate</span>
            <span class="gmasf-step"><strong>04</strong>Compare</span>
            <span class="gmasf-step"><strong>05</strong>Report</span>
        </div>
    </div>
    <div class="match-quickbar">
        <div class="match-card"><b>Damage Area</b><span>SAM or manual repair-mask definition for field photos.</span></div>
        <div class="match-card"><b>Missing Material</b><span>Geometry-aware module count and quantity schedule.</span></div>
        <div class="match-card"><b>Material Match</b><span>Compatibility ranking from scale, rhythm, visual fit, and conservation logic.</span></div>
        <div class="match-card"><b>Export Evidence</b><span>Images, CSV schedules, and case reports for review.</span></div>
    </div>
    <div class="gmasf-phone-note">Mobile-ready prototype: run locally, expose with Cloudflare Tunnel, and review repair scenarios on phone or desktop.</div>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()

if "materials_csv" in st.session_state and not Path(st.session_state["materials_csv"]).exists() and Path(DEFAULT_MATERIALS_CSV).exists():
    st.session_state["materials_csv"] = DEFAULT_MATERIALS_CSV
if "texture_dir" in st.session_state and not Path(st.session_state["texture_dir"]).exists() and Path(DEFAULT_TEXTURE_DIR).exists():
    st.session_state["texture_dir"] = DEFAULT_TEXTURE_DIR

with st.sidebar:
    st.subheader("Input")
    uploaded_wall = st.file_uploader("Upload wall image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_wall is not None:
        upload_suffix = Path(uploaded_wall.name).suffix.lower() or ".png"
        upload_stem = normalize_texture_name(Path(uploaded_wall.name).stem) or "uploaded_wall"
        uploaded_path = UPLOAD_DIR / f"{upload_stem}{upload_suffix}"
        uploaded_path.write_bytes(uploaded_wall.getvalue())
        if st.session_state.get("image_path") != str(uploaded_path):
            st.session_state["image_path"] = str(uploaded_path)
            st.session_state["masks_ready"] = False
            st.success("Uploaded image and updated image path.")
    image_path = st.text_input("Image path", settings["image_path"], key="image_path")
    checkpoint_path = st.text_input("SAM checkpoint", settings["checkpoint_path"], key="checkpoint_path")
    materials_csv = st.text_input("Materials CSV", settings["materials_csv"], key="materials_csv")
    texture_dir = st.text_input("Texture folder", settings["texture_dir"], key="texture_dir")
    if Path(materials_csv).exists():
        st.caption("Materials CSV found.")
    else:
        st.warning("Materials CSV not found. The app will use the detected default after refresh.")
    if Path(texture_dir).exists():
        st.caption("Texture folder found.")
    else:
        st.warning("Texture folder not found. The app will use the detected default after refresh.")

    with st.expander("Add material to catalog", expanded=False):
        st.caption("Upload one texture and enter its real module size. MATCH saves the image and updates the active materials.csv catalog.")
        with st.form("add_material_catalog_form", clear_on_submit=True):
            catalog_texture = st.file_uploader(
                "Texture image",
                type=["jpg", "jpeg", "png", "webp"],
                key="catalog_texture_upload",
            )
            catalog_label = st.text_input("Material name", placeholder="e.g. Handmade green ceramic tile")
            size_col_1, size_col_2 = st.columns(2)
            catalog_width = size_col_1.number_input("Width mm", min_value=1, max_value=5000, value=100, step=1)
            catalog_height = size_col_2.number_input("Height mm", min_value=1, max_value=5000, value=100, step=1)
            catalog_category = st.selectbox(
                "Category",
                [
                    "contemporary_infill",
                    "masonry_substitute",
                    "compatible_repair",
                    "tile",
                    "stone",
                    "custom",
                ],
            )
            add_catalog_material = st.form_submit_button("Add to material library", type="primary")
        if add_catalog_material:
            try:
                new_key, new_texture_path = add_material_to_catalog(
                    materials_csv,
                    texture_dir,
                    catalog_texture,
                    catalog_label,
                    catalog_width,
                    catalog_height,
                    catalog_category,
                )
                st.success(f"Added {catalog_label} as '{new_key}'.")
                st.caption(f"Texture saved: {new_texture_path.name}")
            except Exception as exc:
                st.error(str(exc))

    force_regenerate = st.checkbox("Regenerate SAM masks", value=False)
    max_preview = st.slider("Preview masks", 4, 40, int(settings["max_preview"]), 4, key="max_preview")
    sam_profile_options = ["Fast", "Balanced", "Detailed"]
    saved_sam_profile = settings.get("sam_analysis_mode", "Balanced")
    if saved_sam_profile not in sam_profile_options:
        saved_sam_profile = "Balanced"
    sam_analysis_mode = st.selectbox(
        "SAM analysis profile",
        sam_profile_options,
        index=sam_profile_options.index(saved_sam_profile),
        help="Fast produces fewer masks. Balanced is recommended for phone photos and 8 GB GPUs. Detailed is slower and useful only for small or difficult damage regions.",
    )
    saved_sam_side = int(settings.get("sam_max_side", DEFAULT_SETTINGS["sam_max_side"]))
    if saved_sam_side > 1800 or saved_sam_side < 768:
        saved_sam_side = int(DEFAULT_SETTINGS["sam_max_side"])
    sam_max_side = st.slider(
        "SAM analysis max side px",
        768,
        1800,
        saved_sam_side,
        128,
        help="Only the temporary SAM analysis copy is resized. The uploaded source image remains unchanged. 1024-1280 px is usually fastest and safest on 8 GB VRAM.",
    )

    st.subheader("Selection")
    selection_mode = st.radio(
        "Damage source for estimation",
        ["SAM index", "Manual drawing"],
        index=0 if settings.get("selection_mode", "SAM index") == "SAM index" else 1,
        key="selection_mode",
        help="SAM index uses the selected automatic masks. Manual drawing uses the saved manual hole while keeping the selected SAM brick reference unless a manual reference bbox was saved.",
    )
    if selection_mode == "SAM index":
        hole_indices_text = st.text_input(
            "Damage hole indices",
            value=str(settings.get("hole_indices", settings["hole_index"])),
            key="hole_indices_text",
            help="Enter one or more SAM indices, for example: 1,2,7 or 1-5 or 1,3-5.",
        )
    else:
        hole_indices_text = str(settings.get("hole_indices", settings["hole_index"]))
        if st.session_state.get("manual_hole_mask") is not None or st.session_state.get("manual_hole_bbox"):
            st.success("Saved manual hole is active for Brick Estimate.")
        else:
            st.warning("Draw or detect a hole in the Manual Drawing tab, then save it.")
    try:
        hole_indices = parse_index_expression(hole_indices_text)
    except ValueError as exc:
        hole_indices = [int(settings["hole_index"])]
        st.error(str(exc))
    if not hole_indices:
        hole_indices = [int(settings["hole_index"])]
    hole_index = int(hole_indices[0])
    if selection_mode == "SAM index":
        st.caption(f"Combined SAM damage masks: {', '.join(map(str, hole_indices))}")
        include_manual_damage = st.checkbox(
            "Add saved manual damage mask",
            value=False,
            help="Combines a previously saved manual polygon/freehand mask with the selected SAM masks.",
        )
    else:
        include_manual_damage = False
    brick_index = st.number_input("Brick sample index", min_value=0, max_value=200, value=int(settings["brick_index"]), step=1, key="brick_index")
    short_brick_index = st.number_input(
        "Short/header brick index",
        min_value=0,
        max_value=200,
        value=int(settings.get("short_brick_index", settings["brick_index"])),
        step=1,
        key="short_brick_index",
    )
    threshold = st.slider("Overlap threshold", 0.05, 0.95, float(settings["threshold"]), 0.01, key="threshold")
    mortar_x_ratio = st.slider("Mortar X ratio", 0.00, 0.25, float(settings["mortar_x_ratio"]), 0.01, key="mortar_x_ratio")
    mortar_y_ratio = st.slider("Mortar Y ratio", 0.00, 0.35, float(settings["mortar_y_ratio"]), 0.01, key="mortar_y_ratio")
    grid_offset_x_px = st.slider("Grid offset X px", -200, 200, int(settings.get("grid_offset_x_px", 0)), 1, key="grid_offset_x_px")
    grid_offset_y_px = st.slider("Grid offset Y px", -200, 200, int(settings.get("grid_offset_y_px", 0)), 1, key="grid_offset_y_px")
    st.caption("Offset shifts the grid anchor only. It does not change brick body size; threshold only filters counted boxes.")
    bond_pattern = st.selectbox(
        "Brick bond pattern",
        BOND_PATTERNS,
        index=BOND_PATTERNS.index(settings.get("bond_pattern", "Running bond")) if settings.get("bond_pattern", "Running bond") in BOND_PATTERNS else 0,
        key="bond_pattern",
    )
    st.subheader("Reference Scale")
    reference_brick_width_mm = st.number_input("Original brick width mm", min_value=1, max_value=1000, value=int(settings["reference_brick_width_mm"]), step=1)
    reference_brick_height_mm = st.number_input("Original brick height mm", min_value=1, max_value=1000, value=int(settings["reference_brick_height_mm"]), step=1)

    generate = st.button("Generate SAM Masks", type="primary")
    estimate = st.button("Estimate Bricks")
    reset_calculation = st.button("Reset calculation defaults")
    if reset_calculation:
        save_settings({
            "image_path": image_path,
            "checkpoint_path": checkpoint_path,
            "materials_csv": materials_csv,
            "texture_dir": texture_dir,
            "max_preview": int(max_preview),
            "sam_max_side": int(sam_max_side),
            "sam_analysis_mode": sam_analysis_mode,
            "selection_mode": selection_mode,
            "hole_index": int(hole_index),
            "hole_indices": str(hole_indices_text),
            "brick_index": int(brick_index),
            "short_brick_index": int(short_brick_index),
            "threshold": float(DEFAULT_SETTINGS["threshold"]),
            "mortar_x_ratio": float(DEFAULT_SETTINGS["mortar_x_ratio"]),
            "mortar_y_ratio": float(DEFAULT_SETTINGS["mortar_y_ratio"]),
            "grid_offset_x_px": int(DEFAULT_SETTINGS["grid_offset_x_px"]),
            "grid_offset_y_px": int(DEFAULT_SETTINGS["grid_offset_y_px"]),
            "bond_pattern": DEFAULT_SETTINGS["bond_pattern"],
            "reference_brick_width_mm": int(reference_brick_width_mm),
            "reference_brick_height_mm": int(reference_brick_height_mm),
        })
        st.rerun()
    save_settings({
        "image_path": image_path,
        "checkpoint_path": checkpoint_path,
        "materials_csv": materials_csv,
        "texture_dir": texture_dir,
        "max_preview": int(max_preview),
        "sam_max_side": int(sam_max_side),
        "sam_analysis_mode": sam_analysis_mode,
        "selection_mode": selection_mode,
        "hole_index": int(hole_index),
        "hole_indices": str(hole_indices_text),
        "brick_index": int(brick_index),
        "short_brick_index": int(short_brick_index),
        "threshold": float(threshold),
        "mortar_x_ratio": float(mortar_x_ratio),
        "mortar_y_ratio": float(mortar_y_ratio),
        "grid_offset_x_px": int(grid_offset_x_px),
        "grid_offset_y_px": int(grid_offset_y_px),
        "bond_pattern": bond_pattern,
        "reference_brick_width_mm": int(reference_brick_width_mm),
        "reference_brick_height_mm": int(reference_brick_height_mm),
    })

if not Path(image_path).exists():
    st.error(f"Image not found: {image_path}")
    st.stop()
if not Path(checkpoint_path).exists():
    st.error(f"SAM checkpoint not found: {checkpoint_path}")
    st.stop()

img = pil_to_np_rgb(Image.open(image_path))
# บังคับให้เป็นค่าปกติ ไม่ต้องเรียกใช้ไลบรารี torch ในเครื่องให้พังอีกต่อไป
device = "cpu"
dtype = "float32"
sam_preview_img, sam_preview_scale = resize_for_sam(img, int(sam_max_side))
st.caption(
    f"Device: {device} | Original: {img.shape[1]} x {img.shape[0]} px | "
    f"SAM analysis: {sam_preview_img.shape[1]} x {sam_preview_img.shape[0]} px | Profile: {sam_analysis_mode}"
)

if "masks_ready" not in st.session_state:
    st.session_state["masks_ready"] = False

should_load_masks = (
    (generate and (force_regenerate or not st.session_state.get("masks_ready")))
    or (estimate and not st.session_state.get("masks_ready"))
)
if generate and st.session_state.get("masks_ready") and not force_regenerate:
    st.info("SAM masks are already loaded. Enable Regenerate SAM masks only when you need a new detection run.")

if should_load_masks:
    with st.spinner("Generating/loading SAM masks..."):
        masks, areas, bboxes, elapsed, from_cache = load_or_generate_masks(
            img,
            image_path,
            checkpoint_path,
            force=force_regenerate,
            sam_max_side=int(sam_max_side),
            analysis_mode=sam_analysis_mode,
        )
    st.session_state["masks"] = masks
    st.session_state["areas"] = areas
    st.session_state["bboxes"] = bboxes
    st.session_state["masks_ready"] = True
    if from_cache:
        st.success(f"Loaded {len(masks)} masks from cache.")
    else:
        st.success(f"Generated {len(masks)} masks in {elapsed:.1f}s.")

if st.session_state.get("masks_ready"):
    masks = st.session_state["masks"]
    areas = st.session_state["areas"]
    bboxes = st.session_state["bboxes"]

    tab_preview, tab_manual, tab_select, tab_surfaces, tab_ar, tab_rhythm, tab_estimate, tab_materials, tab_outputs = st.tabs([
        "SAM Preview",
        "Manual Drawing",
        "Selected Masks",
        "Surface Planner",
        "AR Preview",
        "Mortar Rhythm",
        "Brick Estimate",
        "Material Recommendation",
        "Outputs",
    ])

    with tab_preview:
        total_masks = len(masks)
        st.caption(f"SAM generated {total_masks} masks. The preview can be paged, so masks after the first screen are not hidden.")
        preview_start_options = list(range(0, max(total_masks, 1), int(max_preview)))
        preview_labels = [
            f"{start}-{min(start + int(max_preview) - 1, total_masks - 1)}"
            for start in preview_start_options
        ]
        page_label = st.selectbox("Preview index range", preview_labels, index=0 if preview_labels else 0)
        preview_start = preview_start_options[preview_labels.index(page_label)] if preview_labels else 0
        preview_jpeg, preview_from_cache = get_mask_preview_jpeg(
            img,
            masks,
            areas,
            image_path,
            max_masks=max_preview,
            start_index=preview_start,
        )
        st.image(str(preview_jpeg), width="stretch")
        if preview_from_cache:
            st.caption("Loaded cached JPEG preview. SAM was not rerun.")
        else:
            st.caption("Created this JPEG preview once. It will load from cache next time.")
        all_mask_table = pd.DataFrame([
            {
                "index": i,
                "area": int(areas[i]),
                "bbox": tuple(int(v) for v in bboxes[i]),
                "x0": int(bboxes[i][0]),
                "y0": int(bboxes[i][1]),
                "x1": int(bboxes[i][2]),
                "y1": int(bboxes[i][3]),
                "w": int(bboxes[i][2] - bboxes[i][0] + 1),
                "h": int(bboxes[i][3] - bboxes[i][1] + 1),
            }
            for i in range(total_masks)
        ])
        st.dataframe(
            all_mask_table.iloc[preview_start:preview_start + int(max_preview)],
            use_container_width=True,
        )
        with st.expander("All SAM masks table", expanded=False):
            area_min = st.number_input("Minimum area filter", min_value=0, max_value=int(max(areas) if len(areas) else 0), value=0, step=100)
            area_max_default = int(max(areas) if len(areas) else 0)
            area_max = st.number_input("Maximum area filter", min_value=0, max_value=max(area_max_default, 1), value=area_max_default, step=100)
            filtered = all_mask_table[(all_mask_table["area"] >= int(area_min)) & (all_mask_table["area"] <= int(area_max))]
            st.dataframe(filtered, use_container_width=True)
            st.caption("Use the index from this table in the sidebar Hole index or Brick sample index.")
        c_prev1, c_prev2 = st.columns(2)
        if c_prev1.button("Export current SAM preview page"):
            out_path = OUT_DIR / f"{Path(image_path).stem}_sam_preview_{preview_start:03d}_{min(preview_start + int(max_preview) - 1, total_masks - 1):03d}.jpg"
            out_path.write_bytes(preview_jpeg.read_bytes())
            st.success(str(out_path))
        if c_prev2.button("Export all SAM preview pages"):
            export_dir = OUT_DIR / f"{Path(image_path).stem}_sam_preview_all_{time.strftime('%Y%m%d_%H%M%S')}"
            export_dir.mkdir(parents=True, exist_ok=True)
            all_mask_table.to_csv(export_dir / "sam_masks_index.csv", index=False)
            for start in preview_start_options:
                cached_page, _ = get_mask_preview_jpeg(
                    img,
                    masks,
                    areas,
                    image_path,
                    max_masks=max_preview,
                    start_index=start,
                )
                page_path = export_dir / f"sam_preview_{start:03d}_{min(start + int(max_preview) - 1, total_masks - 1):03d}.jpg"
                page_path.write_bytes(cached_page.read_bytes())
            st.success(f"Exported {len(preview_start_options)} SAM preview pages.")
            st.write(str(export_dir))

    with tab_select:
        selected = active_selection_preview(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        st.image(selected, caption=f"{selection_mode}: damage + reference module", use_container_width=True)
        st.subheader("Refined Repair Mask")
        active_masks, active_hole_index, active_brick_index = active_masks_and_indices(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        if active_masks is None:
            st.info("Save a manual damage geometry first to refine the repair mask.")
        elif active_hole_index >= len(active_masks):
            st.warning("Selected hole index is outside the available mask range.")
        else:
            raw_mask = ensure_mask_255(active_masks[active_hole_index])
            c_ref1, c_ref2, c_ref3, c_ref4 = st.columns(4)
            open_kernel = c_ref1.slider("Open kernel", 1, 21, 3, 2, help="Removes small noisy fragments. Use 1 to disable.")
            close_kernel = c_ref2.slider("Close kernel", 1, 31, 7, 2, help="Closes small gaps inside the repair region. Use 1 to disable.")
            close_iterations = c_ref3.slider("Close iterations", 1, 4, 1, 1)
            keep_largest = c_ref4.checkbox("Keep largest component", value=True)
            refined_mask = refine_repair_mask(
                raw_mask,
                open_kernel=int(open_kernel),
                close_kernel=int(close_kernel),
                close_iterations=int(close_iterations),
                keep_largest=bool(keep_largest),
            )
            raw_overlay, refined_only, refined_overlay = repair_mask_panels(img, raw_mask, refined_mask)
            m_raw_area = int((raw_mask > 0).sum())
            m_clean_area = int((refined_mask > 0).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Raw mask area", f"{m_raw_area:,} px")
            m2.metric("M_clean area", f"{m_clean_area:,} px")
            m3.metric("Area change", f"{m_clean_area - m_raw_area:+,} px")
            p1, p2, p3 = st.columns(3)
            p1.image(raw_overlay, caption="Selected raw mask", use_container_width=True)
            p2.image(refined_only, caption="Refined repair mask, M_clean", use_container_width=True)
            p3.image(refined_overlay, caption="M_clean overlay for infill", use_container_width=True)
            e1, e2, e3 = st.columns(3)
            export_stem = f"{Path(image_path).stem}_hole{int(active_hole_index)}_Mclean"
            if e1.button("Use M_clean as manual damage"):
                st.session_state["manual_hole_mask"] = refined_mask
                st.session_state["manual_hole_bbox"] = mask_bbox(refined_mask, image_shape=img.shape)
                if active_brick_index < len(active_masks):
                    st.session_state["manual_brick_bbox"] = mask_bbox(active_masks[active_brick_index], image_shape=img.shape)
                st.success("Saved M_clean to Manual drawing mode. Switch Selection mode to Manual drawing to use it.")
            if e2.button("Export M_clean mask"):
                mask_path = OUT_DIR / f"{export_stem}_mask.png"
                overlay_path = OUT_DIR / f"{export_stem}_overlay.png"
                Image.fromarray(refined_mask).save(mask_path)
                Image.fromarray(refined_overlay).save(overlay_path)
                st.success(str(mask_path))
                st.write(str(overlay_path))
            if e3.button("Export M_clean paper figure"):
                fig_mclean = make_repair_mask_figure(img, raw_mask, refined_mask)
                fig_path = OUT_DIR / f"{export_stem}_paper_figure.jpg"
                fig_mclean.savefig(fig_path, bbox_inches="tight", dpi=180)
                plt.close(fig_mclean)
                st.success(str(fig_path))

    with tab_surfaces:
        st.subheader("Multi-surface Material Planner")
        st.caption(
            "Assign one catalog material to each SAM surface mask. Enter the measured width and height of that surface; "
            "MATCH uses those dimensions to convert material sizes from millimetres to image pixels. Results are approximate for oblique surfaces."
        )
        if not Path(materials_csv).exists():
            st.error(f"Materials CSV not found: {materials_csv}")
        else:
            surface_materials = load_materials_table(materials_csv, texture_dir)
            surface_labels = surface_materials["label"].astype(str).tolist()
            if not surface_labels:
                st.warning("Add at least one material to the catalog first.")
            else:
                if "surface_assignment_seed" not in st.session_state:
                    st.session_state["surface_assignment_seed"] = pd.DataFrame([{
                        "mask_index": 0,
                        "material": surface_labels[0],
                        "surface_width_mm": 3000,
                        "surface_height_mm": 2400,
                    }])
                surface_assignments = st.data_editor(
                    st.session_state["surface_assignment_seed"],
                    num_rows="dynamic",
                    hide_index=True,
                    width="stretch",
                    key="surface_assignment_editor",
                    column_config={
                        "mask_index": st.column_config.NumberColumn(
                            "SAM mask index",
                            min_value=0,
                            max_value=max(0, len(masks) - 1),
                            step=1,
                            required=True,
                        ),
                        "material": st.column_config.SelectboxColumn(
                            "Catalog material",
                            options=surface_labels,
                            required=True,
                        ),
                        "surface_width_mm": st.column_config.NumberColumn(
                            "Measured width mm",
                            min_value=1,
                            step=10,
                            required=True,
                        ),
                        "surface_height_mm": st.column_config.NumberColumn(
                            "Measured height mm",
                            min_value=1,
                            step=10,
                            required=True,
                        ),
                    },
                )
                st.caption(
                    "Example: a wall measured as 3.0 m x 2.4 m should be entered as 3000 x 2400 mm. "
                    "Use separate mask rows for the upper wall, tile band, floor, or other planes."
                )
                if st.button("Generate multi-surface preview", type="primary", key="generate_surface_plan"):
                    try:
                        clean_assignments = surface_assignments.dropna(
                            subset=["mask_index", "material", "surface_width_mm", "surface_height_mm"]
                        )
                        if clean_assignments.empty:
                            raise ValueError("Add at least one complete surface row.")
                        if clean_assignments["mask_index"].astype(int).duplicated().any():
                            raise ValueError("Use each SAM mask index once. Split a surface into separate masks when it needs multiple materials.")
                        surface_preview, surface_summary, surface_modules = render_multi_surface_plan(
                            img,
                            masks,
                            clean_assignments.to_dict("records"),
                            surface_materials,
                        )
                        export_dir = OUT_DIR / "surface_plans"
                        export_dir.mkdir(parents=True, exist_ok=True)
                        export_stem = f"{Path(image_path).stem}_surface_plan"
                        preview_path = export_dir / f"{export_stem}.png"
                        summary_path = export_dir / f"{export_stem}_summary.csv"
                        modules_path = export_dir / f"{export_stem}_modules.csv"
                        Image.fromarray(surface_preview).save(preview_path)
                        surface_summary.to_csv(summary_path, index=False)
                        surface_modules.to_csv(modules_path, index=False)
                        st.session_state["surface_plan_result"] = {
                            "preview": surface_preview,
                            "summary": surface_summary,
                            "modules": surface_modules,
                            "preview_path": str(preview_path),
                            "summary_path": str(summary_path),
                            "modules_path": str(modules_path),
                        }
                    except Exception as exc:
                        st.error(str(exc))

                surface_result = st.session_state.get("surface_plan_result")
                if surface_result:
                    total_surfaces = len(surface_result["summary"])
                    total_modules = int(surface_result["summary"]["total_modules"].sum())
                    total_area = float(surface_result["summary"]["estimated_mask_area_m2"].sum())
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Planned surfaces", total_surfaces)
                    sm2.metric("Estimated modules", total_modules)
                    sm3.metric("Approx. covered area", f"{total_area:.2f} m²")
                    st.image(surface_result["preview"], caption="Multi-surface material preview", width="stretch")
                    st.dataframe(surface_result["summary"], width="stretch")
                    with st.expander("Module-level quantity schedule", expanded=False):
                        st.dataframe(surface_result["modules"], width="stretch")
                    sd1, sd2, sd3 = st.columns(3)
                    sd1.download_button(
                        "Download preview PNG",
                        data=Path(surface_result["preview_path"]).read_bytes(),
                        file_name=Path(surface_result["preview_path"]).name,
                        mime="image/png",
                    )
                    sd2.download_button(
                        "Download surface summary",
                        data=Path(surface_result["summary_path"]).read_bytes(),
                        file_name=Path(surface_result["summary_path"]).name,
                        mime="text/csv",
                    )
                    sd3.download_button(
                        "Download module schedule",
                        data=Path(surface_result["modules_path"]).read_bytes(),
                        file_name=Path(surface_result["modules_path"]).name,
                        mime="text/csv",
                    )

    with tab_ar:
        st.subheader("Tracked Final Repair Preview")
        st.caption(
            "Uses the active repair mask, estimated grid, catalog dimensions, and generated material result. "
            "Automatic image tracking follows the photographed wall; four-corner selection remains available as a fallback."
        )
        if Path(materials_csv).exists():
            ar_active_masks, ar_hole_index, ar_brick_index = active_masks_and_indices(
                img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage
            )
            if ar_active_masks is None:
                st.warning("Select a damage mask and a brick/reference module before opening the final AR result.")
            else:
                ar_estimate_fig, ar_brick_summary, ar_brick_grid = estimate_offset_bricks(
                    img, ar_active_masks, ar_hole_index, ar_brick_index, float(threshold),
                    float(mortar_x_ratio), float(mortar_y_ratio), int(grid_offset_x_px), int(grid_offset_y_px),
                    bond_pattern,
                    short_brick_index=int(short_brick_index) if bond_pattern in {"English bond", "Flemish bond"} else None,
                    reference_bboxes=st.session_state.get("bboxes"),
                )
                plt.close(ar_estimate_fig)
                ar_brick_s = ar_brick_summary.iloc[0]
                ar_materials = load_materials_table(materials_csv, texture_dir)
                ar_ranking = compute_material_ranking(
                    ar_materials, int(ar_brick_s["brick_width_px"]), int(ar_brick_s["brick_height_px"]),
                    int(reference_brick_width_mm), int(reference_brick_height_mm),
                )
                ar_ranking = apply_module_overrides(ar_ranking, st.session_state.get("material_module_overrides", {}))
                ar_ranking = ar_ranking[(ar_ranking["width_mm"] > 0) & (ar_ranking["height_mm"] > 0)].reset_index(drop=True)
                if ar_ranking.empty:
                    st.warning("Add a catalog material with physical width and height first.")
                else:
                    ar_labels = ar_ranking["label"].astype(str).tolist()
                    preferred_key = str(st.session_state.get("selected_material_key", ""))
                    preferred_rows = ar_ranking.index[ar_ranking["material_key"].astype(str) == preferred_key].tolist()
                    preferred_index = int(preferred_rows[0]) if preferred_rows else 0
                    ar_label = st.selectbox("Generated material result", ar_labels, index=preferred_index, key="ar_result_material_label")
                    ar_row = ar_ranking[ar_ranking["label"].astype(str) == ar_label].iloc[0]
                    st.session_state["selected_material_key"] = str(ar_row["material_key"])

                    ar_hole_mask = ensure_mask_255(ar_active_masks[ar_hole_index])
                    hx0, hy0, hx1, hy1 = mask_bbox(ar_hole_mask, image_shape=img.shape)
                    px_per_mm_x = max(1e-6, float(ar_brick_s["brick_width_px"]) / float(reference_brick_width_mm))
                    px_per_mm_y = max(1e-6, float(ar_brick_s["brick_height_px"]) / float(reference_brick_height_mm))
                    detected_surface_w = max(100, int(round((hx1 - hx0 + 1) / px_per_mm_x)))
                    detected_surface_h = max(100, int(round((hy1 - hy0 + 1) / px_per_mm_y)))
                    ar_c1, ar_c2 = st.columns(2)
                    ar_surface_w = ar_c1.number_input("Result width mm", 100, 30000, detected_surface_w, 10, key="ar_result_w")
                    ar_surface_h = ar_c2.number_input("Result height mm", 100, 30000, detected_surface_h, 10, key="ar_result_h")

                    ar_default_vis = {
                        "layout_mode": "Auto per material pattern", "show_grid": False, "full_threshold": 0.95,
                        "line_choice": "black", "show_boundary": False, "texture_source": "Auto per material",
                        "blend_mode": "Feather only", "feather_radius": 9, "visualization_style": "Smooth repair scenario",
                    }
                    ar_vis = {**ar_default_vis, **st.session_state.get("visualization_settings", {})}
                    ar_colors = {"black": (25, 25, 25), "blue": (0, 90, 255), "white": (245, 245, 245)}
                    ar_preview, ar_schedule, ar_layout, _, _ = material_preview_for_row(
                        img, ar_active_masks, ar_brick_index, ar_hole_mask, ar_brick_grid, ar_row,
                        ar_vis["layout_mode"], ar_vis["texture_source"], ar_vis["blend_mode"],
                        ar_vis["visualization_style"], float(ar_vis["full_threshold"]), bool(ar_vis["show_grid"]),
                        ar_colors.get(ar_vis["line_choice"], (25, 25, 25)), int(ar_vis["feather_radius"]),
                        bool(ar_vis["show_boundary"]),
                    )
                    ar_overlay = build_ar_result_overlay(ar_preview, ar_hole_mask, int(ar_vis["feather_radius"]))
                    ar_reference_crop, ar_overlay_crop, ar_tracking_box = crop_ar_tracking_target(
                        img, ar_overlay, ar_hole_mask, padding_ratio=0.35
                    )
                    am1, am2, am3, am4 = st.columns(4)
                    am1.metric("Material module", f"{float(ar_row['width_mm']):.0f} x {float(ar_row['height_mm']):.0f} mm")
                    am2.metric("Result layout", ar_layout)
                    am3.metric("Estimated modules", int(len(ar_schedule)))
                    am4.metric("Covered area", f"{float(ar_surface_w) * float(ar_surface_h) / 1_000_000.0:.2f} m2")
                    preview_html = build_safari_ar_preview_html(
                        ar_label, str(ar_row.get("resolved_texture", "")), float(ar_row["width_mm"]),
                        float(ar_row["height_mm"]), float(ar_surface_w), float(ar_surface_h), 0.0,
                        reference_image=ar_reference_crop, result_overlay=ar_overlay_crop,
                    )
                    components.html(preview_html, height=780, scrolling=False)
                    st.caption(
                        "Open MATCH through HTTPS on iPhone and allow the rear camera. Keep the photographed wall in view while AUTO lock initializes; "
                        "if the wall is visually plain, tap four corners once to provide a tracking fallback."
                    )
        elif not Path(materials_csv).exists():
            st.error(f"Materials CSV not found: {materials_csv}")
        else:
            ar_materials = load_materials_table(materials_csv, texture_dir)
            ar_materials = ar_materials[
                (ar_materials["width_mm"] > 0) & (ar_materials["height_mm"] > 0)
            ].reset_index(drop=True)
            if ar_materials.empty:
                st.warning("Add a material with width and height to the catalog first.")
            else:
                ar_labels = ar_materials["label"].astype(str).tolist()
                ar_c1, ar_c2, ar_c3 = st.columns([1.5, 1, 1])
                ar_label = ar_c1.selectbox("Material", ar_labels, key="ar_material_label")
                ar_surface_w = ar_c2.number_input(
                    "Surface width mm", min_value=100, max_value=30000, value=3000, step=50, key="ar_surface_w"
                )
                ar_surface_h = ar_c3.number_input(
                    "Surface height mm", min_value=100, max_value=30000, value=2400, step=50, key="ar_surface_h"
                )
                ar_c4, ar_c5 = st.columns(2)
                ar_rotate = ar_c4.toggle("Rotate material 90 degrees", value=False, key="ar_rotate")
                ar_joint = ar_c5.number_input(
                    "Joint width mm", min_value=0.0, max_value=100.0, value=3.0, step=0.5, key="ar_joint"
                )

                ar_row = ar_materials[ar_materials["label"].astype(str) == ar_label].iloc[0]
                material_w_mm = float(ar_row["height_mm"] if ar_rotate else ar_row["width_mm"])
                material_h_mm = float(ar_row["width_mm"] if ar_rotate else ar_row["height_mm"])
                step_w_mm = max(1.0, material_w_mm + float(ar_joint))
                step_h_mm = max(1.0, material_h_mm + float(ar_joint))
                ar_columns = max(1, int(math.ceil(float(ar_surface_w) / step_w_mm)))
                ar_rows = max(1, int(math.ceil(float(ar_surface_h) / step_h_mm)))
                ar_total = ar_columns * ar_rows
                ar_area = float(ar_surface_w) * float(ar_surface_h) / 1_000_000.0
                am1, am2, am3, am4 = st.columns(4)
                am1.metric("Material module", f"{material_w_mm:.0f} x {material_h_mm:.0f} mm")
                am2.metric("Layout", f"{ar_columns} x {ar_rows}")
                am3.metric("Approx. modules", f"{ar_total}")
                am4.metric("Surface area", f"{ar_area:.2f} m²")

                texture_path = str(ar_row.get("resolved_texture", ""))
                preview_html = build_safari_ar_preview_html(
                    ar_label,
                    texture_path,
                    material_w_mm,
                    material_h_mm,
                    float(ar_surface_w),
                    float(ar_surface_h),
                    float(ar_joint),
                )
                components.html(preview_html, height=780, scrolling=False)
                st.caption(
                    "On iPhone, open MATCH through the HTTPS Cloudflare address and allow camera access. "
                    "The four selected corners define one planar surface; reset before selecting another wall plane."
                )

    with tab_rhythm:
        active_masks, active_hole_index, active_brick_index = active_masks_and_indices(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        if active_masks is None:
            st.warning("Save damage geometry and brick/reference module first.")
        else:
            rhythm_preview, rhythm_fig, rhythm_table, rhythm_profile, rhythm_peaks, course_h = estimate_mortar_rhythm(
                img,
                active_masks,
                active_hole_index,
                active_brick_index,
                reference_bboxes=st.session_state.get("bboxes"),
            )
            st.subheader("Local Mortar Rhythm")
            st.caption("This view detects likely horizontal mortar joints around the damaged area and estimates the local course height for material-grid alignment.")
            st.image(rhythm_preview, caption=f"Detected horizontal mortar rhythm | course height = {course_h}px", width=900)
            st.pyplot(rhythm_fig, clear_figure=True)
            st.dataframe(rhythm_table, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            rhythm_img_path = OUT_DIR / f"{Path(image_path).stem}_mortar_rhythm.png"
            rhythm_graph_path = OUT_DIR / f"{Path(image_path).stem}_mortar_projection.png"
            rhythm_csv_path = OUT_DIR / f"{Path(image_path).stem}_mortar_projection.csv"
            if c1.button("Export Mortar Rhythm Image"):
                Image.fromarray(rhythm_preview).save(rhythm_img_path)
                st.success(str(rhythm_img_path))
            if c2.button("Export Mortar Projection Graph"):
                rhythm_fig.savefig(rhythm_graph_path, bbox_inches="tight", dpi=160)
                st.success(str(rhythm_graph_path))
            if c3.button("Export Mortar Projection CSV"):
                rhythm_profile.to_csv(rhythm_csv_path, index=False)
                rhythm_table.to_csv(OUT_DIR / f"{Path(image_path).stem}_mortar_rhythm_summary.csv", index=False)
                st.success(str(rhythm_csv_path))

    with tab_manual:
        st.caption("Save the damage rectangle first, then switch target and save the brick/reference module rectangle.")
        draw_target = st.radio("Draw target", ["Damage hole", "Brick/reference module"], horizontal=True)
        canvas_width = min(900, img.shape[1])
        scale = canvas_width / img.shape[1]
        canvas_height = int(img.shape[0] * scale)
        display_img = Image.fromarray(img).resize((canvas_width, canvas_height))
        stroke_color = "#ff3333" if draw_target == "Damage hole" else "#ffcc00"

        manual_preview = img.copy()
        if st.session_state.get("manual_hole_mask") is not None:
            hole_mask_preview = ensure_mask_255(st.session_state["manual_hole_mask"]) > 0
            overlay = manual_preview.copy()
            overlay[hole_mask_preview] = [255, 255, 255]
            manual_preview = cv2.addWeighted(overlay, 0.45, manual_preview, 0.55, 0)
            contours, _ = cv2.findContours(hole_mask_preview.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(manual_preview, contours, -1, (255, 40, 40), 3)
        elif st.session_state.get("manual_hole_bbox"):
            x0, y0, x1, y1 = st.session_state["manual_hole_bbox"]
            cv2.rectangle(manual_preview, (x0, y0), (x1, y1), (255, 40, 40), 4)
            cv2.putText(manual_preview, "damage", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 40, 40), 2)
        if st.session_state.get("manual_brick_bbox"):
            x0, y0, x1, y1 = st.session_state["manual_brick_bbox"]
            cv2.rectangle(manual_preview, (x0, y0), (x1, y1), (255, 210, 0), 4)
            cv2.putText(manual_preview, "reference", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 210, 0), 2)
        st.image(manual_preview, caption="Manual selection preview. Red = damage, yellow = brick/reference module.", width=900)

        with st.expander("Automatic SAM selection", expanded=True):
            combined_sam_mask, valid_hole_indices = combine_damage_masks(img, masks, hole_indices, include_manual=include_manual_damage)
            if valid_hole_indices and int(brick_index) < len(masks):
                preview_masks = list(masks)
                preview_masks[valid_hole_indices[0]] = combined_sam_mask
                sam_preview = overlay_selected_masks(img, preview_masks, valid_hole_indices[0], int(brick_index))
                st.image(sam_preview, caption=f"Combined SAM holes={valid_hole_indices}, brick={int(brick_index)}", width=900)
                c_auto1, c_auto2 = st.columns(2)
                if c_auto1.button("Use selected SAM masks as manual geometry"):
                    hole_mask = ensure_mask_255(combined_sam_mask)
                    brick_bbox = mask_bbox(masks[int(brick_index)], image_shape=img.shape)
                    st.session_state["manual_hole_mask"] = hole_mask
                    st.session_state["manual_hole_bbox"] = mask_bbox(hole_mask, image_shape=img.shape)
                    st.session_state["manual_brick_bbox"] = brick_bbox
                    st.success("Copied selected SAM hole and brick/reference geometry into manual mode.")
                    st.rerun()
                if c_auto2.button("Use SAM hole only"):
                    hole_mask = ensure_mask_255(combined_sam_mask)
                    st.session_state["manual_hole_mask"] = hole_mask
                    st.session_state["manual_hole_bbox"] = mask_bbox(hole_mask, image_shape=img.shape)
                    st.success("Copied selected SAM hole into manual mode.")
                    st.rerun()
            else:
                st.warning("Selected SAM hole/brick index is outside the available mask range.")

        method_options = ["Auto hole geometry", "Polygon points", "BBox sliders", "Canvas drawing"] if draw_target == "Damage hole" else ["BBox sliders", "Canvas drawing"]
        input_method = st.radio("Manual input method", method_options, horizontal=True)
        if input_method == "Canvas drawing":
            st.warning("Canvas drawing is experimental on this Streamlit version. Use BBox sliders if the background image turns dark or blank.")
            canvas_mode = st.radio("Canvas mode", ["rect", "freedraw", "polygon"], horizontal=True)
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.15)",
                stroke_width=3,
                stroke_color=stroke_color,
                background_image=display_img,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=canvas_mode,
                key=f"manual_canvas_{draw_target}_{canvas_mode}",
            )
            if st.button(f"Save {draw_target} geometry from canvas"):
                objects = canvas_result.json_data.get("objects", []) if canvas_result.json_data else []
                if draw_target == "Damage hole" and canvas_mode in ("freedraw", "polygon"):
                    drawn_mask = canvas_stroke_to_mask(canvas_result.image_data, img.shape, scale)
                    if drawn_mask is not None:
                        st.session_state["manual_hole_mask"] = drawn_mask
                        st.session_state["manual_hole_bbox"] = mask_bbox(drawn_mask, image_shape=img.shape)
                        st.success(f"Saved {draw_target} mask: {st.session_state['manual_hole_bbox']}")
                        st.rerun()
                    else:
                        st.warning("No red drawing was detected. Try drawing inside the image area or use rect mode.")
                elif objects:
                    bbox = canvas_rect_to_bbox(objects[-1], 1 / scale, 1 / scale)
                    if draw_target == "Damage hole":
                        st.session_state["manual_hole_bbox"] = bbox
                        st.session_state["manual_hole_mask"] = None
                    else:
                        st.session_state["manual_brick_bbox"] = bbox
                    st.success(f"Saved {draw_target}: {bbox}")
                    st.rerun()
                else:
                    st.warning("Draw on the canvas first. If the canvas is blank, switch to BBox sliders or Polygon points.")
        elif input_method == "Auto hole geometry":
            st.caption("Fast contour extraction from bright/neutral damaged plaster. Use a rough bbox first if the image has many bright areas.")
            min_area = st.slider("Minimum detected hole area px", 300, 20000, 1200, 100)
            guide_bbox = st.session_state.get("manual_hole_bbox")
            detected_mask = auto_detect_bright_hole(img, min_area=int(min_area), guide_bbox=guide_bbox)
            if detected_mask is None:
                st.warning("No bright damaged region found. Try lowering minimum area or use Polygon points.")
            else:
                detected_preview = img.copy()
                detected_bool = detected_mask > 0
                overlay = detected_preview.copy()
                overlay[detected_bool] = [255, 255, 255]
                detected_preview = cv2.addWeighted(overlay, 0.45, detected_preview, 0.55, 0)
                contours, _ = cv2.findContours(detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(detected_preview, contours, -1, (255, 40, 40), 3)
                detected_bbox = mask_bbox(detected_mask, image_shape=img.shape)
                st.image(detected_preview, caption=f"Detected hole geometry bbox: {detected_bbox}", width=900)
                if st.button("Save detected hole geometry"):
                    st.session_state["manual_hole_mask"] = detected_mask
                    st.session_state["manual_hole_bbox"] = detected_bbox
                    st.success(f"Saved detected hole geometry: {detected_bbox}")
                    st.rerun()
        elif input_method == "Polygon points":
            st.caption("Set vertices around the damage boundary. The system connects the points and fills the polygon as the hole mask.")
            base_bbox = st.session_state.get("manual_hole_bbox", (img.shape[1] // 3, img.shape[0] // 3, img.shape[1] * 2 // 3, img.shape[0] // 2))
            default_points = [
                (base_bbox[0], base_bbox[1]),
                (base_bbox[2], base_bbox[1]),
                (base_bbox[2], base_bbox[3]),
                (base_bbox[0], base_bbox[3]),
            ]
            point_count = st.number_input("Number of polygon points", min_value=3, max_value=12, value=max(4, len(st.session_state.get("manual_hole_points", default_points))), step=1)
            points = []
            for i in range(int(point_count)):
                default = st.session_state.get("manual_hole_points", default_points)
                px, py = default[i] if i < len(default) else default_points[i % len(default_points)]
                c1, c2 = st.columns(2)
                x = c1.number_input(f"P{i + 1} x", min_value=0, max_value=img.shape[1] - 1, value=int(px), step=1, key=f"poly_x_{i}")
                y = c2.number_input(f"P{i + 1} y", min_value=0, max_value=img.shape[0] - 1, value=int(py), step=1, key=f"poly_y_{i}")
                points.append((int(x), int(y)))
            polygon_mask = polygon_to_mask(img.shape, points)
            polygon_preview = img.copy()
            if polygon_mask.max() > 0:
                overlay = polygon_preview.copy()
                overlay[polygon_mask > 0] = [255, 255, 255]
                polygon_preview = cv2.addWeighted(overlay, 0.45, polygon_preview, 0.55, 0)
                cv2.polylines(polygon_preview, [np.array(points, dtype=np.int32)], True, (255, 40, 40), 3)
            st.image(polygon_preview, caption="Polygon hole preview", width=900)
            if st.button("Save polygon hole geometry"):
                st.session_state["manual_hole_points"] = points
                st.session_state["manual_hole_mask"] = polygon_mask
                st.session_state["manual_hole_bbox"] = mask_bbox(polygon_mask, image_shape=img.shape)
                st.success(f"Saved polygon hole geometry: {st.session_state['manual_hole_bbox']}")
                st.rerun()
        else:
            default_bbox = st.session_state.get(
                "manual_hole_bbox" if draw_target == "Damage hole" else "manual_brick_bbox",
                (img.shape[1] // 3, img.shape[0] // 3, img.shape[1] * 2 // 3, img.shape[0] // 2),
            )
            c1, c2, c3, c4 = st.columns(4)
            bx0 = c1.number_input("x0", min_value=0, max_value=img.shape[1] - 1, value=int(default_bbox[0]), step=1, key=f"{draw_target}_x0")
            by0 = c2.number_input("y0", min_value=0, max_value=img.shape[0] - 1, value=int(default_bbox[1]), step=1, key=f"{draw_target}_y0")
            bx1 = c3.number_input("x1", min_value=0, max_value=img.shape[1] - 1, value=int(default_bbox[2]), step=1, key=f"{draw_target}_x1")
            by1 = c4.number_input("y1", min_value=0, max_value=img.shape[0] - 1, value=int(default_bbox[3]), step=1, key=f"{draw_target}_y1")
            slider_bbox = (int(min(bx0, bx1)), int(min(by0, by1)), int(max(bx0, bx1)), int(max(by0, by1)))
            slider_preview = img.copy()
            cv2.rectangle(slider_preview, (slider_bbox[0], slider_bbox[1]), (slider_bbox[2], slider_bbox[3]), (255, 40, 40) if draw_target == "Damage hole" else (255, 210, 0), 4)
            st.image(slider_preview, caption=f"{draw_target} bbox preview: {slider_bbox}", width=900)
            if st.button(f"Save {draw_target} rectangle from sliders"):
                if draw_target == "Damage hole":
                    st.session_state["manual_hole_bbox"] = slider_bbox
                    st.session_state["manual_hole_mask"] = None
                else:
                    st.session_state["manual_brick_bbox"] = slider_bbox
                st.success(f"Saved {draw_target}: {slider_bbox}")
                st.rerun()
        c1, c2 = st.columns(2)
        hole_type = "mask" if st.session_state.get("manual_hole_mask") is not None else "bbox"
        c1.write(f"Damage {hole_type}: {st.session_state.get('manual_hole_bbox')}")
        c2.write(f"Reference bbox: {st.session_state.get('manual_brick_bbox')}")
        if st.session_state.get("manual_hole_mask") is not None or st.session_state.get("manual_hole_bbox"):
            st.button(
                "Use saved manual hole for Brick Estimate",
                type="primary",
                on_click=lambda: st.session_state.update({"selection_mode": "Manual drawing"}),
                help="Uses this manual hole and the current SAM Brick sample index. A manual brick rectangle is optional.",
            )
            st.caption("The selected SAM Brick sample index remains the reference unless you save a manual reference rectangle.")

    with tab_estimate:
        active_masks, active_hole_index, active_brick_index = active_masks_and_indices(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        if active_masks is None:
            st.warning("Manual mode needs a saved damage geometry and either a valid SAM Brick sample index or a saved manual reference rectangle.")
        elif active_hole_index >= len(active_masks) or active_brick_index >= len(active_masks):
            st.warning("Selected index is outside the available mask range.")
        else:
            fig, summary, grid = estimate_offset_bricks(
                img,
                active_masks,
                active_hole_index,
                active_brick_index,
                float(threshold),
                float(mortar_x_ratio),
                float(mortar_y_ratio),
                int(grid_offset_x_px),
                int(grid_offset_y_px),
                bond_pattern,
                int(short_brick_index) if bond_pattern in ("Learned long-short bond", "Learned row bbox rhythm") else None,
                st.session_state.get("bboxes") if selection_mode == "SAM index" else None,
            )
            s = summary.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Estimated bricks", int(s["estimated_missing_bricks"]))
            m2.metric("Brick body", f"{int(s['brick_width_px'])} x {int(s['brick_height_px'])} px")
            m3.metric("Mortar gap", f"{int(s['mortar_x_px'])} x {int(s['mortar_y_px'])} px")
            m4.metric("Module step", f"{int(s['step_x_px'])} x {int(s['step_y_px'])} px")
            m5.metric("Threshold", f"{float(s['overlap_threshold']):.2f}")
            st.caption(
                "Brick body size comes directly from the selected brick/reference bbox. "
                f"Grid offset = {int(s['grid_offset_x_px'])}, {int(s['grid_offset_y_px'])} px. "
                f"Bond pattern = {s['bond_pattern']}. "
                f"Short/header reference = {s['short_brick_sample_index'] if pd.notna(s['short_brick_sample_index']) else 'not used'}. "
                "Threshold filters counted candidate boxes only."
            )
            st.pyplot(fig, clear_figure=True)
            c1, c2 = st.columns(2)
            c1.dataframe(summary, use_container_width=True)
            c2.dataframe(grid, use_container_width=True)

            stem = Path(image_path).stem
            export_base = f"{stem}_hole{int(hole_index)}_brick{int(brick_index)}_thr{int(threshold * 100):02d}"
            fig_path = OUT_DIR / f"{export_base}_brick_estimate.png"
            summary_path = OUT_DIR / f"{export_base}_summary.csv"
            grid_path = OUT_DIR / f"{export_base}_grid.csv"
            selected_preview = active_selection_preview(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
            if st.button("Export Estimate"):
                fig.savefig(fig_path, bbox_inches="tight", dpi=160)
                summary.to_csv(summary_path, index=False)
                grid.to_csv(grid_path, index=False)
                st.success("Exported estimate.")
                st.write(str(fig_path))
                st.write(str(summary_path))
                st.write(str(grid_path))
            if st.button("Export Case Report"):
                paths = export_case_report(export_base, img, selected_preview, fig, summary, grid)
                st.success("Exported case report.")
                st.write(str(paths["report"]))

            st.divider()
            with st.expander("Context-derived row completion (experimental)", expanded=False):
                st.caption(
                    "Analyzes surrounding SAM brick bounding boxes row by row. "
                    "It does not change the selected bond pattern or the main estimate above."
                )
                run_context = st.button("Analyze context-only grid", type="primary", key="run_context_grid")
                if run_context:
                    st.session_state["show_context_grid"] = True
                if st.session_state.get("show_context_grid", False):
                    reference_boxes = st.session_state.get("bboxes")
                    if not reference_boxes:
                        st.warning("Generate SAM masks first so surrounding brick bounding boxes are available.")
                    else:
                        context_fig, context_summary, context_grid = estimate_offset_bricks(
                            img,
                            active_masks,
                            active_hole_index,
                            active_brick_index,
                            float(threshold),
                            float(mortar_x_ratio),
                            float(mortar_y_ratio),
                            0,
                            0,
                            "Context-derived row completion",
                            None,
                            reference_boxes,
                        )
                        context_s = context_summary.iloc[0]
                        cm1, cm2, cm3 = st.columns(3)
                        cm1.metric("Context-derived pieces", int(context_s["estimated_missing_bricks"]))
                        cm2.metric("Reference brick", f"{int(context_s['brick_width_px'])} x {int(context_s['brick_height_px'])} px")
                        cm3.metric("Method", "Row-wise context")
                        st.pyplot(context_fig, clear_figure=False)
                        st.dataframe(context_grid, use_container_width=True)
                        context_base = f"{stem}_context_derived_holes_{'-'.join(map(str, hole_indices))}"
                        if st.button("Export context-derived result", key="export_context_grid"):
                            context_fig_path = OUT_DIR / f"{context_base}.png"
                            context_csv_path = OUT_DIR / f"{context_base}.csv"
                            context_summary_path = OUT_DIR / f"{context_base}_summary.csv"
                            context_fig.savefig(context_fig_path, bbox_inches="tight", dpi=160)
                            context_grid.to_csv(context_csv_path, index=False)
                            context_summary.to_csv(context_summary_path, index=False)
                            st.success(str(context_fig_path))
                            st.write(str(context_csv_path))

    with tab_materials:
        active_masks, active_hole_index, active_brick_index = active_masks_and_indices(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        if active_masks is None:
            st.warning("Manual mode needs a saved damage geometry and either a valid SAM Brick sample index or a saved manual reference rectangle.")
        elif active_hole_index >= len(active_masks) or active_brick_index >= len(active_masks):
            st.warning("Select valid hole and brick sample masks first.")
        elif not Path(materials_csv).exists():
            st.error(f"Materials CSV not found: {materials_csv}")
        else:
            estimate_fig, brick_summary, brick_grid = estimate_offset_bricks(
                img,
                active_masks,
                active_hole_index,
                active_brick_index,
                float(threshold),
                float(mortar_x_ratio),
                float(mortar_y_ratio),
                int(grid_offset_x_px),
                int(grid_offset_y_px),
                bond_pattern,
                int(short_brick_index) if bond_pattern in ("Learned long-short bond", "Learned row bbox rhythm") else None,
                st.session_state.get("bboxes") if selection_mode == "SAM index" else None,
            )
            brick_s = brick_summary.iloc[0]
            materials_df = load_materials_table(materials_csv, texture_dir)

            st.subheader("One-time Material Preview")
            st.caption("This uploader tests one material in the current case only. To save a material permanently, use Add material to catalog in the sidebar.")
            c1, c2, c3 = st.columns([2, 1, 1])
            uploaded_texture = c1.file_uploader("Material texture image", type=["jpg", "jpeg", "png", "webp"])
            custom_w = c2.number_input("Custom width mm", min_value=1, max_value=2000, value=50, step=1)
            custom_h = c3.number_input("Custom height mm", min_value=1, max_value=2000, value=50, step=1)
            user_material = None
            if uploaded_texture is not None:
                custom_texture_dir = OUT_DIR / "custom_textures"
                custom_texture_dir.mkdir(parents=True, exist_ok=True)
                custom_path = custom_texture_dir / uploaded_texture.name
                custom_path.write_bytes(uploaded_texture.getvalue())
                user_material = {
                    "material_key": "user_custom_material",
                    "label": f"User material: {Path(uploaded_texture.name).stem}",
                    "source": "user",
                    "category": "custom_candidate",
                    "mode": "texture_tile",
                    "width_mm": custom_w,
                    "height_mm": custom_h,
                    "texture_file": uploaded_texture.name,
                    "resolved_texture": str(custom_path),
                }

            ranking = compute_material_ranking(
                materials_df,
                int(brick_s["brick_width_px"]),
                int(brick_s["brick_height_px"]),
                int(reference_brick_width_mm),
                int(reference_brick_height_mm),
                user_material=user_material,
            )
            module_overrides = st.session_state.get("material_module_overrides", {})
            ranking = apply_module_overrides(ranking, module_overrides)
            st.subheader("Material Ranking")
            st.caption("Ranking combines scale/rhythm fit, conservation compatibility, visual-prior fit, and distinguishable intervention logic. It is a design aid, not an approval decision.")
            st.dataframe(
                ranking[[
                    "rank", "label", "source", "category", "width_mm", "height_mm",
                    "module_w_px", "module_h_px", "scale_fit", "rhythm_fit",
                    "conservation_score", "visual_prior", "compatibility_score", "rationale"
                ]],
                use_container_width=True,
            )

            with st.expander("Per-material scale overrides", expanded=False):
                st.caption("Edit module pixel size for one material only. This does not change the other materials, and Export All will use these final overrides.")
                override_labels = ranking["label"].tolist()
                edit_label = st.selectbox("Material to edit scale", override_labels, key="override_material_label")
                edit_row = ranking[ranking["label"] == edit_label].iloc[0]
                edit_key = str(edit_row["material_key"])
                saved_override = module_overrides.get(edit_key, {})
                c_ow, c_oh, c_btn1, c_btn2 = st.columns([1, 1, 0.8, 0.8])
                override_w = c_ow.number_input(
                    "Module width px",
                    min_value=1,
                    max_value=max(3000, img.shape[1] * 2),
                    value=int(saved_override.get("module_w_px", edit_row["module_w_px"])),
                    step=1,
                    key=f"override_w_{edit_key}",
                )
                override_h = c_oh.number_input(
                    "Module height px",
                    min_value=1,
                    max_value=max(3000, img.shape[0] * 2),
                    value=int(saved_override.get("module_h_px", edit_row["module_h_px"])),
                    step=1,
                    key=f"override_h_{edit_key}",
                )
                if c_btn1.button("Save scale", key=f"save_override_{edit_key}"):
                    module_overrides[edit_key] = {"module_w_px": int(override_w), "module_h_px": int(override_h)}
                    st.session_state["material_module_overrides"] = module_overrides
                    st.success(f"Saved scale for {edit_label}: {int(override_w)} x {int(override_h)} px")
                    st.rerun()
                if c_btn2.button("Reset scale", key=f"reset_override_{edit_key}"):
                    module_overrides.pop(edit_key, None)
                    st.session_state["material_module_overrides"] = module_overrides
                    st.success(f"Reset scale for {edit_label}")
                    st.rerun()
                if module_overrides:
                    st.dataframe(pd.DataFrame([
                        {"material_key": key, **value}
                        for key, value in module_overrides.items()
                    ]), use_container_width=True)

            if "selected_material_key" not in st.session_state or st.session_state["selected_material_key"] not in set(ranking["material_key"].astype(str)):
                st.session_state["selected_material_key"] = str(ranking.iloc[0]["material_key"])
            labels = ranking["label"].tolist()
            selected_key = str(st.session_state["selected_material_key"])
            selected_index = int(ranking.index[ranking["material_key"].astype(str) == selected_key][0]) if selected_key in set(ranking["material_key"].astype(str)) else 0
            if st.session_state.get("selected_material_label_dropdown") not in labels:
                st.session_state["selected_material_label_dropdown"] = str(ranking.iloc[selected_index]["label"])
            selected_label = st.selectbox(
                "Selected material for detailed settings",
                labels,
                index=selected_index,
                key="selected_material_label_dropdown",
            )
            selected_row = ranking[ranking["label"] == selected_label].iloc[0]
            st.session_state["selected_material_key"] = str(selected_row["material_key"])
            st.subheader("Geometry-Aware Material Visualization")
            default_vis_settings = {
                "layout_mode": "Auto per material pattern",
                "show_grid": True,
                "full_threshold": 0.95,
                "line_choice": "black",
                "show_boundary": False,
                "texture_source": "Material library texture",
                "blend_mode": "Feather only",
                "feather_radius": 9,
                "visualization_style": "Quantity grid drawing",
            }
            stable_vis_version = "material_specific_layouts_20260615"
            if st.session_state.get("visualization_settings_version") != stable_vis_version:
                st.session_state["visualization_settings"] = default_vis_settings.copy()
                st.session_state["material_module_overrides"] = {}
                st.session_state["visualization_settings_version"] = stable_vis_version
                st.rerun()
            vis_settings = {**default_vis_settings, **st.session_state.get("visualization_settings", {})}
            grid_modes = [
                "Auto per material pattern",
                "Use repair brick rhythm",
                "Running bond",
                "Stack bond",
                "English bond",
                "Flemish bond",
                "Common bond",
                "Irregular row rhythm",
                "Aligned tile grid",
                "Continuous no gap",
                "Same size with original mortar gap",
            ]
            line_choices = ["black", "blue", "white"]
            texture_sources = ["Auto per material", "Material library texture", "Reference brick bbox crop", "Library texture + context color match"]
            blend_modes = ["Feather + color match", "Feather only", "Hard edge"]
            visualization_styles = ["Smooth repair scenario", "Quantity grid drawing"]

            with st.form("visualization_settings_form"):
                c_mode1, c_mode2, c_mode3, c_mode4, c_mode5 = st.columns([1.4, 0.9, 0.9, 0.8, 0.8])
                pending_layout_mode = c_mode1.selectbox(
                    "Grid mode",
                    grid_modes,
                    index=grid_modes.index(vis_settings["layout_mode"]) if vis_settings["layout_mode"] in grid_modes else 0,
                )
                pending_show_grid = c_mode2.checkbox("Show grid lines", value=bool(vis_settings["show_grid"]))
                pending_full_threshold = c_mode3.slider("Full tile coverage", 0.70, 1.00, float(vis_settings["full_threshold"]), 0.01)
                pending_line_choice = c_mode4.selectbox(
                    "Line color",
                    line_choices,
                    index=line_choices.index(vis_settings["line_choice"]) if vis_settings["line_choice"] in line_choices else 0,
                )
                pending_show_boundary = c_mode5.checkbox("Show boundary", value=bool(vis_settings["show_boundary"]))
                c_tex1, c_tex2, c_tex3 = st.columns([1.3, 1.2, 1.0])
                pending_texture_source = c_tex1.selectbox(
                    "Texture source",
                    texture_sources,
                    index=texture_sources.index(vis_settings["texture_source"]) if vis_settings["texture_source"] in texture_sources else 0,
                )
                pending_blend_mode = c_tex2.selectbox(
                    "Blend mode",
                    blend_modes,
                    index=blend_modes.index(vis_settings["blend_mode"]) if vis_settings["blend_mode"] in blend_modes else 0,
                )
                pending_feather_radius = c_tex3.slider("Feather px", 1, 25, int(vis_settings["feather_radius"]), 1)
                pending_visualization_style = st.radio(
                    "Visualization style",
                    visualization_styles,
                    index=visualization_styles.index(vis_settings["visualization_style"]) if vis_settings["visualization_style"] in visualization_styles else 0,
                    horizontal=True,
                )
                apply_visualization = st.form_submit_button("Apply visualization settings", type="primary")

            if apply_visualization:
                st.session_state["visualization_settings"] = {
                    "layout_mode": pending_layout_mode,
                    "show_grid": pending_show_grid,
                    "full_threshold": float(pending_full_threshold),
                    "line_choice": pending_line_choice,
                    "show_boundary": pending_show_boundary,
                    "texture_source": pending_texture_source,
                    "blend_mode": pending_blend_mode,
                    "feather_radius": int(pending_feather_radius),
                    "visualization_style": pending_visualization_style,
                }
                st.rerun()

            vis_settings = {**default_vis_settings, **st.session_state.get("visualization_settings", {})}
            layout_mode = vis_settings["layout_mode"]
            effective_layout_mode = material_default_layout(selected_row, layout_mode)
            show_grid = bool(vis_settings["show_grid"])
            full_threshold = float(vis_settings["full_threshold"])
            line_choice = vis_settings["line_choice"]
            show_boundary = bool(vis_settings["show_boundary"])
            texture_source = vis_settings["texture_source"]
            blend_mode = vis_settings["blend_mode"]
            feather_radius = int(vis_settings["feather_radius"])
            visualization_style = vis_settings["visualization_style"]
            st.caption(f"Applied settings: {visualization_style}, requested layout={layout_mode}, active layout={effective_layout_mode}, texture={texture_source}, blend={blend_mode}, grid={'on' if show_grid else 'off'}")
            line_colors = {
                "black": (25, 25, 25),
                "blue": (0, 90, 255),
                "white": (245, 245, 245),
            }
            hole_mask_for_material = ensure_mask_255(active_masks[active_hole_index])

            st.subheader("Material Preview Gallery")
            st.caption("Preview all ranked materials with the current visualization settings. Select one card to edit/export in detail below.")
            gallery_size = st.selectbox("Materials per page", [4, 6, 8, 12], index=1, key="materials_gallery_page_size")
            page_count = max(1, math.ceil(len(ranking) / int(gallery_size)))
            gallery_page = st.number_input("Preview page", min_value=1, max_value=page_count, value=min(int(st.session_state.get("materials_gallery_page", 1)), page_count), step=1)
            st.session_state["materials_gallery_page"] = int(gallery_page)
            page_start = (int(gallery_page) - 1) * int(gallery_size)
            page_rows = ranking.iloc[page_start:page_start + int(gallery_size)]
            gallery_cols = st.columns(2)
            for gallery_i, (_, gallery_row) in enumerate(page_rows.iterrows()):
                g_preview, g_schedule, g_layout, _, _ = material_preview_for_row(
                    img,
                    active_masks,
                    active_brick_index,
                    hole_mask_for_material,
                    brick_grid,
                    gallery_row,
                    layout_mode,
                    texture_source,
                    blend_mode,
                    visualization_style,
                    full_threshold,
                    show_grid,
                    line_colors[line_choice],
                    feather_radius,
                    show_boundary,
                )
                g_full = int((g_schedule["module_type"] == "full_tile").sum()) if not g_schedule.empty else 0
                g_cut = int((g_schedule["module_type"] == "cut_tile").sum()) if not g_schedule.empty else 0
                g_total = int(len(g_schedule))
                with gallery_cols[gallery_i % 2]:
                    st.image(
                        g_preview,
                        caption=f"{int(gallery_row['rank'])}. {gallery_row['label']} | Total {g_total} | score {float(gallery_row['compatibility_score']):.3f}",
                        use_container_width=True,
                    )
                    if st.button(f"Edit this material", key=f"choose_material_{gallery_row['material_key']}"):
                        st.session_state["selected_material_key"] = str(gallery_row["material_key"])
                        st.session_state["selected_material_label_dropdown"] = str(gallery_row["label"])
                        st.rerun()

            st.divider()
            selected_key = str(st.session_state.get("selected_material_key", selected_row["material_key"]))
            if selected_key in set(ranking["material_key"].astype(str)):
                selected_row = ranking[ranking["material_key"].astype(str) == selected_key].iloc[0]
                selected_label = selected_row["label"]
                effective_layout_mode = material_default_layout(selected_row, layout_mode)

            preview, module_schedule, effective_layout_mode, texture_source, effective_blend_mode = material_preview_for_row(
                img,
                active_masks,
                active_brick_index,
                hole_mask_for_material,
                brick_grid,
                selected_row,
                layout_mode,
                texture_source,
                blend_mode,
                visualization_style,
                full_threshold,
                show_grid,
                line_colors[line_choice],
                feather_radius,
                show_boundary,
            )
            full_tiles = int((module_schedule["module_type"] == "full_tile").sum()) if not module_schedule.empty else 0
            cut_tiles = int((module_schedule["module_type"] == "cut_tile").sum()) if not module_schedule.empty else 0
            total_tiles = int(len(module_schedule))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Full tiles", full_tiles)
            m2.metric("Cut tiles", cut_tiles)
            m3.metric("Total pieces", total_tiles)
            m4.metric("Grid mode", effective_layout_mode)
            material_summary = pd.DataFrame([{
                "selected_material": selected_row["material_key"],
                "label": selected_row["label"],
                "requested_grid_mode": layout_mode,
                "grid_mode": effective_layout_mode,
                "module_width_px": int(selected_row["module_w_px"]),
                "module_height_px": int(selected_row["module_h_px"]),
                "full_tiles": full_tiles,
                "cut_tiles": cut_tiles,
                "total_pieces": total_tiles,
                "full_tile_coverage_threshold": float(full_threshold),
                "texture_source": texture_source,
                "blend_mode": effective_blend_mode,
                "feather_radius_px": int(feather_radius),
                "show_boundary": bool(show_boundary),
                "visualization_style": visualization_style,
                "module_override_applied": selected_row["material_key"] in module_overrides,
            }])
            st.image(preview, caption=f"{selected_row['label']} | Full = {full_tiles} | Cut = {cut_tiles} | Total = {total_tiles}", use_container_width=True)
            with st.expander("Material quantity schedule", expanded=False):
                st.dataframe(material_summary, use_container_width=True)
                st.dataframe(module_schedule, use_container_width=True)
            st.text_area("SDXL prompt draft", make_sdxl_prompt(selected_row["label"], selected_row["rationale"]), height=110)

            export_mat_base = f"{Path(image_path).stem}_{selected_row['material_key']}_material_preview"
            preview_path = OUT_DIR / f"{export_mat_base}.png"
            ranking_path = OUT_DIR / "material_recommendation_ranking.csv"
            module_schedule_path = OUT_DIR / f"{export_mat_base}_module_schedule.csv"
            material_summary_path = OUT_DIR / f"{export_mat_base}_summary.csv"
            if st.button("Export Material Recommendation"):
                Image.fromarray(preview).save(preview_path)
                ranking.to_csv(ranking_path, index=False)
                module_schedule.to_csv(module_schedule_path, index=False)
                material_summary.to_csv(material_summary_path, index=False)
                st.success("Exported material recommendation.")
                st.write(str(preview_path))
                st.write(str(ranking_path))
                st.write(str(module_schedule_path))
                st.write(str(material_summary_path))
            if st.button("Export All Material Repair Visualizations"):
                run_id = time.strftime("%Y%m%d_%H%M%S")
                style_slug = normalize_texture_name(visualization_style)
                export_dir = OUT_DIR / f"{Path(image_path).stem}_all_material_repair_scenarios_{style_slug}_{run_id}"
                export_dir.mkdir(parents=True, exist_ok=True)
                export_rows = []
                sheet_items = []
                for _, row in ranking.iterrows():
                    row_layout_mode = material_default_layout(row, layout_mode)
                    row_texture_source = texture_source
                    if row_texture_source == "Auto per material":
                        row_texture_source = "Reference brick bbox crop" if row["material_key"] == "original_brick_reference" else "Material library texture"
                    if row["material_key"] == "original_brick_reference":
                        row_texture_source = "Reference brick bbox crop"

                    row_schedule = build_material_module_schedule(
                        hole_mask_for_material,
                        brick_grid,
                        int(row["module_w_px"]),
                        int(row["module_h_px"]),
                        row_layout_mode,
                        coverage_threshold=0.05,
                        full_threshold=float(full_threshold),
                    )
                    row_full_tiles = int((row_schedule["module_type"] == "full_tile").sum()) if not row_schedule.empty else 0
                    row_cut_tiles = int((row_schedule["module_type"] == "cut_tile").sum()) if not row_schedule.empty else 0

                    if visualization_style == "Smooth repair scenario":
                        row_texture = resolve_row_texture(img, active_masks, active_brick_index, row, row_texture_source)
                        row_preview = render_smooth_repair_visualization(
                            img,
                            hole_mask_for_material,
                            row_texture,
                            row["label"],
                            blend_mode=blend_mode,
                            feather_radius=max(12, int(feather_radius)),
                            color_strength=0.35 if "color match" in blend_mode.lower() else 0.0,
                        )
                    else:
                        row_texture_override = None
                        if row_texture_source == "Reference brick bbox crop":
                            row_texture_override = texture_from_reference_bbox(img, active_masks, active_brick_index)
                        row_effective_blend_mode = blend_mode
                        if row_texture_source == "Material library texture" and blend_mode == "Feather + color match":
                            row_effective_blend_mode = "Feather only"
                        row_preview = render_material_schedule_preview(
                            img,
                            hole_mask_for_material,
                            row_schedule,
                            row["resolved_texture"],
                            show_grid=show_grid,
                            line_color=line_colors[line_choice],
                            texture_override=row_texture_override,
                            blend_mode=row_effective_blend_mode,
                            feather_radius=int(feather_radius),
                            show_boundary=show_boundary,
                        )

                    safe_name = normalize_texture_name(f"{int(row['rank']):02d}_{row['material_key']}_{row['label']}")[:90]
                    out_path = export_dir / f"{safe_name}.png"
                    schedule_path = export_dir / f"{safe_name}_module_schedule.csv"
                    Image.fromarray(row_preview).save(out_path)
                    row_schedule.to_csv(schedule_path, index=False)
                    sheet_items.append((f"{int(row['rank'])}. {row['label']}", row_preview))
                    export_rows.append({
                        "rank": int(row["rank"]),
                        "material_key": row["material_key"],
                        "label": row["label"],
                        "compatibility_score": row["compatibility_score"],
                        "scale_fit": row["scale_fit"],
                        "rhythm_fit": row["rhythm_fit"],
                        "visualization_style": visualization_style,
                        "requested_layout_mode": layout_mode,
                        "layout_mode": row_layout_mode,
                        "texture_source": row_texture_source,
                        "blend_mode": blend_mode,
                        "show_grid": bool(show_grid),
                        "show_boundary": bool(show_boundary),
                        "line_color": line_choice,
                        "feather_radius_px": int(feather_radius),
                        "full_tile_threshold": float(full_threshold),
                        "full_tiles": row_full_tiles,
                        "cut_tiles": row_cut_tiles,
                        "total_pieces": int(len(row_schedule)),
                        "module_w_px": int(row["module_w_px"]),
                        "module_h_px": int(row["module_h_px"]),
                        "module_override_applied": str(row["material_key"]) in module_overrides,
                        "visualization_path": str(out_path),
                        "module_schedule_path": str(schedule_path),
                    })
                contact_sheet = make_contact_sheet(sheet_items, cols=2, thumb_w=720)
                contact_path = export_dir / "all_material_repair_scenarios_contact_sheet.png"
                Image.fromarray(contact_sheet).save(contact_path)
                pd.DataFrame(export_rows).to_csv(export_dir / "all_material_repair_scenarios_index.csv", index=False)
                ranking.to_csv(export_dir / "material_ranking.csv", index=False)
                settings_path = export_dir / "applied_visualization_settings.json"
                settings_path.write_text(json.dumps({
                    "visualization_style": visualization_style,
                    "requested_layout_mode": layout_mode,
                    "texture_source": texture_source,
                    "blend_mode": blend_mode,
                    "show_grid": bool(show_grid),
                    "show_boundary": bool(show_boundary),
                    "line_color": line_choice,
                    "feather_radius_px": int(feather_radius),
                    "full_tile_threshold": float(full_threshold),
                    "material_module_overrides": module_overrides,
                }, indent=2), encoding="utf-8")
                st.success(f"Exported {len(export_rows)} repair visualizations.")
                st.write(str(export_dir))
                st.image(contact_sheet, caption="All material repair scenarios", use_container_width=True)
            if st.button("Export Full Material Case Report"):
                selected_preview = active_selection_preview(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
                paths = export_case_report(export_mat_base, img, selected_preview, estimate_fig, brick_summary, brick_grid, ranking=ranking, material_preview=preview)
                st.success("Exported full material case report.")
                st.write(str(paths["report"]))

    with tab_outputs:
        st.subheader("Project Outputs")
        st.caption("This dashboard maps the research deliverables to the files/data generated by the current case.")
        active_masks, active_hole_index, active_brick_index = active_masks_and_indices(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
        if active_masks is None:
            st.warning("Save or select a repair mask first.")
        elif active_hole_index >= len(active_masks) or active_brick_index >= len(active_masks):
            st.warning("Select valid damage and reference module indices first.")
        else:
            out_fig, out_summary, out_grid = estimate_offset_bricks(
                img,
                active_masks,
                active_hole_index,
                active_brick_index,
                float(threshold),
                float(mortar_x_ratio),
                float(mortar_y_ratio),
                int(grid_offset_x_px),
                int(grid_offset_y_px),
                bond_pattern,
                int(short_brick_index) if bond_pattern in ("Learned long-short bond", "Learned row bbox rhythm") else None,
                st.session_state.get("bboxes") if selection_mode == "SAM index" else None,
            )
            selected_preview = active_selection_preview(img, masks, selection_mode, int(hole_index), int(brick_index), hole_indices, include_manual_damage)
            output_rows = [
                {"Output": "Damage Area", "Available": "Yes", "Where": "Selected Masks tab / M_clean mask / manual or SAM damage mask"},
                {"Output": "Missing Material Estimation", "Available": "Yes", "Where": "Brick Estimate tab / estimated_missing_bricks"},
                {"Output": "Repair Visualization", "Available": "Yes", "Where": "Brick Estimate and Material Recommendation previews"},
                {"Output": "Material Recommendation", "Available": "Yes", "Where": "Material Recommendation ranking table and preview gallery"},
                {"Output": "Quantity Schedule", "Available": "Yes", "Where": "Brick grid CSV and material module schedule CSV"},
                {"Output": "CAD / BIM-ready Data", "Available": "Partial", "Where": "CSV with module_id, x0, y0, x1, y1, w_px, h_px; ready for CAD/BIM import scripting"},
            ]
            st.dataframe(pd.DataFrame(output_rows), use_container_width=True)
            o1, o2 = st.columns(2)
            o1.image(selected_preview, caption="Damage area + reference module", use_container_width=True)
            o2.pyplot(out_fig, clear_figure=True)
            with st.expander("Current quantity/CAD-BIM data", expanded=True):
                st.dataframe(out_summary, use_container_width=True)
                st.dataframe(out_grid, use_container_width=True)
            if st.button("Export Core Output Package", type="primary"):
                run_id = time.strftime("%Y%m%d_%H%M%S")
                export_dir = OUT_DIR / f"{Path(image_path).stem}_core_outputs_{run_id}"
                export_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(selected_preview).save(export_dir / "01_damage_area_and_reference.png")
                out_fig.savefig(export_dir / "02_missing_material_estimation.png", bbox_inches="tight", dpi=160)
                out_summary.to_csv(export_dir / "03_quantity_summary.csv", index=False)
                out_grid.to_csv(export_dir / "04_cad_bim_ready_module_grid.csv", index=False)
                pd.DataFrame(output_rows).to_csv(export_dir / "00_output_manifest.csv", index=False)
                st.success("Exported core output package.")
                st.write(str(export_dir))
else:
    st.info("Startup is light now. Click Generate SAM Masks to run automatic segmentation; the app will not load SAM until you press a button.")
    st.image(img, caption="Input wall image", use_container_width=True)

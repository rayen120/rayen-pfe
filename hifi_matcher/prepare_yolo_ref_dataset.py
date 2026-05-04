"""
Build a YOLOv8 dataset for reference-region detection from existing folders.

Input folders:
- Filters: C:/Users/espace info/Pictures/filtres
- Etickets: C:/Users/espace info/Pictures/etiquette

Output:
- <out>/images/train, <out>/images/val
- <out>/labels/train, <out>/labels/val
- <out>/data.yaml

Label class:
- 0: reference
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
REF_RE = re.compile(r"(SA|SN)\s*([0-9OILBQGZ]{4,6})", re.IGNORECASE)

if os.name == "nt":
    for p in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in IMAGE_EXTS:
            pil = Image.open(path)
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            arr = np.array(pil)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def list_images(folder: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            out.append(p)
    return out


def tesseract_data(rgb: np.ndarray, psm: int) -> dict:
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        return pytesseract.image_to_data(rgb, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:
        return {}


def _normalize_token(token: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", (token or "").upper())
    trans = str.maketrans({"O": "0", "I": "1", "L": "1", "B": "8", "Q": "0", "G": "6", "Z": "2"})
    return t.translate(trans)


def _find_ref_in_text(text: str) -> bool:
    if not text:
        return False
    t = _normalize_token(text)
    if not t:
        return False
    return bool(REF_RE.search(t))


@dataclass
class CandidateBox:
    x: int
    y: int
    w: int
    h: int
    score: float


def _extract_candidate_boxes(bgr: np.ndarray) -> List[CandidateBox]:
    h, w = bgr.shape[:2]
    variants: List[Tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))
    variants.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)))
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)))

    out: List[CandidateBox] = []

    for _name, rgb in variants:
        for psm in (6, 7, 11):
            d = tesseract_data(rgb, psm)
            if not d or "text" not in d:
                continue
            n = len(d["text"])
            line_groups = {}
            for i in range(n):
                txt = (d["text"][i] or "").strip()
                if not txt:
                    continue
                try:
                    x = int(d["left"][i])
                    y = int(d["top"][i])
                    ww = int(d["width"][i])
                    hh = int(d["height"][i])
                    conf = float(d["conf"][i]) if str(d["conf"][i]).strip() not in ("", "-1") else 0.0
                    blk = int(d.get("block_num", [0] * n)[i])
                    par = int(d.get("par_num", [0] * n)[i])
                    line = int(d.get("line_num", [0] * n)[i])
                except Exception:
                    continue
                if ww < 8 or hh < 8:
                    continue
                key = (blk, par, line)
                line_groups.setdefault(key, []).append((txt, x, y, ww, hh, conf))
                if not _find_ref_in_text(txt):
                    continue
                # expand box to capture whole reference string area
                px = max(4, int(0.18 * ww))
                py = max(4, int(0.45 * hh))
                xa = max(0, x - px)
                ya = max(0, y - py)
                xb = min(w, x + ww + px)
                yb = min(h, y + hh + py)
                bw = xb - xa
                bh = yb - ya
                if bw < 12 or bh < 10:
                    continue
                out.append(CandidateBox(xa, ya, bw, bh, conf))
            # Also parse merged OCR lines because SA and digits are often split into separate tokens.
            for words in line_groups.values():
                words = sorted(words, key=lambda t: t[1])
                line_text = " ".join(w[0] for w in words if w[0]).strip()
                if not _find_ref_in_text(line_text):
                    continue
                xs = [w[1] for w in words]
                ys = [w[2] for w in words]
                x2s = [w[1] + w[3] for w in words]
                y2s = [w[2] + w[4] for w in words]
                confs = [w[5] for w in words]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w, max(x2s))
                y2 = min(h, max(y2s))
                ww = x2 - x1
                hh = y2 - y1
                if ww < 12 or hh < 10:
                    continue
                px = max(4, int(0.12 * ww))
                py = max(4, int(0.35 * hh))
                xa = max(0, x1 - px)
                ya = max(0, y1 - py)
                xb = min(w, x2 + px)
                yb = min(h, y2 + py)
                out.append(CandidateBox(xa, ya, xb - xa, yb - ya, float(np.mean(confs) if confs else 0.0)))

    return out


def _merge_boxes(boxes: List[CandidateBox], w: int, h: int) -> Optional[CandidateBox]:
    if not boxes:
        return None
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)[:12]
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x + b.w for b in boxes)
    y2 = max(b.y + b.h for b in boxes)
    # slight global padding
    pad_x = max(6, int(0.02 * w))
    pad_y = max(6, int(0.02 * h))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 20 or bh < 12:
        return None
    score = float(sum(b.score for b in boxes) / max(1, len(boxes)))
    return CandidateBox(x1, y1, bw, bh, score)


def yolo_line_for_box(box: CandidateBox, img_w: int, img_h: int) -> str:
    xc = (box.x + box.w / 2.0) / float(img_w)
    yc = (box.y + box.h / 2.0) / float(img_h)
    ww = box.w / float(img_w)
    hh = box.h / float(img_h)
    return f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"


def ensure_clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main() -> None:
    _here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", default=r"C:\Users\espace info\Pictures\filtres")
    parser.add_argument("--etickets", default=r"C:\Users\espace info\Pictures\etiquette")
    parser.add_argument("--out", default=os.path.join(_here, "hifi_ref_dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src = list_images(args.filters) + list_images(args.etickets)
    if not src:
        raise SystemExit("No images found in input folders.")

    random.shuffle(src)

    train_img = os.path.join(args.out, "images", "train")
    val_img = os.path.join(args.out, "images", "val")
    train_lbl = os.path.join(args.out, "labels", "train")
    val_lbl = os.path.join(args.out, "labels", "val")
    ensure_clean_dir(train_img)
    ensure_clean_dir(val_img)
    ensure_clean_dir(train_lbl)
    ensure_clean_dir(val_lbl)

    kept = 0
    dropped = 0
    records = []

    for p in src:
        bgr = load_image_bgr(p)
        if bgr is None:
            dropped += 1
            continue
        h, w = bgr.shape[:2]
        cands = _extract_candidate_boxes(bgr)
        merged = _merge_boxes(cands, w, h)
        if merged is None:
            dropped += 1
            continue
        records.append((p, merged))
        kept += 1

    if kept < 8:
        raise SystemExit(f"Too few labeled images detected ({kept}). Need at least 8.")

    n_val = max(1, int(round(len(records) * max(0.05, min(0.4, args.val_ratio)))))
    val_set = set(i for i in random.sample(range(len(records)), n_val))

    for i, (src_path, box) in enumerate(records):
        split = "val" if i in val_set else "train"
        dst_img_dir = val_img if split == "val" else train_img
        dst_lbl_dir = val_lbl if split == "val" else train_lbl
        stem = f"{i:05d}_{os.path.splitext(os.path.basename(src_path))[0]}"
        dst_img = os.path.join(dst_img_dir, stem + ".jpg")
        dst_lbl = os.path.join(dst_lbl_dir, stem + ".txt")

        img = load_image_bgr(src_path)
        if img is None:
            continue
        cv2.imwrite(dst_img, img)
        hh, ww = img.shape[:2]
        line = yolo_line_for_box(box, ww, hh)
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write(line + "\n")

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {args.out.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: reference\n")

    print(f"dataset_out={args.out}")
    print(f"source_images={len(src)}")
    print(f"kept_labeled={kept}")
    print(f"dropped_unlabeled={dropped}")
    print(f"train_count={len(records) - n_val}")
    print(f"val_count={n_val}")
    print(f"data_yaml={yaml_path}")


if __name__ == "__main__":
    main()

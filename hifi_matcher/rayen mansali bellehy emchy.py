import os
import re
import subprocess
import sys
import webbrowser
import sqlite3
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
import json
import threading
import time
from datetime import datetime

import cv2
import imutils
import numpy as np
from PIL import Image, ImageTk
from pillow_heif import register_heif_opener
from rapidfuzz import fuzz
import pytesseract
try:
    from ultralytics import YOLO
except Exception:  # noqa: BLE001
    YOLO = None
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

try:
    from paddleocr import PaddleOCR
except Exception:  # noqa: BLE001
    PaddleOCR = None
try:
    from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore[reportMissingImports]
except Exception:  # noqa: BLE001
    pyzbar_decode = None


register_heif_opener()

if os.name == "nt":
    _tess_paths = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    )
    for _tp in _tess_paths:
        if os.path.isfile(_tp):
            pytesseract.pytesseract.tesseract_cmd = _tp
            break

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# Optional Pi → STM32 conveyor: set HIFI_CONVEYOR_SERIAL (e.g. /dev/ttyACM0), optional HIFI_CONVEYOR_BAUD (default 115200).
# Match → "RUN\n", mismatch → "STOP\n" (see stm32_conveyor/conveyor_control.c).
_matcher_conveyor_uart_last = [None, 0.0]
_matcher_conveyor_uart_lock = threading.Lock()


def _matcher_notify_conveyor_uart_impl(tone):
    if tone not in ("match", "mismatch"):
        return
    port = (os.environ.get("HIFI_CONVEYOR_SERIAL") or "").strip()
    if not port:
        return
    now = time.monotonic()
    with _matcher_conveyor_uart_lock:
        if tone == _matcher_conveyor_uart_last[0] and (now - _matcher_conveyor_uart_last[1]) < 0.22:
            return
        _matcher_conveyor_uart_last[0] = tone
        _matcher_conveyor_uart_last[1] = now
    payload = b"RUN\n" if tone == "match" else b"STOP\n"
    baud = int(os.environ.get("HIFI_CONVEYOR_BAUD") or "115200")

    def _job():
        try:
            import serial
        except ImportError:
            return
        try:
            ser = serial.Serial(port, baud, timeout=0.25, write_timeout=0.25)
            try:
                ser.write(payload)
                ser.flush()
            finally:
                ser.close()
        except Exception:
            pass

    threading.Thread(target=_job, daemon=True).start()


# Lenient pattern for noisy OCR (letters/digits confusions corrected downstream).
REF_PATTERN = re.compile(r"(SA|SN)\s*([0-9OIl]{4,6})", re.IGNORECASE)
# Strict digits-only block for physical filter labels.
STRICT_FILTER_REF_PATTERN = re.compile(r"(SA|SN)\s*(\d{4,6})", re.IGNORECASE)
SA_SN_REF_REGEX = re.compile(r"(SA|SN)\s*(\d{4,6})", re.IGNORECASE)
HIFI_GENERIC_REF_PATTERN = re.compile(r"\b[A-Z]{2}\s?\d{4,6}\b", re.IGNORECASE)
SERIAL_NUMBER_PATTERN = re.compile(r"\b\d{10,16}\b")


def _ean13_checksum_ok(code):
    """GS1 EAN-13 check digit (12 left digits weighted 1,3,1,3,...)."""
    if not code or len(code) != 13 or not code.isdigit():
        return False
    s = sum(int(a) * int(b) for a, b in zip(code[:12], "131313131313"))
    ck = (10 - (s % 10)) % 10
    return ck == int(code[12])


FILTER_PREFIX_LENGTHS = {
    "SA": 5,
    "SN": 6,
}
KNOWN_REFERENCE_SET = set()

ETICKET_CACHE_FILE = ".eticket_ocr_cache_v2.json"
FILTER_CACHE_FILE = ".filter_ocr_cache_v1.json"
FILTER_OCR_CACHE_VERSION = "v6"
YOLO_REF_MODEL_PATH = os.environ.get(
    "HIFI_YOLO_REF_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "ref_detector.pt"),
)
YOLO_REF_CONF = 0.30
YOLO_REF_MAX_BOXES = 4

_PADDLE_OCR_ENGINE = None
_PADDLE_OCR_LOCK = threading.Lock()
_LIVE_OCR_PREVIEW_STOP = threading.Event()
_LIVE_OCR_PREVIEW_THREAD = None
_LIVE_OCR_LAST_ERROR = ""
LIVE_OCR_PREVIEW_SIDE = 960
LIVE_OCR_DETECT_MAX_SIDE = 640
LIVE_OCR_MAX_RATE = 12.0


def _get_paddle_ocr_engine():
    global _PADDLE_OCR_ENGINE
    if PaddleOCR is None:
        return None
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_ENGINE is not None:
            return _PADDLE_OCR_ENGINE
        if _PADDLE_OCR_ENGINE is None:
            try:
                _PADDLE_OCR_ENGINE = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,
                    show_log=False,
                )
            except TypeError:
                _PADDLE_OCR_ENGINE = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=False,
                )
            except Exception:
                return None
    return _PADDLE_OCR_ENGINE


def _paddle_ocr_boxes(bgr):
    ocr = _get_paddle_ocr_engine()
    if ocr is None or bgr is None or bgr.size == 0:
        return []
    with _PADDLE_OCR_LOCK:
        try:
            try:
                res = ocr.ocr(bgr, cls=True)
            except TypeError:
                res = ocr.ocr(bgr)
        except Exception:
            return []
    if not res:
        return []
    lines = res[0] if isinstance(res, (list, tuple)) and res else []
    out = []
    for line in lines or []:
        if not line or len(line) < 2:
            continue
        poly, txt_conf = line[0], line[1]
        if not poly or not txt_conf:
            continue
        text = str(txt_conf[0] or "").strip()
        conf = float(txt_conf[1]) if len(txt_conf) > 1 else 0.0
        if not text:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        out.append((x1, y1, x2, y2, text, conf))
    return out


def _tesseract_word_boxes(gray):
    if gray is None or gray.size == 0:
        return []
    cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    try:
        d = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    n = len(d.get("text", []))
    out = []
    for i in range(n):
        t = (d["text"][i] or "").strip()
        if not t:
            continue
        try:
            x, y, w, h = int(d["left"][i]), int(d["top"][i]), int(d["width"][i]), int(d["height"][i])
            conf = float(d["conf"][i]) if str(d["conf"][i]).strip() not in ("", "-1") else 0.0
        except Exception:
            continue
        out.append((x, y, x + w, y + h, t, conf))
    return out


def _draw_live_overlay(frame_bgr, ocr_fps=None):
    """
    Draw SA/SN word boxes on frame. Returns (vis_bgr, best_ref) where best_ref is the
    highest-confidence valid HIFI reference parsed from box text (or '').
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, ""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    txt_scale = max(0.55, min(1.0, w / 1200.0))
    txt_thick = 2 if w >= 900 else 1
    detect_img = downscale_for_ocr(vis, max_side=LIVE_OCR_DETECT_MAX_SIDE)
    dh, dw = detect_img.shape[:2]
    sx = w / float(max(1, dw))
    sy = h / float(max(1, dh))
    g = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    boxes = _tesseract_word_boxes(clahe)
    best_ref = ""
    best_conf = -1.0
    detected_tokens = []
    for x1, y1, x2, y2, text, conf in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        if not REF_PATTERN.search(text.upper()):
            if text:
                detected_tokens.append(str(text))
            continue
        if text:
            detected_tokens.append(str(text))
        compact = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
        word_ref = ""
        for m in re.finditer(r"(SA|SN)([0-9OILBQGZ]{4,8})", compact):
            cand = correct_reference_parts(
                (m.group(1) or "").upper(),
                (m.group(2) or ""),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            cand = normalize_reference(cand)
            if cand and is_valid_hifi_reference(cand):
                word_ref = cand
                break
        if word_ref and float(conf) > best_conf:
            best_conf = float(conf)
            best_ref = word_ref
        rx1 = int(round(x1 * sx))
        ry1 = int(round(y1 * sy))
        rx2 = int(round(x2 * sx))
        ry2 = int(round(y2 * sy))
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        label = f"{text} ({conf:.0f})" if conf else text
        ty = max(0, ry1 - 8)
        cv2.putText(
            vis,
            label,
            (rx1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            txt_scale,
            (0, 0, 0),
            txt_thick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (rx1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            txt_scale,
            (0, 0, 255),
            txt_thick,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        "Live OCR (SA/SN boxes)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.7, txt_scale),
        (0, 0, 0),
        txt_thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        "Live OCR (SA/SN boxes)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.7, txt_scale),
        (0, 255, 0),
        txt_thick,
        cv2.LINE_AA,
    )
    fps_txt = (
        f"OCR FPS:{ocr_fps:.1f}"
        if ocr_fps is not None
        else "OCR FPS:--"
    )
    cv2.putText(
        vis,
        fps_txt,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.65, txt_scale),
        (0, 0, 0),
        txt_thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        fps_txt,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.65, txt_scale),
        (0, 255, 0),
        txt_thick,
        cv2.LINE_AA,
    )
    # Extra live checks requested by user (presence only).
    blob = re.sub(r"[^A-Z0-9]", "", " ".join(detected_tokens).upper())
    has_eac = bool(("EAC" in blob) or (fuzz.partial_ratio("EAC", blob) >= 66 if blob else False))
    has_hifi = bool(("HIFI" in blob) or (fuzz.partial_ratio("HIFI", blob) >= 68 if blob else False))
    has_phrase = bool(
        (
            ("AIR" in blob)
            and (
                ("FILTRE" in blob)
                or ("FILTER" in blob)
                or (fuzz.partial_ratio("FILTRE", blob) >= 70 if blob else False)
                or (fuzz.partial_ratio("FILTER", blob) >= 70 if blob else False)
            )
        )
    )
    checks = [("EAC", has_eac), ("HIFI", has_hifi), ("PHRASE", has_phrase), ("REF", bool(best_ref))]
    y0 = 52
    for i, (name, ok) in enumerate(checks):
        color = (0, 220, 0) if ok else (0, 0, 255)
        text = f"{name}:{'YES' if ok else 'NO'}"
        y = y0 + (i * 22)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    return vis, best_ref


def _draw_live_ocr_compare_strip(vis_bgr, live_ref, eticket_target_ref):
    """Second-row hint when matcher etiquette target is set (OpenCV BGR image)."""
    tgt = normalize_reference(eticket_target_ref or "")
    if not tgt or vis_bgr is None or vis_bgr.size == 0:
        return vis_bgr
    live = normalize_reference(live_ref or "")
    exact = bool(live and live == tgt)
    score = eticket_compatibility_vs_filter(live, tgt) if live else 0
    h = vis_bgr.shape[0]
    line_a = f"vs etiquette: {tgt} | live: {live or '---'}"
    line_b = f"Exact: {'YES' if exact else 'NO'}   compat: {score}%"
    y1, y2 = 48, 72
    for ln, y in ((line_a, y1), (line_b, y2)):
        cv2.putText(
            vis_bgr,
            ln[:95],
            (10, min(y, h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis_bgr


def _live_ocr_emit_compare_to_matcher_ui(app, live_ref):
    """Thread-safe: schedules a short status append on the Tk root when etiquette compare is active."""
    if app is None:
        return
    tgt = normalize_reference(getattr(app, "matcher_selected_eticket_ref", "") or "")
    if not tgt:
        return
    live = normalize_reference(live_ref or "")
    key = (live, tgt)
    now = time.monotonic()
    last_key = getattr(app, "_live_ocr_compare_emit_key", None)
    last_ts = float(getattr(app, "_live_ocr_compare_emit_ts", 0.0) or 0.0)
    if key == last_key and (now - last_ts) < 1.8:
        return
    app._live_ocr_compare_emit_key = key
    app._live_ocr_compare_emit_ts = now
    exact = bool(live and live == tgt)
    score = eticket_compatibility_vs_filter(live, tgt) if live else 0
    line = (
        f"[Live OCR View] live {live or '(none)'} vs etiquette {tgt} | "
        f"Exact: {'YES' if exact else 'NO'} | compat {score}%"
    )

    def _do():
        try:
            app.note_live_ocr_compare_status(line)
        except Exception:
            pass

    try:
        app.root.after(0, _do)
    except Exception:
        pass


def _live_ocr_preview_worker(app=None):
    global _LIVE_OCR_LAST_ERROR
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        _LIVE_OCR_LAST_ERROR = "Could not open PC camera (index 0)."
        return
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cv2.namedWindow("Live OCR", cv2.WINDOW_NORMAL)
    last = 0.0
    ocr_fps = 0.0
    min_interval = 1.0 / max(1.0, float(LIVE_OCR_MAX_RATE))
    last_overlay = None
    while not _LIVE_OCR_PREVIEW_STOP.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        now = time.monotonic()
        small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        if now - last >= min_interval:
            t0 = time.monotonic()
            small, cam_ref = _draw_live_overlay(small, ocr_fps=ocr_fps)
            if app is not None and normalize_reference(
                getattr(app, "matcher_selected_eticket_ref", "") or ""
            ):
                small = _draw_live_ocr_compare_strip(small, cam_ref, app.matcher_selected_eticket_ref)
                _live_ocr_emit_compare_to_matcher_ui(app, cam_ref)
            dt = max(1e-3, time.monotonic() - t0)
            inst = 1.0 / dt
            ocr_fps = inst if ocr_fps <= 0 else (0.85 * ocr_fps + 0.15 * inst)
            last = now
            last_overlay = small
        else:
            if last_overlay is not None:
                small = last_overlay.copy()
            else:
                cv2.putText(
                    small,
                    "Live OCR (SA/SN boxes)",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    small,
                    f"OCR FPS:{ocr_fps:.1f}" if ocr_fps > 0 else "OCR FPS:--",
                    (10, small.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        cv2.imshow("Live OCR", small)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    try:
        cv2.destroyWindow("Live OCR")
    except Exception:
        pass


def start_live_ocr_preview(master=None, app=None):
    global _LIVE_OCR_PREVIEW_THREAD, _LIVE_OCR_LAST_ERROR
    if _LIVE_OCR_PREVIEW_THREAD is not None and _LIVE_OCR_PREVIEW_THREAD.is_alive():
        messagebox.showinfo("Live OCR", "Live OCR window is already running.", parent=master)
        return
    _LIVE_OCR_LAST_ERROR = ""
    _LIVE_OCR_PREVIEW_STOP.clear()
    _LIVE_OCR_PREVIEW_THREAD = threading.Thread(
        target=lambda: _live_ocr_preview_worker(app),
        daemon=True,
    )
    _LIVE_OCR_PREVIEW_THREAD.start()
    def _check_cam():
        if _LIVE_OCR_PREVIEW_THREAD is None:
            return
        if _LIVE_OCR_PREVIEW_THREAD.is_alive():
            if master is not None and getattr(master, "winfo_exists", lambda: 0)():
                try:
                    master.after(250, _check_cam)
                except Exception:
                    pass
            return
        err = (_LIVE_OCR_LAST_ERROR or "").strip()
        if err and master is not None and getattr(master, "winfo_exists", lambda: 0)():
            messagebox.showerror("Live OCR", err, parent=master)
    if master is not None and getattr(master, "winfo_exists", lambda: 0)():
        try:
            master.after(300, _check_cam)
        except Exception:
            pass


def stop_live_ocr_preview():
    _LIVE_OCR_PREVIEW_STOP.set()

# Downscale large images before Tesseract (speed / memory).
OCR_MAX_SIDE = 1280

# Sharpening kernel for verification popups
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def downscale_for_ocr(image_bgr, max_side=None):
    """Shrink very large photos before OCR (major speed-up; little impact on SA/SN text)."""
    if max_side is None:
        max_side = OCR_MAX_SIDE
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image_bgr
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def sharpen_bgr(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    return cv2.filter2D(image_bgr, -1, SHARPEN_KERNEL)


TESSERACT_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _tesseract_cfg_psm(psm):
    return f"--oem 1 --psm {psm} -c tessedit_char_whitelist={TESSERACT_WHITELIST}"


_YOLO_REF_MODEL = None
_YOLO_REF_MODEL_FAILED = False
_YOLO_DEVICE = (
    0
    if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available())
    else "cpu"
)


def _get_yolo_ref_model():
    global _YOLO_REF_MODEL, _YOLO_REF_MODEL_FAILED
    if _YOLO_REF_MODEL is not None:
        return _YOLO_REF_MODEL
    if _YOLO_REF_MODEL_FAILED or YOLO is None:
        return None
    if not os.path.isfile(YOLO_REF_MODEL_PATH):
        _YOLO_REF_MODEL_FAILED = True
        return None
    try:
        _YOLO_REF_MODEL = YOLO(YOLO_REF_MODEL_PATH)
        return _YOLO_REF_MODEL
    except Exception:  # noqa: BLE001
        _YOLO_REF_MODEL_FAILED = True
        return None


def _yolo_reference_rois(image_bgr, max_boxes=YOLO_REF_MAX_BOXES):
    """
    Detect likely reference text areas with YOLOv8 detector.
    Expected training class is one box around SA/SN reference region.
    """
    model = _get_yolo_ref_model()
    if model is None or image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    try:
        preds = model.predict(
            source=image_bgr,
            conf=YOLO_REF_CONF,
            iou=0.45,
            verbose=False,
            max_det=max(1, int(max_boxes)),
            device=_YOLO_DEVICE,
        )
    except Exception:  # noqa: BLE001
        return []
    if not preds:
        return []
    boxes = getattr(preds[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    items = []
    for b in boxes:
        try:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0]) if b.conf is not None else 0.0
        except Exception:  # noqa: BLE001
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        # Slight padding to keep full text when detector is tight.
        pad_x = max(4, int(0.06 * (x2 - x1)))
        pad_y = max(4, int(0.20 * (y2 - y1)))
        xa = max(0, x1 - pad_x)
        ya = max(0, y1 - pad_y)
        xb = min(w, x2 + pad_x)
        yb = min(h, y2 + pad_y)
        if xb - xa < 20 or yb - ya < 10:
            continue
        items.append((conf, image_bgr[ya:yb, xa:xb].copy()))
    items.sort(key=lambda t: t[0], reverse=True)
    return [roi for _c, roi in items[: max(1, int(max_boxes))]]


def ocr_text_tesseract(image_bgr):
    """Full-image sparse text via Tesseract."""
    if image_bgr is None or image_bgr.size == 0:
        return ""
    img = downscale_for_ocr(image_bgr.copy())
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return pytesseract.image_to_string(rgb, config=_tesseract_cfg_psm(11))
    except Exception:  # noqa: BLE001
        return ""


def _tesseract_image_to_string_timeout(image_rgb, config, timeout_sec):
    """
    Run Tesseract in a daemon thread; signal completion with threading.Event.
    Returns (text, timed_out). If timeout_sec is None, wait until completion.
    """
    result_box = [""]
    done = threading.Event()

    def run():
        try:
            result_box[0] = pytesseract.image_to_string(image_rgb, config=config) or ""
        except Exception:  # noqa: BLE001
            result_box[0] = ""
        finally:
            done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    if timeout_sec is None:
        done.wait()
        return result_box[0], False
    if not done.wait(timeout=max(0.001, float(timeout_sec))):
        return "", True
    return result_box[0], False


def _filter_ocr_four_preprocessed_rgb(image_bgr):
    """Build the four filter OCR inputs (RGB) and PSM for each, in order."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    out = []
    h, w = image_bgr.shape[:2]
    # Attempt 1: center 60%, 3x, grayscale, CLAHE, Otsu, PSM 7
    x0, x1 = int(0.2 * w), int(0.8 * w)
    y0, y1 = int(0.2 * h), int(0.8 * h)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = image_bgr
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    _ret, otsu = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("attempt1_center60_clahe_otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB), 7))
    # Attempt 2: full gray 2x, CLAHE, invert, PSM 6
    gfull = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    up2 = cv2.resize(gfull, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2b = clahe2.apply(up2)
    inv = cv2.bitwise_not(g2b)
    out.append(("attempt2_2x_clahe_inv", cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB), 6))
    # Attempt 3: HSV V, CLAHE, Otsu, PSM 6
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    clahe3 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe3.apply(v)
    _ret, otsu3 = cv2.threshold(v2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("attempt3_hsv_v_clahe_otsu", cv2.cvtColor(otsu3, cv2.COLOR_GRAY2RGB), 6))
    # Attempt 4: full gray adaptive, PSM 11
    g4 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hh, ww = g4.shape[:2]
    bs = max(11, min(hh, ww) // 10)
    if bs % 2 == 0:
        bs += 1
    ad = cv2.adaptiveThreshold(
        g4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, 2
    )
    out.append(("attempt4_adaptive", cv2.cvtColor(ad, cv2.COLOR_GRAY2RGB), 11))
    return out


def extract_reference_tesseract_full(image_bgr, crop_top_fraction=None, max_side=None):
    """Tesseract PSM 11 on crop (optional) then full image."""
    if image_bgr is None or image_bgr.size == 0:
        return "", ""
    ms = max_side if max_side is not None else OCR_MAX_SIDE

    def pipeline_on_bgr(bgr):
        bgr = downscale_for_ocr(bgr, max_side=ms)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            text = pytesseract.image_to_string(rgb, config=_tesseract_cfg_psm(11))
        except Exception:  # noqa: BLE001
            text = ""
        r = extract_reference_from_text(text)
        if r:
            return r, text
        compact = re.sub(r"\s+", "", (text or "").upper())
        m = SA_SN_REF_REGEX.search(compact)
        if m:
            cand = correct_reference_parts(
                m.group(1).upper(),
                m.group(2),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if cand:
                return cand, text
        return "", text

    if crop_top_fraction is not None and 0 < crop_top_fraction < 1:
        h, _w = image_bgr.shape[:2]
        cropped = image_bgr[: max(1, int(h * crop_top_fraction)), :]
        ref, text = pipeline_on_bgr(cropped)
        if ref:
            return ref, text
        return pipeline_on_bgr(image_bgr)
    return pipeline_on_bgr(image_bgr)


def extract_strict_filter_reference(text):
    if not text:
        return ""
    variants = (text, re.sub(r"\s+", "", text.upper()))
    for candidate in variants:
        for m in re.finditer(r"(SA|SN)\s*([0-9OILBQGZ]{4,8})", candidate, flags=re.IGNORECASE):
            fixed = correct_reference_parts(
                m.group(1).upper(),
                m.group(2),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if fixed:
                return fixed
    return ""


def extract_reference_hifi_eac_priority(text):
    """
    Priority extraction for filter labels:
    1) lines/segments containing HIFI
    2) then segments containing EAC
    3) then full text fallback
    """
    if not text:
        return ""
    raw = text.upper()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    priority_segments = []
    priority_segments.extend([ln for ln in lines if "HIFI" in ln])
    priority_segments.extend([ln for ln in lines if "EAC" in ln and ln not in priority_segments])
    priority_segments.append(raw)

    for seg in priority_segments:
        m = re.search(r"\b(SA|SN)\s*([0-9OILBQGZ]{4,6})\b", seg, re.IGNORECASE)
        if m:
            fixed = correct_reference_parts(
                m.group(1),
                m.group(2),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if fixed:
                return fixed
        compact = re.sub(r"[^A-Z0-9]", "", seg.upper())
        m2 = re.search(r"(SA|SN)([0-9OILBQGZ]{4,6})", compact)
        if m2:
            fixed = correct_reference_parts(
                m2.group(1),
                m2.group(2),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if fixed:
                return fixed
    return ""


def extract_reference_from_very_noisy_text(text):
    """
    Recover SA/SN references from highly fragmented OCR, e.g.:
    - "S A 1 0 3 7 4"
    - "SAI0374" / "SN92O41O"
    """
    if not text:
        return ""
    up = (text or "").upper()
    # Keep only useful symbols and normalize common OCR confusions.
    cleaned = re.sub(r"[^A-Z0-9]", "", up)
    if not cleaned:
        return ""
    trans = str.maketrans({"O": "0", "I": "1", "L": "1", "B": "8", "G": "6", "Z": "2", "Q": "0"})
    cleaned = cleaned.translate(trans)

    # Direct compact pattern first.
    for m in re.finditer(r"(SA|SN)([0-9]{4,6})", cleaned):
        fixed = correct_reference_parts(
            m.group(1),
            m.group(2),
            allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
        )
        if fixed:
            snap = closest_known_reference(fixed, min_score=88)
            return snap or fixed

    # Fragmented stream fallback: look for S/A or S/N then next 4-6 digits.
    chars = list(cleaned)
    for i in range(len(chars) - 1):
        p = "".join(chars[i:i + 2])
        if p not in ("SA", "SN"):
            continue
        tail = "".join(c for c in chars[i + 2:] if c.isdigit())
        if len(tail) < 4:
            continue
        for n in (6, 5, 4):
            if len(tail) >= n:
                cand = correct_reference_parts(
                    p,
                    tail[:n],
                    allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
                )
                if cand:
                    snap = closest_known_reference(cand, min_score=88)
                    return snap or cand
    return ""


def _pick_best_reference_candidate(found, source_text=""):
    """
    Pick the most plausible filter reference from OCR candidates.
    Prefer repeated/explicitly seen refs over catalog-only guesses.
    """
    if not found:
        return ""
    source_up = (source_text or "").upper()
    source_compact = re.sub(r"[^A-Z0-9]", "", source_up)
    counts = {}
    first_idx = {}
    for idx, ref in enumerate(found):
        counts[ref] = counts.get(ref, 0) + 1
        if ref not in first_idx:
            first_idx[ref] = idx

    def score(ref):
        prefix, digits = split_reference(ref)
        s = counts.get(ref, 1) * 100
        # Strong signal: OCR text contains this exact compact token.
        if ref and ref in source_compact:
            s += 900
        # Mid signal: OCR text contains same prefix+digits separated by spaces.
        if prefix and digits and re.search(rf"{prefix}\s*{digits}", source_up):
            s += 450
        if ref in KNOWN_REFERENCE_SET:
            s += 120
        return s

    uniq = list(counts.keys())
    uniq.sort(key=lambda r: (-score(r), first_idx.get(r, 10**9)))
    return uniq[0] if uniq else ""


def extract_reference_from_text(text):
    if not text:
        return ""
    found = []
    for candidate in (text, re.sub(r"\s+", "", text.upper())):
        for m in REF_PATTERN.finditer(candidate):
            prefix = m.group(1).upper()
            digit_part = m.group(2)
            fixed = correct_reference_parts(
                prefix,
                digit_part,
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if fixed:
                found.append(fixed)
    if found:
        picked = _pick_best_reference_candidate(found, text)
        if picked:
            return picked
    compact = re.sub(r"[^A-Z0-9]", "", text.upper())
    for prefix, expected_len in FILTER_PREFIX_LENGTHS.items():
        m = re.search(rf"{prefix}([0-9OIl]{{{expected_len}}})", compact)
        if m:
            fixed = correct_reference_parts(prefix, m.group(1), allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()))
            if fixed:
                snap = closest_known_reference(fixed)
                if snap:
                    return snap
                return fixed
    for m in re.finditer(r"(SA|SN)([0-9OIL]{4,6})", compact):
        fixed = correct_reference_parts(m.group(1), m.group(2), allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()))
        if not fixed:
            continue
        snap = closest_known_reference(fixed)
        if snap:
            return snap
        return fixed
    return ""


def _collect_filter_ref_candidates(text):
    """Collect and normalize multiple SA/SN candidates from one OCR text blob."""
    if not text:
        return []
    out = []
    for fn in (
        extract_reference_from_text,
        extract_reference_from_very_noisy_text,
    ):
        try:
            r = fn(text)
        except Exception:
            r = ""
        r = normalize_reference(r)
        if r and is_valid_hifi_reference(r) and r not in out:
            out.append(r)
    for r in _strict_refs_from_raw_text(text):
        rn = normalize_reference(r)
        if rn and is_valid_hifi_reference(rn) and rn not in out:
            out.append(rn)
    return out


def _live_ref_from_text_fast(text):
    """Fast SA/SN parse focused on OCR confusions for live camera."""
    if not text:
        return ""
    up = (text or "").upper()
    compact = re.sub(r"[^A-Z0-9]", "", up)
    trans = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
    for src in (up, compact, compact.translate(trans)):
        for m in re.finditer(r"(SA|SN)\s*([A-Z0-9]{4,8})", src):
            fixed = correct_reference_parts(
                m.group(1).upper(),
                m.group(2),
                allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
            )
            if fixed and is_valid_hifi_reference(fixed):
                return normalize_reference(fixed)
    return ""


def _best_known_ref_from_text_confusion(raw_text, min_score=86):
    """
    Map noisy OCR text to the closest known reference using confusion-aware scoring.
    """
    if not raw_text or not KNOWN_REFERENCE_SET:
        return ""
    up = (raw_text or "").upper()
    compact = re.sub(r"[^A-Z0-9]", "", up)
    if not compact:
        return ""
    # Strictly anchor to SA/SN-like tokens first, then resolve confusion to known catalog.
    trans = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
    best_ref = ""
    best_sc = -1.0
    for m in re.finditer(r"(SA|SN)([A-Z0-9]{4,8})", compact):
        pfx = (m.group(1) or "").upper()
        tail = (m.group(2) or "").translate(trans)
        need = FILTER_PREFIX_LENGTHS.get(pfx, 0)
        if need <= 0:
            continue
        windows = [tail]
        if len(tail) > need:
            windows = [tail[i : i + need] for i in range(0, len(tail) - need + 1)]
        for wd in windows:
            cand = correct_reference_parts(pfx, wd, allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()))
            cand = normalize_reference(cand)
            if not cand or not is_valid_hifi_reference(cand):
                continue
            if cand in KNOWN_REFERENCE_SET:
                return cand
            snapped = closest_known_reference(cand, min_score=90)
            if snapped:
                return normalize_reference(snapped)
            # Last resort: choose best known ref with same prefix/length by digit similarity.
            cp, cd = split_reference(cand)
            for known in KNOWN_REFERENCE_SET:
                kp, kd = split_reference(known)
                if kp != cp or len(kd) != len(cd):
                    continue
                sc = 0.75 * fuzz.ratio(cd, kd) + 0.25 * fuzz.ratio(cand, known)
                if sc > best_sc:
                    best_sc = sc
                    best_ref = normalize_reference(known)
    if best_ref and best_sc >= float(min_score):
        return best_ref
    return ""


def _live_quick_ref_pipeline(frame_bgr):
    """
    Fast live pipeline:
    1) YOLO ROI (fallback full frame)
    2) grayscale + CLAHE
    3) OCR PSM7 on normal + inverted
    4) confusion-aware parse + known-reference snap
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return "", "empty frame"
    rois = _yolo_reference_rois(frame_bgr, max_boxes=2)
    used_yolo = bool(rois)
    if not rois:
        rois = [frame_bgr]
    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=HIFSAN0123456789-"
    candidates = []
    for roi in rois:
        if roi is None or roi.size == 0:
            continue
        up = cv2.resize(roi, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        rot_variants = (
            up,
            cv2.rotate(up, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(up, cv2.ROTATE_180),
            cv2.rotate(up, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        for rimg in rot_variants:
            gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants = (clahe, cv2.bitwise_not(clahe), otsu, cv2.bitwise_not(otsu))
            for v in variants:
                for local_cfg in (
                    cfg,
                    "--oem 1 --psm 11 -c tessedit_char_whitelist=HIFSAN0123456789-",
                ):
                    try:
                        txt = pytesseract.image_to_string(v, config=local_cfg) or ""
                    except Exception:
                        txt = ""
                    c = _live_ref_from_text_fast(txt)
                    if c:
                        candidates.append(c)
                    kk = _best_known_ref_from_text_confusion(txt, min_score=84)
                    if kk:
                        candidates.append(kk)
                    for cc in _strict_refs_from_raw_text(txt):
                        n = normalize_reference(cc)
                        if n and is_valid_hifi_reference(n):
                            candidates.append(n)
    if not candidates:
        return "", ("yolo_no_read" if used_yolo else "fullframe_no_read")
    votes = {}
    for c in candidates:
        c = normalize_reference(c)
        if not c or not is_valid_hifi_reference(c):
            continue
        weight = 1
        if c in KNOWN_REFERENCE_SET:
            weight += 2
        votes[c] = votes.get(c, 0) + weight
    if not votes:
        return "", "no_valid_candidate"
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    best = ranked[0][0]
    best_vote = ranked[0][1]
    second_vote = ranked[1][1] if len(ranked) > 1 else 0
    if best_vote < 2 or (best_vote - second_vote) <= 0:
        return "", "ambiguous_candidate"
    if best not in KNOWN_REFERENCE_SET:
        snapped = closest_known_reference(best, min_score=80)
        if snapped:
            best = normalize_reference(snapped)
    if best in KNOWN_REFERENCE_SET:
        return best, ("yolo_quick_ok" if used_yolo else "fullframe_quick_ok")
    return "", "not_in_index"


def closest_known_reference(candidate, min_score=85):
    """Snap OCR candidate to known reference using OCR-confusion-aware digit scoring."""
    if not candidate or not KNOWN_REFERENCE_SET:
        return ""
    pfx, digits = split_reference(candidate)
    if not pfx or not digits:
        return ""

    confusion_pairs = {
        ("0", "O"), ("O", "0"),
        ("1", "I"), ("I", "1"),
        ("1", "L"), ("L", "1"),
        ("5", "S"), ("S", "5"),
        ("8", "B"), ("B", "8"),
        ("2", "Z"), ("Z", "2"),
        ("6", "G"), ("G", "6"),
        # Common OCR digit-vs-digit confusions on curved embossed text.
        ("0", "8"), ("8", "0"),
        ("3", "8"), ("8", "3"),
        ("6", "8"), ("8", "6"),
        ("5", "6"), ("6", "5"),
        ("3", "9"), ("9", "3"),
        ("0", "9"), ("9", "0"),
    }

    def _digit_confusion_score(a, b):
        if not a or not b or len(a) != len(b):
            return 0.0
        score = 0.0
        for ca, cb in zip(a, b):
            if ca == cb:
                score += 1.0
            elif (ca, cb) in confusion_pairs:
                # Partial credit for likely OCR lookalikes.
                score += 0.64
            else:
                score += 0.0
        return 100.0 * (score / max(1, len(a)))

    best = ""
    best_score = -1
    for known in KNOWN_REFERENCE_SET:
        kp, kd = split_reference(known)
        if kp != pfx or len(kd) != len(digits):
            continue
        dr = fuzz.ratio(digits, kd)
        dc = _digit_confusion_score(digits, kd)
        # Keep a strict gate to avoid accidental snaps between nearby valid refs.
        if max(dr, dc) < 82:
            continue
        whole = fuzz.ratio(candidate, known)
        # Prefer confusion-aware digit alignment over whole-string fuzz.
        s = int((0.60 * dc) + (0.25 * dr) + (0.15 * whole))
        if s > best_score:
            best = known
            best_score = s
    return best if best_score >= min_score else ""


def trace_snap_to_known_reference(raw_text, min_fuzz=80):
    """
    Recover a catalog SA/SN from the full OCR trace when Tesseract scattered digits/lines
    but a true digit block (e.g. 17251) still appears. Sliding-window fuzzy match vs known refs.
    """
    if not raw_text or not KNOWN_REFERENCE_SET:
        return ""
    up = (raw_text or "").upper()
    compact = re.sub(r"[^A-Z0-9]", "", up)
    for known in KNOWN_REFERENCE_SET:
        kn = normalize_reference(known)
        if kn and len(kn) >= 7 and kn in compact:
            return kn
    digit_blob = re.sub(r"[^0-9]", "", up)
    if len(digit_blob) < 4:
        return ""
    best_ref, best_sc = "", -1
    for known in KNOWN_REFERENCE_SET:
        _p, kd = split_reference(known)
        if not kd:
            continue
        L = len(kd)
        if L not in (5, 6) or len(digit_blob) < L:
            continue
        for i in range(0, len(digit_blob) - L + 1):
            win = digit_blob[i : i + L]
            sc = fuzz.ratio(win, kd)
            if sc > best_sc:
                best_sc, best_ref = sc, normalize_reference(known)
    if best_sc >= min_fuzz and is_valid_hifi_reference(best_ref):
        return resolve_filter_ref_with_digit_catalog_vote(raw_text, best_ref)
    return ""


def _extract_refs_from_filter_trace_chunk(text_chunk):
    """Collect valid SA/SN refs from one OCR trace fragment (aligned with pick_ref heuristics)."""
    if not text_chunk:
        return []
    out = []
    fixed = text_chunk
    for bad, good in (("O", "0"), ("I", "1"), ("L", "1"), ("B", "8"), ("G", "6"), ("Z", "2"), ("Q", "0")):
        fixed = re.sub(rf"(?<=\d){bad}(?=\d)", good, fixed, flags=re.IGNORECASE)
    for fn in (
        extract_reference_from_very_noisy_text,
        extract_reference_hifi_eac_priority,
        extract_strict_filter_reference,
        extract_reference_from_text,
    ):
        try:
            r = fn(fixed)
        except Exception:  # noqa: BLE001
            r = ""
        if r:
            n = normalize_reference(r)
            if is_valid_hifi_reference(n):
                out.append(n)
    compact = re.sub(r"[^A-Z0-9]", "", fixed.upper())
    for m in re.finditer(r"(SA|SN)([0-9OIL]{4,6})", compact):
        cand = correct_reference_parts(
            m.group(1).upper(),
            m.group(2),
            allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
        )
        if cand:
            n = normalize_reference(cand)
            if is_valid_hifi_reference(n):
                out.append(n)
    seen = set()
    deduped = []
    for r in out:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def consensus_pick_filter_reference(trace_blob, primary_ref=""):
    """
    Score candidate refs from the full filter OCR trace (all attempts).
    Prefer catalog membership, repeated extractions, and digit-sequence evidence in the trace.
    """
    if not (trace_blob or "").strip() and not primary_ref:
        return ""
    all_refs = []
    for line in trace_blob.splitlines():
        payload = line.split(":", 1)[1] if ":" in line else line
        all_refs.extend(_extract_refs_from_filter_trace_chunk(payload))
    all_refs.extend(_extract_refs_from_filter_trace_chunk(trace_blob))
    prim = normalize_reference(primary_ref) if primary_ref else ""
    if prim and is_valid_hifi_reference(prim):
        all_refs.append(prim)
    snap = trace_snap_to_known_reference(trace_blob)
    if snap:
        if not prim:
            all_refs.append(snap)
        else:
            sp, _sd = split_reference(snap)
            pp, _pd = split_reference(prim)
            if sp and pp and sp == pp:
                all_refs.append(snap)
    if not all_refs:
        return prim if prim and is_valid_hifi_reference(prim) else ""
    digit_blob = re.sub(r"[^0-9]", "", trace_blob.upper())
    scores = {}
    raw_digit_blobs = []
    raw_digit_blobs = []
    raw_digit_blobs = []
    digit_support = {}
    explicit_hits = {}
    for r in set(all_refs):
        cnt = all_refs.count(r)
        sc = cnt * 15
        if KNOWN_REFERENCE_SET and r in KNOWN_REFERENCE_SET:
            sc += 45
        rc = re.sub(r"[^A-Z0-9]", "", r.upper())
        explicit_hits[r] = bool(rc and rc in re.sub(r"[^A-Z0-9]", "", trace_blob.upper()))
        _pf, kd = split_reference(r)
        best_win = 0
        if kd and len(digit_blob) >= len(kd):
            L = len(kd)
            for i in range(0, len(digit_blob) - L + 1):
                best_win = max(best_win, fuzz.ratio(digit_blob[i : i + L], kd))
            sc += int(0.35 * best_win)
        digit_support[r] = best_win
        if explicit_hits[r]:
            sc += 35
        scores[r] = sc
    winner = max(scores.keys(), key=lambda x: scores[x])
    if prim and prim in scores and winner != prim:
        wp, _wd = split_reference(winner)
        pp, _pd = split_reference(prim)
        same_prefix = bool(wp and pp and wp == pp)
        winner_strong = explicit_hits.get(winner, False) or digit_support.get(winner, 0) >= 94
        prim_weak = (not explicit_hits.get(prim, False)) and digit_support.get(prim, 0) <= 75
        if same_prefix and winner_strong and prim_weak and scores[winner] > scores[prim] + 12:
            return winner
        if same_prefix and scores[winner] > scores[prim] and all_refs.count(winner) > all_refs.count(prim):
            return winner
        return prim
    if prim and winner == prim:
        return prim
    return winner


def deskew_image(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 30))
    if coords is None or len(coords) < 5:
        return image_bgr
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.25:
        return image_bgr
    h, w = image_bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image_bgr, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _filter_grayscale_for_text_on_green(bgr):
    """
    Single channel that boosts ink / grey plastic vs saturated grass (green dominates R,B).
    Helps curved references photographed on lawn or foliage.
    """
    if bgr is None or bgr.size == 0:
        return None
    b, g, r = cv2.split(bgr)
    gi = g.astype(np.int32)
    ri, bi = r.astype(np.int32), b.astype(np.int32)
    rb = np.clip((ri + bi) // 2 - (gi * 6) // 10 + 40, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    return cv2.max(L, rb)


def _filter_bgr_for_grass_background(bgr):
    """BGR image with grass suppressed for OCR pipelines."""
    if bgr is None or bgr.size == 0:
        return bgr
    gch = _filter_grayscale_for_text_on_green(bgr)
    if gch is None:
        return bgr
    return cv2.cvtColor(gch, cv2.COLOR_GRAY2BGR)


def _hough_circle_centers_radii(gray, h, w):
    """Collect up to 6 circle hypotheses using multiple Hough strictness levels."""
    if gray is None or gray.size == 0:
        return []
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    min_r = max(8, min(h, w) // 12)
    max_r = min(h, w) // 2
    seen = []
    for param2 in (32, 26, 20, 14):
        c = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(min_r * 2, min(h, w) // 5),
            param1=95,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if c is None:
            continue
        for pt in np.uint16(np.around(c[0])):
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
            if r < 8:
                continue
            dup = False
            for sx, sy, sr in seen:
                if abs(cx - sx) + abs(cy - sy) < max(18, r // 8) and abs(r - sr) < max(12, r // 5):
                    dup = True
                    break
            if not dup:
                seen.append((cx, cy, r))
        if len(seen) >= 6:
            break
    return seen


def _filter_polar_strip_candidates(gray, cx, cy, r, w, h, prefix):
    """Unwrap curved rim text at inner/mid/outer radii; deskew each strip for Tesseract."""
    out = []
    if gray is None or gray.size == 0 or r < 8:
        return out
    max_in = max(8, min(cx, cy, max(0, w - cx - 1), max(0, h - cy - 1)))
    for fr, tag in ((0.78, "inner"), (0.90, "mid"), (0.97, "outer")):
        mxrad = max(8, min(int(r * fr), int(0.98 * max_in)))
        lp = cv2.linearPolar(
            gray,
            (float(cx), float(cy)),
            float(mxrad),
            cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )
        if lp.size == 0:
            continue
        lp_bgr = cv2.cvtColor(lp, cv2.COLOR_GRAY2BGR)
        lp_desk = deskew_image(lp_bgr)
        hh, ww = lp_desk.shape[:2]
        if hh > 0 and ww > 0 and min(hh, ww) < 140:
            lp_desk = cv2.resize(lp_desk, (ww * 2, hh * 2), interpolation=cv2.INTER_CUBIC)
        out.append((f"{prefix}_polar_{tag}", lp_desk))
    return out


def filename_strong_digit_reference_hint(path):
    """
    True when the image basename encodes the part number (e.g. 17088.jpg -> SA17088).
    In that case we trust the filename over OCR to avoid wrong 100% matches on curved caps.
    """
    base = os.path.splitext(os.path.basename(path or ""))[0].upper()
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    compact = re.sub(r"[^A-Z0-9]", "", base)
    if re.fullmatch(r"[0-9]{5}", compact):
        return True
    if re.fullmatch(r"(SA|SN)[0-9]{5,6}", compact):
        return True
    return False


def reconcile_ocr_ref_with_known_filename(path, ocr_ref, raw_text=""):
    """
    Prefer filename-inferred ref when it encodes digits (17088.jpg) or when catalog agrees.
    Fixes curved-cap OCR (e.g. SA14074) vs true part number from filename (SA17088).
    """
    inf = infer_reference_from_filename(path or "")
    if not inf or not is_valid_hifi_reference(inf):
        return ocr_ref or "", raw_text or ""

    ocr_n = normalize_reference(ocr_ref or "")
    strong = filename_strong_digit_reference_hint(path)
    if strong and inf != ocr_n:
        note = (raw_text + "\n" if raw_text else "") + (
            f"Strong filename hint: using {inf} instead of OCR ({ocr_n or 'none'})."
        )
        return inf, note

    if inf not in KNOWN_REFERENCE_SET:
        return ocr_ref or "", raw_text or ""

    if not ocr_n or not is_valid_hifi_reference(ocr_n):
        note = (raw_text + "\n" if raw_text else "") + f"Filename reference (known): {inf}"
        return inf, note
    if ocr_n not in KNOWN_REFERENCE_SET:
        note = (
            (raw_text + "\n" if raw_text else "")
            + f"Filename override: OCR={ocr_n} -> {inf} (known ref from filename; OCR not in catalog)"
        )
        return inf, note
    return ocr_ref, raw_text or ""


def preprocess_for_filter_ocr(image_bgr):
    """Return BGR variants commonly helpful before Tesseract on filter photos."""
    if image_bgr is None or image_bgr.size == 0:
        return []
    base = deskew_image(image_bgr.copy())
    out = [base, _filter_bgr_for_grass_background(base)]
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    out.append(cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR))
    th = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    out.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))
    return out


def build_filter_ocr_candidates(image_bgr):
    """Deskew + scales + Hough circles + polar unwrap strips for ring-shaped refs."""
    candidates = []
    if image_bgr is None or image_bgr.size == 0:
        return candidates
    img = deskew_image(image_bgr.copy())
    h, w = img.shape[:2]
    candidates.append(("deskew", img))
    gray_std = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_g = _filter_grayscale_for_text_on_green(img)
    gray_polar = cv2.max(gray_std, gray_g) if gray_g is not None else gray_std
    circle_list = _hough_circle_centers_radii(gray_std, h, w)
    if gray_g is not None:
        for cx, cy, r in _hough_circle_centers_radii(gray_g, h, w):
            dup = False
            for sx, sy, sr in circle_list:
                if abs(cx - sx) + abs(cy - sy) < max(20, r // 6):
                    dup = True
                    break
            if not dup:
                circle_list.append((cx, cy, r))
    for cx, cy, r in circle_list[:5]:
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            candidates.append(("circle_roi", roi))
        for pname, lp_desk in _filter_polar_strip_candidates(gray_polar, cx, cy, r, w, h, "hough"):
            candidates.append((pname, lp_desk))
    # Fallback when Hough misses the cap: unwrap from image center (common top-down filter photos).
    cx0, cy0 = w // 2, h // 2
    for r_frac in (0.40, 0.48, 0.56):
        r0 = int(r_frac * min(w, h))
        for pname, lp_desk in _filter_polar_strip_candidates(
            gray_polar, cx0, cy0, r0, w, h, f"imgcenter_r{int(100 * r_frac)}"
        ):
            candidates.append((pname, lp_desk))
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    candidates.append(("2x", big))
    g_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(g_full)
    candidates.append(("clahe_full", cv2.cvtColor(g_clahe, cv2.COLOR_GRAY2BGR)))
    return candidates


def preprocess_for_eticket_ocr(image_bgr):
    """Focus on label band (top ~25%)."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, _w = image_bgr.shape[:2]
    return image_bgr[: max(1, int(h * 0.25)), :]


def _eticket_banner_rois(image_bgr):
    """
    Detect dark horizontal SA/SN banner regions on eticket photos.
    Returns tight ROI list, biggest first.
    """
    rois = []
    if image_bgr is None or image_bgr.size == 0:
        return rois
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Dark text band on white/gray label (e.g. "SA 16074" stripe).
    mask = cv2.inRange(gray, 0, 95)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < int(w * 0.16):
            continue
        if bh < 12 or bh > int(h * 0.22):
            continue
        if bw < (3 * bh):
            continue
        if y < int(h * 0.10) or y > int(h * 0.85):
            continue
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    seen = set()
    for x, y, bw, bh in boxes[:10]:
        key = (x // 10, y // 10, bw // 10, bh // 10)
        if key in seen:
            continue
        seen.add(key)
        # Keep a little vertical context around the strip.
        px = max(4, int(0.03 * bw))
        py = max(6, int(0.35 * bh))
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w, x + bw + px)
        y2 = min(h, y + bh + py)
        roi = image_bgr[y1:y2, x1:x2]
        if roi is not None and roi.size > 0:
            rois.append(roi)
    return rois


def build_eticket_ocr_candidates(image_bgr):
    out = []
    if image_bgr is None or image_bgr.size == 0:
        return out
    for i, roi in enumerate(_eticket_banner_rois(image_bgr), start=1):
        out.append((f"banner_roi_{i}", roi))
        g0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g1 = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(g0)
        up = cv2.resize(g1, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        inv = cv2.bitwise_not(up)
        _ret, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append((f"banner_roi_{i}_otsu", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)))
    h, _w = image_bgr.shape[:2]
    top = image_bgr[: max(1, int(h * 0.25)), :]
    out.append(("top25", top))
    g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gc = clahe.apply(g)
    out.append(("top25_clahe", cv2.cvtColor(gc, cv2.COLOR_GRAY2BGR)))
    up = cv2.resize(top, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out.append(("top25_2x", up))
    inv = cv2.bitwise_not(gc)
    out.append(("top25_inv", cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)))
    return out


def build_eticket_raw_crops(image_bgr):
    crops = []
    if image_bgr is None or image_bgr.size == 0:
        return crops
    for roi in _eticket_banner_rois(image_bgr):
        crops.append(roi)
    h, _w = image_bgr.shape[:2]
    crops.append(image_bgr[: max(1, int(h * 0.25)), :])
    if h > 40:
        crops.append(image_bgr[: max(1, int(h * 0.35)), :])
    t = image_bgr[: max(1, int(h * 0.25)), :]
    if t.size > 0:
        crops.append(imutils.resize(t, width=int(t.shape[1] * 2)))
    return [c for c in crops if c is not None and c.size > 0]


def ocr_eticket_reference(image_bgr):
    # First try the banner-focused extractor (same robust logic as maquette eticket auto-detect).
    try:
        ref0 = extract_reference_maquette_eticket_tesseract(image_bgr)
    except Exception:  # noqa: BLE001
        ref0 = ""
    if ref0:
        return ref0, f"banner_extractor: {ref0}"

    for crop in build_eticket_raw_crops(image_bgr):
        try:
            text = ocr_text_tesseract(crop)
        except Exception:  # noqa: BLE001
            text = ""
        ref = extract_reference_from_text(text)
        if ref:
            return ref, text
    for _name, cand in build_eticket_ocr_candidates(image_bgr):
        try:
            text = ocr_text_tesseract(cand)
        except Exception:  # noqa: BLE001
            text = ""
        ref = extract_reference_from_text(text)
        if ref:
            return ref, text
    return "", ""


def ocr_eticket_reference_fast(image_bgr, timeout_sec=2.0):
    """
    Fast eticket OCR for indexing.
    Uses a few cheap passes only; heavy OCR remains available via deep recheck when needed.
    """
    if image_bgr is None or image_bgr.size == 0:
        return "", ""

    img = downscale_for_ocr(image_bgr.copy(), max_side=960)
    h, _w = img.shape[:2]
    top30 = img[: max(1, int(h * 0.30)), :]
    top40 = img[: max(1, int(h * 0.40)), :]
    mid_band = img[max(0, int(h * 0.24)) : min(h, int(h * 0.62)), :]
    variants = [("top30", top30), ("top40", top40), ("mid62", mid_band), ("full", img)]
    started = time.monotonic()
    trace = []

    def remain():
        return max(0.1, timeout_sec - (time.monotonic() - started))

    # First, read only the dark "SA/SN" banner region when available.
    banner_rois = list(_eticket_banner_rois(img))
    try:
        rect = _eticket_largest_dark_rectangle_roi(img)
    except Exception:
        rect = None
    if rect is not None:
        x, y, bw, bh = rect
        if bw > 20 and bh > 10:
            x1 = max(0, x - 8)
            y1 = max(0, y - 8)
            x2 = min(img.shape[1], x + bw + 8)
            y2 = min(img.shape[0], y + bh + 8)
            roi_rect = img[y1:y2, x1:x2]
            if roi_rect is not None and roi_rect.size > 0:
                banner_rois.insert(0, roi_rect)
    for bidx, roi in enumerate(banner_rois, start=1):
        if remain() <= 0.12:
            break
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(g)
        up = cv2.resize(g, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        inv = cv2.bitwise_not(up)
        _ret, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for nm, v in ((f"banner{bidx}_th", th), (f"banner{bidx}_inv", inv), (f"banner{bidx}_up", up)):
            if remain() <= 0.08:
                break
            for psm in (7, 6, 11):
                if remain() <= 0.08:
                    break
                try:
                    txt = pytesseract.image_to_string(v, config=_tesseract_cfg_psm(psm))
                except Exception:  # noqa: BLE001
                    txt = ""
                if txt:
                    trace.append(f"[{nm}_psm{psm}] {txt}")
                ref = extract_reference_hifi_eac_priority(txt) or extract_reference_from_text(txt) or _eticket_ref_from_ocr_text(txt)
                if ref:
                    return ref, "\n".join(trace)

    # Cheap full/top variants before YOLO so we don't spend all timeout budget there.
    for name, v in variants:
        if remain() <= 0.12:
            break
        gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        for ridx, src in enumerate((clahe, otsu, inv), start=1):
            if remain() <= 0.12:
                break
            rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
            for psm in (7, 11):
                if remain() <= 0.12:
                    break
                txt, timed_out = _tesseract_image_to_string_timeout(
                    rgb, _tesseract_cfg_psm(psm), min(0.9, remain())
                )
                trace.append(f"{name}_v{ridx}_psm{psm}: {(txt or '')[:120]}")
                if timed_out:
                    continue
                ref = extract_reference_hifi_eac_priority(txt)
                if not ref:
                    ref = extract_reference_from_text(txt)
                if not ref:
                    ref = _eticket_ref_from_ocr_text(txt)
                if ref:
                    return ref, "\n".join(trace)

    # YOLO pass only if enough time remains (can be expensive on some machines).
    if remain() > 0.9:
        yolo_rois = _yolo_reference_rois(img, max_boxes=3)
        for ridx, roi in enumerate(yolo_rois, start=1):
            if remain() <= 0.12:
                break
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            for psm in (7, 8, 11):
                if remain() <= 0.12:
                    break
                rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
                txt, timed_out = _tesseract_image_to_string_timeout(
                    rgb, _tesseract_cfg_psm(psm), min(0.8, remain())
                )
                trace.append(f"yolo_roi{ridx}_psm{psm}: {(txt or '')[:120]}")
                if timed_out:
                    continue
                ref = extract_reference_hifi_eac_priority(txt) or extract_reference_from_text(txt) or _eticket_ref_from_ocr_text(txt)
                if ref:
                    return ref, "\n".join(trace)
    return "", "\n".join(trace)


def ocr_filter_reference(image_bgr, deadline=None):
    """
    OCR for filter references with stronger preprocessing.
    Steps:
    1) full-image gray + contrast boost + sharpen
    2) top-40% retry with same preprocessing
    3) fallback to the existing four-attempt pipeline
    """
    if image_bgr is None or image_bgr.size == 0:
        return "", ""

    def pick_ref(text):
        if not text:
            return ""
        fixed = text
        for bad, good in (("O", "0"), ("I", "1"), ("L", "1"), ("B", "8"), ("G", "6"), ("Z", "2"), ("Q", "0")):
            fixed = re.sub(rf"(?<=\d){bad}(?=\d)", good, fixed, flags=re.IGNORECASE)
        cands = _collect_filter_ref_candidates(fixed)
        if cands:
            if KNOWN_REFERENCE_SET:
                # Prefer known references and strong digit-trace evidence.
                scored = []
                digit_blob = re.sub(r"[^0-9]", "", fixed.upper())
                compact = re.sub(r"[^A-Z0-9]", "", fixed.upper())
                for c in cands:
                    sc = 0
                    if c in KNOWN_REFERENCE_SET:
                        sc += 60
                    cc = re.sub(r"[^A-Z0-9]", "", c.upper())
                    if cc and cc in compact:
                        sc += 35
                    _p, d = split_reference(c)
                    if d:
                        sc += int(0.35 * _best_digit_window_fuzz(digit_blob, d))
                    scored.append((sc, c))
                scored.sort(reverse=True)
                best_c = scored[0][1]
                voted = normalize_reference(resolve_filter_ref_with_digit_catalog_vote(fixed, best_c))
                if voted and is_valid_hifi_reference(voted):
                    return voted
                return best_c
            return cands[0]
        # Domain-specific priority: HIFI first, then EAC, then SA/SN reference.
        p = extract_reference_hifi_eac_priority(fixed)
        if p:
            snap = closest_known_reference(p, min_score=92)
            return snap or p
        m = HIFI_GENERIC_REF_PATTERN.search((text or "").upper())
        if m:
            cand = normalize_reference(m.group(0))
            if is_valid_hifi_reference(cand):
                snap = closest_known_reference(cand, min_score=92)
                return snap or cand
        m2 = HIFI_GENERIC_REF_PATTERN.search((fixed or "").upper())
        if m2:
            cand2 = normalize_reference(m2.group(0))
            if is_valid_hifi_reference(cand2):
                snap = closest_known_reference(cand2, min_score=92)
                return snap or cand2
        r = extract_strict_filter_reference(fixed)
        if r:
            snap = closest_known_reference(r, min_score=92)
            return snap or r
        base = extract_reference_from_text(fixed)
        if base:
            snap = closest_known_reference(base, min_score=92)
            return snap or base
        # Noisy recovery last: useful on fragmented curved text, but too risky to trust first.
        noisy = extract_reference_from_very_noisy_text(fixed)
        if noisy:
            return noisy

        # Last chance: if only noisy digits are recognized, map to closest known ref.
        if KNOWN_REFERENCE_SET:
            compact = re.sub(r"[^A-Z0-9]", "", (fixed or "").upper())
            for m3 in re.finditer(r"(SA|SN)?([0-9]{4,6})", compact):
                prefix = (m3.group(1) or "").upper()
                digits = m3.group(2)
                cands = []
                for kr in KNOWN_REFERENCE_SET:
                    kp, kd = split_reference(kr)
                    if not kp or not kd:
                        continue
                    if prefix and kp != prefix:
                        continue
                    if len(kd) != len(digits):
                        continue
                    dr = fuzz.ratio(digits, kd)
                    if dr < 88:
                        continue
                    score = int((0.8 * dr) + (0.2 * fuzz.ratio(compact, kr)))
                    cands.append((score, kr))
                if cands:
                    cands.sort(reverse=True)
                    if cands[0][0] >= 90:
                        return cands[0][1]
        return ""

    def preprocess_for_ocr(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boosted = cv2.convertScaleAbs(gray, alpha=1.7, beta=8)
        sharp = cv2.filter2D(boosted, -1, SHARPEN_KERNEL)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)

    def preprocess_ring_variants(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(clahe, cv2.MORPH_GRADIENT, k)
        return (
            cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB),
        )

    trace = []
    first_ref = ""
    img = downscale_for_ocr(image_bgr.copy())

    def run_attempt(label, rgb, psm):
        nonlocal first_ref
        cfg = _tesseract_cfg_psm(psm)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "", "", True
            # Keep each OCR call short so multiple attempts fit in global timeout.
            call_timeout = max(0.2, min(1.2, remaining))
            txt, timed_out = _tesseract_image_to_string_timeout(rgb, cfg, call_timeout)
        else:
            txt, timed_out = _tesseract_image_to_string_timeout(rgb, cfg, None)
        trace.append(f"{label}: {txt[:300]}")
        if timed_out:
            trace.append(f"{label}: timeout")
            return "", txt, True
        pr = pick_ref(txt)
        if pr and not first_ref:
            first_ref = pr
        return pr, txt, False

    def finalize(primary_guess):
        """Pick best ref from full trace vs first-pass guess (consensus over all OCR lines)."""
        blob = "\n".join(trace)
        pre_consensus = consensus_pick_filter_reference(blob, primary_guess or "")
        if normalize_reference(pre_consensus or "") != normalize_reference(primary_guess or ""):
            trace.append(
                f"consensus_pick: {pre_consensus}"
                + (f" (over first_pass={primary_guess})" if primary_guess else " (from_trace)")
            )
        chosen = resolve_filter_ref_with_digit_catalog_vote(blob, pre_consensus or "")
        if not chosen:
            return (primary_guess or "", blob)
        if normalize_reference(chosen) != normalize_reference(pre_consensus or ""):
            trace.append(f"digit_catalog_vote: {chosen} (over consensus {pre_consensus})")
        return chosen, "\n".join(trace)

    # Attempt 0: YOLOv8 detected ROIs around reference text (if model is present).
    yolo_rois = _yolo_reference_rois(img, max_boxes=YOLO_REF_MAX_BOXES)
    for ridx, roi in enumerate(yolo_rois, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        roi_rgb = preprocess_for_ocr(roi)
        for psm in (7, 8, 11):
            r, _txt, _timed_out = run_attempt(f"yolo_roi{ridx}_psm{psm}", roi_rgb, psm)
            if r:
                return finalize(r)

    # Grass / outdoor: de-emphasize green background before standard passes.
    grass_bgr = _filter_bgr_for_grass_background(img)
    for psm in (7, 8, 11):
        if deadline is not None and time.monotonic() >= deadline:
            break
        _ref, _txt, timed_out = run_attempt(f"grassaware_pre_psm{psm}", preprocess_for_ocr(grass_bgr), psm)
        if timed_out and deadline is not None and time.monotonic() >= deadline:
            return finalize(first_ref)

    # Attempt A: full image preprocessed
    full_pre = preprocess_for_ocr(img)
    for psm in (7, 8, 11):
        if deadline is not None and time.monotonic() >= deadline:
            break
        _ref, _txt, timed_out = run_attempt(f"full_preprocessed_psm{psm}", full_pre, psm)
        if timed_out and deadline is not None and time.monotonic() >= deadline:
            return finalize(first_ref)

    # Attempt B: top 40% preprocessed retry
    h, _w = img.shape[:2]
    top40 = img[: max(1, int(h * 0.4)), :]
    top40_pre = preprocess_for_ocr(top40)
    for psm in (7, 8, 11):
        if deadline is not None and time.monotonic() >= deadline:
            break
        _ref, _txt, timed_out = run_attempt(f"top40_preprocessed_psm{psm}", top40_pre, psm)
        if timed_out and deadline is not None and time.monotonic() >= deadline:
            return finalize(first_ref)

    fr, tr = finalize(first_ref)
    if fr:
        return fr, tr

    # Attempt C: prioritize ring/circular candidates early (common for filter photos).
    ring_candidates = []
    for cname, cand in build_filter_ocr_candidates(img):
        if cname in ("circle_roi", "deskew") or "polar" in str(cname):
            ring_candidates.append((cname, cand))
        if len(ring_candidates) >= 16:
            break
    for cname, cand in ring_candidates:
        if deadline is not None and time.monotonic() >= deadline:
            break
        for ridx, rv in enumerate(preprocess_ring_variants(cand), start=1):
            for rot_name, rot in (
                ("r0", rv),
                ("r90", cv2.rotate(rv, cv2.ROTATE_90_CLOCKWISE)),
                ("r180", cv2.rotate(rv, cv2.ROTATE_180)),
                ("r270", cv2.rotate(rv, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ):
                for psm in (11, 8, 7):
                    r, _txt, _timed_out = run_attempt(f"early_{cname}_v{ridx}_{rot_name}_psm{psm}", rot, psm)
                    if r:
                        return finalize(r)

    # Existing fallback attempts
    for label, rgb, psm in _filter_ocr_four_preprocessed_rgb(img):
        if deadline is not None and time.monotonic() >= deadline:
            break
        r, _txt, _timed_out = run_attempt(label, rgb, psm)
        if r:
            return finalize(r)

    # Extra robust fallback for ring/angled filter photos.
    for cand in preprocess_for_filter_ocr(img):
        if deadline is not None and time.monotonic() >= deadline:
            break
        r, _txt, _timed_out = run_attempt("extra_preprocess_filter", preprocess_for_ocr(cand), 6)
        if r:
            return finalize(r)

    for cname, cand in build_filter_ocr_candidates(img):
        if deadline is not None and time.monotonic() >= deadline:
            break
        rgb_c = preprocess_for_ocr(cand)
        for psm in (11, 8, 6):
            r, _txt, _timed_out = run_attempt(f"extra_{cname}_psm{psm}", rgb_c, psm)
            if r:
                return finalize(r)
    return finalize("")


def normalize_reference(text):
    if not text:
        return ""
    return re.sub(r"\s+", "", text.upper()).strip()


def infer_reference_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0].upper()
    # Ignore copy/index suffixes like "(1)" which can corrupt inferred digits.
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    explicit = re.search(r"\b([A-Z]{2})\s*([0-9]{4,6})\b", name)
    if explicit:
        cand = normalize_reference(explicit.group(1) + explicit.group(2))
        if is_valid_hifi_reference(cand):
            return cand
    compact = re.sub(r"[^A-Z0-9]", "", name)
    for prefix, expected_len in FILTER_PREFIX_LENGTHS.items():
        m = re.search(rf"{prefix}([0-9]{{{expected_len}}})", compact)
        if m:
            return f"{prefix}{m.group(1)}"
    digit_groups = re.findall(r"[0-9]{5,6}", compact)
    if not digit_groups:
        return ""
    digits = digit_groups[-1]
    if len(digits) == 5:
        return f"SA{digits}"
    if len(digits) == 6:
        # Ambiguous when no explicit SA/SN prefix is present.
        # Avoid forcing SN###### from bare digits; prefer nearest known ref
        # only if one is clearly supported by a one-digit deletion.
        if KNOWN_REFERENCE_SET:
            best_ref = ""
            best_sc = -1
            second_sc = -1
            for kr in KNOWN_REFERENCE_SET:
                kp, kd = split_reference(kr)
                if kp != "SA" or len(kd) != 5:
                    continue
                local = 0
                for i in range(len(digits)):
                    cand5 = digits[:i] + digits[i + 1 :]
                    local = max(local, fuzz.ratio(cand5, kd))
                if local > best_sc:
                    second_sc = best_sc
                    best_sc = local
                    best_ref = normalize_reference(kr)
                elif local > second_sc:
                    second_sc = local
            if best_ref and best_sc >= 94 and (best_sc - max(0, second_sc)) >= 4:
                return best_ref
        return ""
    return ""


def authoritative_filename_reference(path):
    """
    When the basename clearly encodes the catalog ref (e.g. 17088.jpg -> SA17088), that value
    is the long-term ground truth for matching — curved-cap OCR must not override it.
    Returns normalized ref or "".
    """
    if not path:
        return ""
    inf = infer_reference_from_filename(path)
    if not inf or not is_valid_hifi_reference(inf):
        return ""
    # Special case: 6-digit bare filenames are usually ambiguous, but if inference
    # resolved to a known catalog reference, treat it as authoritative.
    base = os.path.splitext(os.path.basename(path or ""))[0].upper()
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    compact = re.sub(r"[^A-Z0-9]", "", base)
    if re.fullmatch(r"[0-9]{6}", compact) and inf in KNOWN_REFERENCE_SET:
        return normalize_reference(inf)
    if not filename_strong_digit_reference_hint(path):
        return ""
    return normalize_reference(inf)


def make_filter_cache_key(path, mtime):
    return f"{path}|{mtime}|{FILTER_OCR_CACHE_VERSION}"


def split_reference(ref):
    if not ref:
        return "", ""
    match = re.match(r"^([A-Z]{2,3})([0-9]{5,6})$", ref)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def reference_difference_count(ref_a, ref_b):
    a = normalize_reference(ref_a or "")
    b = normalize_reference(ref_b or "")
    n = max(len(a), len(b))
    if n == 0:
        return 0
    diff = 0
    for i in range(n):
        ca = a[i] if i < len(a) else ""
        cb = b[i] if i < len(b) else ""
        if ca != cb:
            diff += 1
    return diff


def eticket_compatibility_vs_filter(filter_ref, eticket_ref):
    """Text-only similarity for ranking; 0 if SA/SN prefix differs."""
    fn = normalize_reference(filter_ref)
    if not fn:
        return 0
    f_prefix, f_digits = split_reference(fn)
    en = normalize_reference(eticket_ref or "")
    e_prefix, e_digits = split_reference(en)
    if not e_prefix or not e_digits:
        return 0
    if e_prefix != f_prefix:
        return 0
    digits_score = fuzz.ratio(f_digits, e_digits)
    # Stricter digit gate: different part numbers must not score like a near-match (final project safety).
    if digits_score < 88:
        return int(0.32 * digits_score)
    base = fuzz.ratio(fn, en)
    return int((0.48 * 100) + (0.42 * digits_score) + (0.10 * base))


def is_valid_hifi_reference(ref):
    prefix, digits = split_reference(ref)
    expected = FILTER_PREFIX_LENGTHS.get(prefix)
    return bool(expected and len(digits) == expected)


def _best_digit_window_fuzz(digit_blob, kd):
    """Best fuzzy match of a digit string anywhere inside the OCR digit stream (handles curved / noisy OCR)."""
    if not digit_blob or not kd:
        return 0
    L = len(kd)
    if len(digit_blob) < L:
        return 0
    return max(
        fuzz.ratio(digit_blob[i : i + L], kd) for i in range(0, len(digit_blob) - L + 1)
    )


def resolve_filter_ref_with_digit_catalog_vote(raw_text, candidate_ref):
    """
    Pick the catalog SA/SN whose digits best match the digit stream in the full OCR trace.
    Fixes wrong SA snaps (e.g. SA14074) when the trace actually supports SN920410 on curved labels.
    """
    cand = normalize_reference(candidate_ref or "")
    if not raw_text:
        return cand
    compact = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if KNOWN_REFERENCE_SET:
        for known in KNOWN_REFERENCE_SET:
            kn = normalize_reference(known)
            if kn and len(kn) >= 7 and kn in compact:
                return kn
    if not KNOWN_REFERENCE_SET:
        return cand

    digit_blob = re.sub(r"[^0-9]", "", raw_text.upper())
    if len(digit_blob) < 4:
        return cand

    cand_prefix, _cand_digits = split_reference(cand) if cand else ("", "")
    scored = []
    for known in KNOWN_REFERENCE_SET:
        kp, kd = split_reference(known)
        if not kd or len(kd) not in (5, 6):
            continue
        # Do not let noisy catalog voting flip SA <-> SN when OCR already produced a valid prefix.
        if cand_prefix and kp and kp != cand_prefix:
            continue
        sc = _best_digit_window_fuzz(digit_blob, kd)
        kn = normalize_reference(known)
        if kn in compact:
            sc = max(sc, 98)
        elif kd in digit_blob:
            sc = max(sc, 96)
        scored.append((sc, known))

    if not scored:
        return cand

    scored.sort(key=lambda x: x[0], reverse=True)
    top_sc, top_k = scored[0][0], scored[0][1]
    second_sc = scored[1][0] if len(scored) > 1 else -1
    top_kn = normalize_reference(top_k)

    cand_sc = 0.0
    if cand and is_valid_hifi_reference(cand):
        _cp, cd = split_reference(cand)
        if cd:
            cand_sc = _best_digit_window_fuzz(digit_blob, cd)

    if top_sc < 84:
        return cand

    if second_sc >= 0 and top_sc - second_sc < 4 and top_sc < 93:
        return cand if cand else normalize_reference(top_k)

    # If OCR candidate is unknown but top known ref has good digit evidence,
    # prefer known catalog value to avoid drifting to non-existing references.
    if cand and cand not in KNOWN_REFERENCE_SET and top_sc >= 82:
        tp, _td = split_reference(top_kn)
        cp, _cd = split_reference(cand)
        if tp and cp and tp == cp:
            if top_sc >= cand_sc + 6 or cand_sc <= 76:
                return top_kn

    if top_sc >= 93:
        return top_kn
    if top_sc >= 88 and top_sc >= cand_sc + 8:
        return top_kn
    if top_sc >= 86 and cand_sc <= 72:
        return top_kn
    if not cand and top_sc >= 86:
        return top_kn
    return cand


def correct_reference_parts(prefix_raw, digits_raw, allowed_prefixes=None):
    letter_map = str.maketrans({
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    })
    digit_map = str.maketrans({
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
    })
    prefix = prefix_raw.translate(letter_map)
    digits = digits_raw.translate(digit_map)
    if not prefix.isalpha() or not digits.isdigit():
        return ""
    if allowed_prefixes and prefix not in allowed_prefixes:
        return ""
    candidate = normalize_reference(prefix + digits)
    if allowed_prefixes == set(FILTER_PREFIX_LENGTHS.keys()) and not is_valid_hifi_reference(candidate):
        return ""
    return candidate


def load_image_bgr(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".heic":
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Maquette Eticket "Auto Detect Ref" â€” HSV dark banner + Tesseract (see _auto_detect_ref_task).
_ETICKET_BANNER_PAD_PX = 20


def _eticket_correct_chars_between_digits(s):
    """O / I / l / B fixes when sandwiched between digits (before regex)."""
    if not s:
        return s
    chars = list(s)
    mapping = {"O": "0", "I": "1", "l": "1", "B": "8", "G": "6", "Z": "2"}
    for i, c in enumerate(chars):
        if c not in mapping:
            continue
        left = i > 0 and chars[i - 1].isdigit()
        right = i < len(chars) - 1 and chars[i + 1].isdigit()
        if left and right:
            chars[i] = mapping[c]
    return "".join(chars)


def _eticket_ref_from_ocr_text(text):
    if not text:
        return ""
    fixed = _eticket_correct_chars_between_digits(text)
    for candidate in (fixed, re.sub(r"\s+", "", fixed.upper())):
        m = SA_SN_REF_REGEX.search(candidate)
        if not m:
            continue
        ref = correct_reference_parts(
            m.group(1).upper(),
            m.group(2),
            allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
        )
        if ref:
            return ref
    return ""


def _strict_refs_from_raw_text(raw_text):
    """Extract SA/SN refs from noisy OCR text with digit/letter correction."""
    out = []
    if not raw_text:
        return out
    upper = (raw_text or "").upper()
    compact = re.sub(r"[^A-Z0-9]", "", upper)
    patterns = [upper, compact]
    expected_lengths = set(FILTER_PREFIX_LENGTHS.values())
    for src in patterns:
        for m in re.finditer(r"(SA|SN)\s*([A-Z0-9]{4,8})", src):
            prefix = (m.group(1) or "").upper()
            tail = re.sub(r"[^A-Z0-9]", "", m.group(2) or "")
            if not tail:
                continue
            target_len = FILTER_PREFIX_LENGTHS.get(prefix, 0)
            if target_len <= 0:
                continue
            windows = [tail]
            if len(tail) > target_len:
                windows = [tail[i : i + target_len] for i in range(0, len(tail) - target_len + 1)]
            for wd in windows:
                if len(wd) not in expected_lengths:
                    continue
                fixed = correct_reference_parts(prefix, wd, allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()))
                if fixed and fixed not in out:
                    out.append(fixed)
    return out


def _eticket_largest_dark_rectangle_roi(image_bgr):
    """Find largest dark wide rectangle: threshold<50 + contour shape filtering."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _ret, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 100:
            continue
        if w < (2 * h):
            continue
        area = float(w * h)
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def _eticket_ref_from_sparse_words(text):
    """Try each word and each consecutive word pair (PSM 11 fallback)."""
    words = re.findall(r"[A-Za-z0-9]+", text or "")
    if not words:
        return ""
    for w in words:
        r = _eticket_ref_from_ocr_text(w)
        if r:
            return r
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i + 1]}"
        r = _eticket_ref_from_ocr_text(pair)
        if r:
            return r
    return ""


def extract_reference_maquette_eticket_tesseract(image_bgr):
    """
    Step 1-2: dark banner contour -> crop + 4x + invert + Otsu + PSM7.
    Step 3 fallback: full image PSM11 and scan words + word pairs.
    No UI here (manual dialog handled by caller).
    """
    if image_bgr is None or image_bgr.size == 0:
        return ""
    ih, iw = image_bgr.shape[:2]
    candidates = []
    rect = _eticket_largest_dark_rectangle_roi(image_bgr)
    if rect is not None:
        candidates.append(rect)

    # Additional dark-band candidates for label photos (captures SA/SN banner reliably).
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _ret, mask2 = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, k)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 120:
            continue
        if w < (2 * h):
            continue
        if h > int(ih * 0.35):
            continue
        if y > int(ih * 0.85):
            continue
        candidates.append((x, y, w, h))

    # Try biggest candidates first.
    candidates = sorted(candidates, key=lambda r: r[2] * r[3], reverse=True)
    seen = set()
    for x, y, rw, rh in candidates[:12]:
        key = (x // 8, y // 8, rw // 8, rh // 8)
        if key in seen:
            continue
        seen.add(key)
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + rw + pad)
        y2 = min(ih, y + rh + pad)
        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        scaled = cv2.resize(roi, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(g)
        _ret, otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for img in (otsu, inv):
            for cfg in (_tesseract_cfg_psm(7), _tesseract_cfg_psm(6), "--oem 1 --psm 7"):
                try:
                    txt = pytesseract.image_to_string(img, config=cfg)
                except Exception:  # noqa: BLE001
                    txt = ""
                ref = _eticket_ref_from_ocr_text(txt)
                if ref:
                    return ref
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    inv_full = cv2.bitwise_not(gray_full)
    _ret, otsu_full = cv2.threshold(inv_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    full_variants = (gray_full, inv_full, otsu_full)
    full_cfgs = (_tesseract_cfg_psm(11), "--oem 1 --psm 11", _tesseract_cfg_psm(6), "--oem 1 --psm 6")
    for fv in full_variants:
        for cfg in full_cfgs:
            try:
                full_txt = pytesseract.image_to_string(fv, config=cfg)
            except Exception:  # noqa: BLE001
                full_txt = ""
            ref = _eticket_ref_from_sparse_words(full_txt)
            if ref:
                return ref
    return ""


def extract_eticket_serial_number(image_bgr):
    """Extract barcode/serial number digits from lower eticket area."""
    if image_bgr is None or image_bgr.size == 0:
        return ""
    h, w = image_bgr.shape[:2]
    # Serial (N° Gencod) is typically printed in lower third, below barcode.
    lower = image_bgr[max(0, int(h * 0.58)) : min(h, int(h * 0.99)), :]
    if lower is None or lower.size == 0:
        lower = image_bgr
    gh, gw = lower.shape[:2]
    # Prefer the gencod text strip where the clean serial is printed.
    g0 = max(0, int(gh * 0.62))
    g1 = min(gh, int(gh * 0.98))
    x0 = max(0, int(gw * 0.05))
    x1 = min(gw, int(gw * 0.95))
    gencod_band = lower[g0:g1, x0:x1]
    if gencod_band is None or gencod_band.size == 0:
        gencod_band = lower
    variants = [gencod_band, lower]
    try:
        variants.append(cv2.resize(gencod_band, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.resize(lower, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC))
    except Exception:
        pass
    # 1) Prefer true barcode decode when available (more reliable than OCR).
    if pyzbar_decode is not None:
        decoded_scores = {}
        for src in variants:
            if src is None or src.size == 0:
                continue
            try:
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = None
            for candidate in (src, gray):
                if candidate is None:
                    continue
                try:
                    items = pyzbar_decode(candidate)
                except Exception:
                    items = []
                for it in items or []:
                    try:
                        data = (it.data or b"").decode("utf-8", errors="ignore")
                    except Exception:
                        data = ""
                    digits = re.sub(r"[^0-9]", "", data or "")
                    if 10 <= len(digits) <= 16:
                        decoded_scores[digits] = decoded_scores.get(digits, 0) + (10 if len(digits) == 13 else 6)
        if decoded_scores:
            ranked_dec = sorted(
                decoded_scores.items(),
                key=lambda kv: (kv[1], 1 if len(kv[0]) == 13 else 0, len(kv[0])),
                reverse=True,
            )
            return ranked_dec[0][0]
    # 2) OpenCV barcode detector fallback (if present in installed OpenCV).
    try:
        detector = cv2.barcode_BarcodeDetector()
    except Exception:
        detector = None
    if detector is not None:
        opencv_scores = {}
        for src in variants:
            if src is None or src.size == 0:
                continue
            try:
                ok, decoded_infos, _decoded_type, _corners = detector.detectAndDecode(src)
            except Exception:
                ok, decoded_infos = False, []
            if not ok:
                continue
            for info in decoded_infos or []:
                digits = re.sub(r"[^0-9]", "", str(info or ""))
                if 10 <= len(digits) <= 16:
                    opencv_scores[digits] = opencv_scores.get(digits, 0) + (9 if len(digits) == 13 else 5)
        if opencv_scores:
            ranked_cv = sorted(
                opencv_scores.items(),
                key=lambda kv: (kv[1], 1 if len(kv[0]) == 13 else 0, len(kv[0])),
                reverse=True,
            )
            return ranked_cv[0][0]
    # 3) OCR fallback.
    scores = {}
    for src in variants:
        if src is None or src.size == 0:
            continue
        try:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
            _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            proc_images = (clahe, otsu, cv2.bitwise_not(otsu))
        except Exception:
            proc_images = ()
        for img in proc_images:
            for psm in (6, 7, 11):
                try:
                    txt = pytesseract.image_to_string(
                        img,
                        config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789",
                    )
                except Exception:
                    txt = ""
                if not txt:
                    continue
                # Keep sequences even if OCR inserted spaces.
                compact = re.sub(r"[^0-9]", "", txt)
                if 10 <= len(compact) <= 16:
                    scores[compact] = scores.get(compact, 0) + (4 if len(compact) >= 13 else 2)
                # Rebuild split barcode lines such as: "3 661200 113494" -> "3661200113494"
                for ln in txt.splitlines():
                    parts = re.findall(r"\d+", ln or "")
                    if len(parts) >= 2:
                        joined = "".join(parts)
                        if 10 <= len(joined) <= 16:
                            bonus = 6 if len(joined) == 13 else 3
                            scores[joined] = scores.get(joined, 0) + bonus
                for m in SERIAL_NUMBER_PATTERN.finditer(txt):
                    val = (m.group(0) or "").strip()
                    if val:
                        scores[val] = scores.get(val, 0) + 3
                # Focus exact 13-digit candidates when present.
                for m13 in re.finditer(r"(?<!\d)(\d{13})(?!\d)", compact):
                    v13 = m13.group(1)
                    if v13:
                        scores[v13] = scores.get(v13, 0) + 8
    if not scores:
        return ""
    # Pick highest vote, prefer exact 13-digit barcode serials.
    ranked = sorted(
        scores.items(),
        key=lambda kv: (kv[1], 1 if len(kv[0]) == 13 else 0, len(kv[0])),
        reverse=True,
    )
    return ranked[0][0]


def extract_eticket_serial_number_step4_strict(image_bgr):
    """
    Step-4 strict serial extraction:
    read only the barcode-digit strip (N° Gencod area), ignore all other text.
    Primary ROI is bottom 18% of image, with bottom 25% fallback.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ""
    try:
        hh, ww = image_bgr.shape[:2]
        mx = max(hh, ww)
        if mx > 2200:
            sc = 2200.0 / float(mx)
            image_bgr = cv2.resize(image_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    t0 = time.monotonic()
    deadline = t0 + 5.0
    h, _w = image_bgr.shape[:2]
    roi_list = [
        image_bgr[max(0, int(h * 0.82)) : h, :],  # primary bottom 18%
        image_bgr[max(0, int(h * 0.75)) : h, :],  # fallback bottom 25%
        image_bgr[max(0, int(h * 0.70)) : h, :],  # tolerant fallback
        image_bgr[max(0, int(h * 0.65)) : h, :],  # extra fallback
    ]
    focused_rois = []
    for roi in roi_list:
        if roi is None or roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        x0 = max(0, int(rw * 0.03))
        x1 = min(rw, int(rw * 0.97))
        for frac in (0.35, 0.45, 0.55, 0.65):
            y0 = max(0, int(rh * frac))
            band = roi[y0:rh, x0:x1]
            if band is not None and band.size > 0:
                focused_rois.append(band)
            band_full = roi[y0:rh, :]
            if band_full is not None and band_full.size > 0:
                focused_rois.append(band_full)
        # Per-ROI fallback: keep a broad band when strip proposals are weak.
        if rh > 8:
            broad = roi[max(0, int(rh * 0.55)) : rh, :]
            if broad is not None and broad.size > 0:
                focused_rois.append(broad)

    scores = {}

    def _ingest_ocr_text(txt, weight=3):
        if not txt:
            return False
        tokens = re.findall(r"\d+", txt)
        if not tokens:
            return False
        joined = "".join(tokens)
        candidates = []
        if len(joined) == 13:
            candidates.append(joined)
        elif len(joined) > 13:
            for i in range(0, len(joined) - 13 + 1):
                candidates.append(joined[i : i + 13])
        compact = re.sub(r"[^0-9]", "", txt)
        if len(compact) >= 13:
            for i in range(0, len(compact) - 13 + 1):
                candidates.append(compact[i : i + 13])
        hit = False
        for cand in candidates:
            if len(cand) != 13 or (not cand.isdigit()):
                continue
            sc = scores.get(cand, 0) + int(weight)
            if _ean13_checksum_ok(cand):
                sc += 10
                hit = True
            scores[cand] = sc
        return hit

    # 1) Fast tiny-screenshot line OCR first.
    try:
        line = image_bgr[max(0, int(h * 0.70)) : h, :]
        if line is not None and line.size > 0:
            for scale in (7.0, 9.0):
                if time.monotonic() > deadline:
                    break
                up = cv2.resize(line, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                g = cv2.GaussianBlur(g, (3, 3), 0)
                c = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(g)
                _ret, o = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                for src, psm in ((c, 7), (o, 7), (cv2.bitwise_not(o), 6)):
                    if time.monotonic() > deadline:
                        break
                    try:
                        t = pytesseract.image_to_string(
                            src,
                            config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789",
                            timeout=1.2,
                        )
                    except Exception:
                        t = ""
                    if _ingest_ocr_text(t, weight=6):
                        valid_fast = [(k, v) for k, v in scores.items() if _ean13_checksum_ok(k)]
                        if valid_fast:
                            valid_fast.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
                            return valid_fast[0][0]
    except Exception:
        pass

    # 2) Normal strict OCR on focused bottom ROIs with hard time budget.
    for roi in focused_rois:
        if time.monotonic() > deadline:
            break
        if roi is None or roi.size == 0:
            continue
        try:
            up4 = cv2.resize(roi, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            up5 = cv2.resize(roi, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
        except Exception:
            up4, up5 = None, None
        for up in (up4, up5):
            if time.monotonic() > deadline:
                break
            if up is None or up.size == 0:
                continue
            try:
                gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
                _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                proc = (clahe, otsu, cv2.bitwise_not(otsu))
            except Exception:
                proc = ()
            for img in proc:
                if time.monotonic() > deadline:
                    break
                for psm in (7, 6):
                    if time.monotonic() > deadline:
                        break
                    try:
                        txt = pytesseract.image_to_string(
                            img,
                            config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789",
                            timeout=1.2,
                        )
                    except Exception:
                        txt = ""
                    _ingest_ocr_text(txt, weight=3)
    valid = [(code, sc) for code, sc in scores.items() if _ean13_checksum_ok(code)]
    if not valid:
        # Last strict fallback: use debug OCR digits, but only accept strong GS1-like valid EAN13.
        try:
            dbg = (extract_eticket_serial_number_step4_debug_digits(image_bgr) or "").strip()
        except Exception:
            dbg = ""
        if dbg:
            digits = re.sub(r"[^0-9]", "", dbg)
            cands = set()
            if len(digits) == 13:
                cands.add(digits)
                cands.add("3" + digits[1:])
            elif len(digits) == 12:
                cands.add("3" + digits)
            elif len(digits) > 13:
                for i in range(0, len(digits) - 13 + 1):
                    cands.add(digits[i : i + 13])
            valid_dbg = [c for c in cands if len(c) == 13 and c.isdigit() and _ean13_checksum_ok(c)]
            # Keep only the expected barcode family to avoid wrong random numbers.
            valid_dbg = [c for c in valid_dbg if c.startswith("366120")]
            if valid_dbg:
                valid_dbg.sort(key=lambda c: (fuzz.ratio(c, digits[:13]), c), reverse=True)
                return valid_dbg[0]
        # Final fallback: generic serial reader (includes barcode decode) with strict gates.
        try:
            generic = (extract_eticket_serial_number(image_bgr) or "").strip()
        except Exception:
            generic = ""
        if (
            generic
            and len(generic) == 13
            and generic.isdigit()
            and _ean13_checksum_ok(generic)
            and generic.startswith("366120")
        ):
            return generic
        return ""
    valid.sort(key=lambda kv: (kv[1], kv[0]), reverse=True)
    return valid[0][0]


def extract_eticket_serial_number_step4_debug_digits(image_bgr):
    """
    Debug helper for manual "Test Serial Photo" only.
    Returns best OCR digit hypothesis from barcode-number zone without checksum gating.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ""
    try:
        hh, ww = image_bgr.shape[:2]
        mx = max(hh, ww)
        if mx > 1800:
            sc = 1800.0 / float(mx)
            image_bgr = cv2.resize(image_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    try:
        h, _w = image_bgr.shape[:2]
    except Exception:
        return ""
    t0 = time.monotonic()
    deadline = t0 + 3.0
    rois = [
        image_bgr[max(0, int(h * 0.60)) : h, :],
        image_bgr[max(0, int(h * 0.68)) : h, :],
        image_bgr[max(0, int(h * 0.75)) : h, :],
    ]
    candidates = {}
    for roi in rois:
        if time.monotonic() > deadline:
            break
        if roi is None or roi.size == 0:
            continue
        for scale in (6.0, 8.0, 10.0):
            if time.monotonic() > deadline:
                break
            try:
                up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(gray)
                _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                proc = (clahe, otsu, cv2.bitwise_not(otsu))
            except Exception:
                proc = ()
            for img in proc:
                if time.monotonic() > deadline:
                    break
                for psm in (6, 7, 11):
                    if time.monotonic() > deadline:
                        break
                    try:
                        txt = pytesseract.image_to_string(
                            img,
                            config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789",
                            timeout=1.2,
                        )
                    except Exception:
                        txt = ""
                    if not txt:
                        continue
                    d = re.sub(r"[^0-9]", "", txt)
                    if len(d) < 6:
                        continue
                    for i in range(0, max(1, len(d) - 13 + 1)):
                        w = d[i : i + 13] if len(d) >= 13 else d
                        if not w:
                            continue
                        score = candidates.get(w, 0) + 1
                        if len(w) == 13:
                            score += 2
                        if w.startswith("3"):
                            score += 1
                        candidates[w] = score
    if not candidates:
        return ""
    ranked = sorted(candidates.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    best = ranked[0][0]
    if len(best) > 13:
        return best[:13]
    return best


class HifiMatcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HIFI Filter to Eticket Matcher")
        self.root.geometry("1200x760")

        self.filters_folder = ""
        self.etickets_folder = ""
        self.filter_images = []
        self.eticket_index = []
        self.filter_photo = None
        self.eticket_photo = None
        self.client_filter_photo = None
        self.client_eticket_photo = None
        self.client_maquette_eticket_photo = None
        self.client_maquette_filtre_photo = None
        self.client_visual_match_photo = None
        self.filter_cache = {}
        self.visual_score_cache = {}
        self.is_busy = False
        self.client_verify_in_progress = False
        self.auto_detect_eticket_busy = False
        self.auto_detect_filtre_busy = False
        self.auto_detect_serial_busy = False
        self.auto_detect_for_save_busy = False
        self.selected_client_id = None
        self.clients = []
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clients.db")
        self.matcher_window = None
        self.client_window = None
        self.known_references = set()
        self.camera_filter_path = ""
        self.camera_capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_captures")
        self.live_camera_mode = False
        self.live_camera_job = None
        self.live_camera_cap = None
        self.live_last_match_ts = 0.0
        self.live_ref_streak_ref = ""
        self.live_ref_streak_count = 0
        # Live camera: be patient before treating SA/SN as stable (seconds between strict OCR, settle time, streak count).
        self.live_strict_ocr_interval_sec = 6.0
        self.live_streak_required = 6
        self.live_strict_ready_ts = 0.0
        self.live_overlay_interval_sec = 0.55
        self.live_overlay_confirm_needed = 1
        self.live_overlay_last_ts = 0.0
        self.live_overlay_cache = None
        self.live_frame_save_interval_sec = 1.1
        self.live_frame_last_save_ts = 0.0
        self.live_ui_interval_ms = 33
        self.live_preview_max_side = 860
        self.live_strict_busy = False
        self.live_strict_worker = None
        self.live_forced_ref = ""
        self.live_forced_ref_ts = 0.0
        self.live_last_presence = {}
        self._auto_restart_job = None
        self._auto_restart_session = 0
        self._was_live_when_match_started = False
        self._auto_restart_base_msg = ""
        self._auto_restart_tone = "neutral"
        # After a live-originated mismatch: skip full OCR overlay redraw until a match (or live restarts).
        self._live_overlay_freeze_after_mismatch = False
        # Suppress repeated strict-OCR find_match spam for the same wrong ref while frozen.
        self._live_last_mismatch_blocked_ref = ""
        self._live_last_mismatch_mono_ts = 0.0
        # After live mismatch: keep result panel until user stops and starts Live Camera again (no auto-restart, no new find_match).
        self._live_hold_mismatch_until_camera_restart = False
        # Matcher: user-selected etiquette image to compare against live filter reads.
        self.matcher_selected_eticket_path = ""
        self.matcher_selected_eticket_ref = ""
        self.matcher_selected_eticket_serial_path = ""
        # Cache expensive presence OCR profiles (path/mtime keyed) for fast repeated live results.
        self.presence_profile_cache = {}
        # Cache strict Step-4 serial OCR (path/mtime keyed) to avoid repeated heavy OCR.
        self.step4_serial_cache = {}
        self.dashboard_feed_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "dashboard",
            "data",
            "exceptions.json",
        )
        self._ensure_dashboard_feed_file()

        self._init_db()
        self._build_ui()
        self.set_global_status("OCR Ready âœ…")

    def _configure_camera_capture(self, cap):
        if cap is None:
            return
        # Keep camera feed lightweight for real-time OCR responsiveness.
        try:
            # Slightly higher source detail helps curved label text OCR.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _refresh_known_references(self):
        refs = set()
        for row in self.clients:
            try:
                r = normalize_reference(row[4])
            except Exception:  # noqa: BLE001
                r = ""
            if r and is_valid_hifi_reference(r):
                refs.add(r)
        for item in self.eticket_index:
            r = normalize_reference(item.get("reference", ""))
            if r and is_valid_hifi_reference(r):
                refs.add(r)
        self.known_references = refs
        global KNOWN_REFERENCE_SET
        KNOWN_REFERENCE_SET = set(refs)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    maquette_eticket_path TEXT NOT NULL DEFAULT '',
                    maquette_filtre_path TEXT NOT NULL DEFAULT '',
                    filter_reference_code TEXT NOT NULL,
                    maquette_eticket_serial_code TEXT NOT NULL DEFAULT '',
                    date_added TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_name_nocase ON clients(LOWER(name))"
            )
            self._migrate_clients_schema(conn)
            conn.commit()
        finally:
            conn.close()

    def _migrate_clients_schema(self, conn):
        """Add maquette_filtre_path and rename maquette_photo_path -> maquette_eticket_path without losing data."""
        cur = conn.execute("PRAGMA table_info(clients)")
        cols = {row[1] for row in cur.fetchall()}
        if "maquette_eticket_path" not in cols and "maquette_photo_path" in cols:
            try:
                conn.execute("ALTER TABLE clients RENAME COLUMN maquette_photo_path TO maquette_eticket_path")
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS clients_mig (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        maquette_eticket_path TEXT NOT NULL DEFAULT '',
                        maquette_filtre_path TEXT NOT NULL DEFAULT '',
                        filter_reference_code TEXT NOT NULL,
                        maquette_eticket_serial_code TEXT NOT NULL DEFAULT '',
                        date_added TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO clients_mig (id, name, maquette_eticket_path, maquette_filtre_path, filter_reference_code, maquette_eticket_serial_code, date_added)
                    SELECT id, name, maquette_photo_path, '', filter_reference_code, '', date_added FROM clients
                    """
                )
                conn.execute("DROP TABLE clients")
                conn.execute("ALTER TABLE clients_mig RENAME TO clients")
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_clients_name_nocase ON clients(LOWER(name))"
                )
        cur = conn.execute("PRAGMA table_info(clients)")
        cols = {row[1] for row in cur.fetchall()}
        if "maquette_filtre_path" not in cols:
            conn.execute("ALTER TABLE clients ADD COLUMN maquette_filtre_path TEXT NOT NULL DEFAULT ''")
        if "maquette_eticket_serial_code" not in cols:
            conn.execute("ALTER TABLE clients ADD COLUMN maquette_eticket_serial_code TEXT NOT NULL DEFAULT ''")

    def _build_ui(self):
        self.root.title("HIFI Filter to Eticket Matcher")
        self.global_status_var = tk.StringVar(value="OCR Ready âœ…")
        self.root.withdraw()
        self.open_matcher_window()
        self.open_client_window()

    def _center_window(self, window, width, height):
        sw = window.winfo_screenwidth()
        sh = window.winfo_screenheight()
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def _on_close_matcher_window(self):
        stop_live_ocr_preview()
        if self.matcher_window is not None and self.matcher_window.winfo_exists():
            self.matcher_window.destroy()
            self.matcher_window = None

    def _on_close_client_window(self):
        if self.client_window is not None and self.client_window.winfo_exists():
            self.client_window.destroy()
            self.client_window = None

    def _clear_stale_matcher_status_if_idle(self):
        """Drop leftover live-match text when reopening Matcher with camera off (not a real new result)."""
        if getattr(self, "live_camera_mode", False) or getattr(self, "is_busy", False):
            return
        st = getattr(self, "status_text", None)
        if st is None:
            return
        try:
            if not st.winfo_exists():
                return
            blob = st.get("1.0", "end-1c")
        except Exception:
            return
        if "Filter reference (live):" in blob:
            self.set_status("Select both folders to start.", tone="neutral")

    def open_matcher_window(self):
        if self.matcher_window is not None and self.matcher_window.winfo_exists():
            self.matcher_window.deiconify()
            self.matcher_window.lift()
            try:
                self._clear_stale_matcher_status_if_idle()
            except Exception:
                pass
            return
        self.matcher_window = tk.Toplevel(self.root)
        self.matcher_window.title("HIFI Filter to Eticket Matcher - Matcher")
        self.matcher_window.geometry("1200x760")
        self.matcher_window.protocol("WM_DELETE_WINDOW", self._on_close_matcher_window)
        self._build_matcher_ui(self.matcher_window)

    def _build_matcher_ui(self, parent):
        top_frame = ttk.Frame(parent, padding=10)
        top_frame.pack(fill=tk.X)

        self.select_filters_btn = ttk.Button(top_frame, text="Select Filters Folder", command=self.select_filters_folder)
        self.select_filters_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.filters_label = ttk.Label(top_frame, text="No filters folder selected")
        self.filters_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        self.select_etickets_btn = ttk.Button(top_frame, text="Select Etickets Folder", command=self.select_etickets_folder)
        self.select_etickets_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.etickets_label = ttk.Label(top_frame, text="No etickets folder selected")
        self.etickets_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        self.refresh_eticket_cache_btn = ttk.Button(
            top_frame, text="Refresh Cache", command=self.refresh_eticket_cache
        )
        self.refresh_eticket_cache_btn.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        selector_frame = ttk.Frame(parent, padding=10)
        selector_frame.pack(fill=tk.X)

        ttk.Label(selector_frame, text="Filter Image:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.filter_selector = ttk.Combobox(selector_frame, state="readonly", width=60)
        self.filter_selector.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.filter_selector.bind("<<ComboboxSelected>>", self.on_filter_selected)
        self.capture_cam_btn = ttk.Button(
            selector_frame,
            text="Capture Camera",
            command=self.capture_filter_from_camera,
        )
        self.capture_cam_btn.grid(row=0, column=2, padx=5)
        self.live_cam_btn = ttk.Button(
            selector_frame,
            text="Live Camera Auto",
            command=self.toggle_live_camera_mode,
        )
        self.live_cam_btn.grid(row=0, column=3, padx=5)
        self.live_ocr_btn = ttk.Button(
            selector_frame,
            text="Live OCR View",
            command=lambda: start_live_ocr_preview(parent, self),
        )
        self.live_ocr_btn.grid(row=0, column=4, padx=5)
        self.find_btn = ttk.Button(selector_frame, text="Find Match", command=self.find_match)
        self.find_btn.grid(row=0, column=5, padx=8)
        self.reset_btn = ttk.Button(selector_frame, text="Reset", command=self.reset)
        self.reset_btn.grid(row=0, column=6, padx=8)
        self.dashboard_btn = ttk.Button(
            selector_frame,
            text="Open QC Dashboard (local)",
            command=self.open_local_qc_dashboard,
        )
        self.dashboard_btn.grid(row=0, column=7, padx=8)

        ttk.Label(selector_frame, text="Eticket (compare target):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=(8, 0))
        self.eticket_selector = ttk.Combobox(selector_frame, state="readonly", width=60)
        self.eticket_selector.grid(row=1, column=1, padx=5, sticky=tk.W, pady=(8, 0))
        self.eticket_selector.bind("<<ComboboxSelected>>", self.on_matcher_eticket_selected)
        ttk.Label(
            selector_frame,
            text="Pick label → green OCR boxes; live camera compares filter ref to this.",
        ).grid(row=1, column=2, columnspan=5, sticky=tk.W, padx=8, pady=(8, 0))
        ttk.Label(selector_frame, text="Eticket serial photo (optional):").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=(8, 0)
        )
        self.matcher_serial_photo_var = tk.StringVar(value="")
        ttk.Entry(
            selector_frame,
            textvariable=self.matcher_serial_photo_var,
            width=60,
            state="readonly",
        ).grid(row=2, column=1, padx=5, sticky=tk.W, pady=(8, 0))
        self.browse_serial_photo_btn = ttk.Button(
            selector_frame,
            text="Browse Serial Photo",
            command=self.browse_matcher_serial_photo,
        )
        self.browse_serial_photo_btn.grid(row=2, column=2, padx=5, pady=(8, 0), sticky=tk.W)
        self.test_serial_photo_btn = ttk.Button(
            selector_frame,
            text="Test Serial Photo",
            command=self.test_matcher_serial_photo_now,
        )
        self.test_serial_photo_btn.grid(row=2, column=3, padx=5, pady=(8, 0), sticky=tk.W)

        manual_frame = ttk.Frame(parent, padding=10)
        manual_frame.pack(fill=tk.X)
        ttk.Label(manual_frame, text="Manual Filter Reference (optional):").grid(row=0, column=0, sticky=tk.W)
        self.manual_ref_var = tk.StringVar()
        self.manual_ref_entry = ttk.Entry(manual_frame, textvariable=self.manual_ref_var, width=24)
        self.manual_ref_entry.grid(row=0, column=1, padx=5, sticky=tk.W)

        image_frame = ttk.Frame(parent, padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)

        self.filter_preview_label = ttk.Label(image_frame, text="Filter preview")
        self.filter_preview_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.eticket_preview_label = ttk.Label(image_frame, text="Matched eticket preview")
        self.eticket_preview_label.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.status_text = tk.Text(parent, height=8, wrap=tk.WORD)
        self.status_text.pack(fill=tk.X, padx=10, pady=10)
        # Result panel color states: neutral / match / mismatch.
        self._status_neutral_bg = self.status_text.cget("bg")
        self._status_neutral_fg = self.status_text.cget("fg")
        self._status_match_bg = "#dff6df"
        self._status_mismatch_bg = "#ffdede"
        self.set_status("Select both folders to start.", tone="neutral")

        self.global_status_bar = ttk.Label(
            parent,
            textvariable=self.global_status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(8, 5),
        )
        self.global_status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._populate_matcher_eticket_selector()

    def open_client_window(self):
        if self.client_window is not None and self.client_window.winfo_exists():
            self.client_window.deiconify()
            self.client_window.lift()
            return
        self.client_window = tk.Toplevel(self.root)
        self.client_window.title("HIFI Filter to Eticket Matcher - Client Database")
        self.client_window.geometry("1200x760")
        self.client_window.protocol("WM_DELETE_WINDOW", self._on_close_client_window)
        self._build_client_db_ui(self.client_window)
        self.refresh_client_list()

    def _build_client_db_ui(self, parent):
        form_frame = ttk.LabelFrame(parent, text="Add / Edit Client", padding=10)
        form_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(form_frame, text="Client Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.client_name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.client_name_var, width=30).grid(row=0, column=1, sticky=tk.W, padx=5, pady=4)

        ttk.Label(form_frame, text="Maquette Eticket:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=4)
        self.client_maquette_eticket_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.client_maquette_eticket_var, width=50).grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)
        ttk.Button(form_frame, text="Browse", command=self.browse_maquette_eticket).grid(row=1, column=2, padx=5, pady=4)
        self.btn_auto_detect_eticket = ttk.Button(
            form_frame,
            text="Auto Detect Ref",
            command=self.auto_detect_ref_from_maquette_eticket_async,
            state=tk.NORMAL,
        )
        self.btn_auto_detect_eticket.grid(row=1, column=3, padx=5, pady=4)
        self.client_maquette_eticket_thumb = ttk.Label(form_frame, text="")
        self.client_maquette_eticket_thumb.grid(row=1, column=4, padx=6, pady=4)

        ttk.Label(form_frame, text="Maquette Filtre:").grid(row=2, column=0, sticky=tk.NW, padx=5, pady=4)
        self.client_maquette_filtre_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.client_maquette_filtre_var, width=50).grid(row=2, column=1, sticky=tk.W, padx=5, pady=4)
        ttk.Button(form_frame, text="Browse", command=self.browse_maquette_filtre).grid(row=2, column=2, padx=5, pady=4)
        self.btn_auto_detect_filtre = ttk.Button(
            form_frame,
            text="Auto Detect Ref",
            command=self.auto_detect_ref_from_maquette_filtre_async,
            state=tk.NORMAL,
        )
        self.btn_auto_detect_filtre.grid(row=2, column=3, padx=5, pady=4)
        self.client_maquette_filtre_thumb = ttk.Label(form_frame, text="")
        self.client_maquette_filtre_thumb.grid(row=2, column=4, padx=6, pady=4)

        ttk.Label(form_frame, text="Filter Reference Code:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=4)
        self.client_ref_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.client_ref_var, width=20).grid(row=3, column=1, sticky=tk.W, padx=5, pady=4)

        ttk.Label(form_frame, text="Maquette Eticket Serial:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=4)
        self.client_serial_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.client_serial_var, width=26).grid(row=4, column=1, sticky=tk.W, padx=5, pady=4)

        btn_row = ttk.Frame(form_frame)
        btn_row.grid(row=5, column=0, columnspan=4, sticky=tk.W, padx=5, pady=8)
        self.btn_save_client = ttk.Button(btn_row, text="Save Client", command=self.save_client)
        self.btn_save_client.pack(side=tk.LEFT, padx=3)
        self.btn_update_client = ttk.Button(btn_row, text="Update Selected", command=self.update_client)
        self.btn_update_client.pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Delete Selected", command=self.delete_client).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_row, text="Clear Form", command=self.clear_client_form).pack(side=tk.LEFT, padx=3)

        search_frame = ttk.Frame(parent, padding=(10, 2))
        search_frame.pack(fill=tk.X)
        ttk.Label(search_frame, text="Search (name or reference):").pack(side=tk.LEFT, padx=5)
        self.client_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.client_search_var, width=35)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<KeyRelease>", self.on_search_clients)

        table_frame = ttk.LabelFrame(parent, text="Clients", padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        cols = ("id", "name", "reference", "maquettes", "date_added")
        self.client_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=10)
        self.client_tree.heading("id", text="ID")
        self.client_tree.heading("name", text="Client Name")
        self.client_tree.heading("reference", text="Reference")
        self.client_tree.heading("maquettes", text="Maquettes")
        self.client_tree.heading("date_added", text="Date Added")
        self.client_tree.column("id", width=60, anchor=tk.CENTER)
        self.client_tree.column("name", width=200)
        self.client_tree.column("reference", width=120, anchor=tk.CENTER)
        self.client_tree.column("maquettes", width=90, anchor=tk.CENTER)
        self.client_tree.column("date_added", width=150, anchor=tk.CENTER)
        self.client_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.client_tree.bind("<<TreeviewSelect>>", self.on_client_select)
        self.client_tree.bind("<Double-1>", self.on_client_double_click_verify)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.client_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.client_tree.configure(yscrollcommand=tree_scroll.set)

        actions = ttk.Frame(parent, padding=(10, 0, 10, 8))
        actions.pack(fill=tk.X)
        self.btn_verify_eticket = ttk.Button(
            actions,
            text="Verify with Eticket (Selected Client)",
            command=self.verify_selected_client_with_eticket,
        )
        self.btn_verify_eticket.pack(side=tk.LEFT, padx=2)
        self.btn_verify_filter = ttk.Button(
            actions,
            text="Verify with Filter (Selected Client)",
            command=self.verify_selected_client_with_filter,
        )
        self.btn_verify_filter.pack(side=tk.LEFT, padx=2)

        preview_frame = ttk.Frame(parent, padding=(10, 4))
        preview_frame.pack(fill=tk.BOTH, expand=True)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        self.client_maquette_preview = ttk.Label(preview_frame, text="Client maquette preview")
        self.client_maquette_preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)
        self.client_eticket_preview = ttk.Label(preview_frame, text="Matched eticket preview")
        self.client_eticket_preview.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)

        self.client_status_text = tk.Text(parent, height=6, wrap=tk.WORD)
        self.client_status_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.set_client_status("Use this panel to manage clients and verify against indexed etickets.")

    def set_client_status(self, text):
        self.client_status_text.delete("1.0", tk.END)
        self.client_status_text.insert(tk.END, text)

    def set_global_status(self, text):
        self.global_status_var.set(text)

    def db_connect(self):
        return sqlite3.connect(self.db_path)

    def browse_maquette_eticket(self):
        file_path = filedialog.askopenfilename(
            title="Select Maquette Eticket (label reference)",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if file_path:
            self.client_maquette_eticket_var.set(file_path)
            self.show_image(file_path, self.client_maquette_preview, "client_filter")
            self._show_maquette_thumb(file_path, self.client_maquette_eticket_thumb, "client_maquette_eticket")
            self.root.after(
                0,
                lambda: self.auto_detect_ref_from_maquette_eticket_async(show_success=False),
            )
            self.root.after(
                0,
                lambda: self.auto_detect_serial_from_maquette_eticket_async(show_success=False),
            )

    def browse_maquette_filtre(self):
        file_path = filedialog.askopenfilename(
            title="Select Maquette Filtre (physical filter reference)",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if file_path:
            self.client_maquette_filtre_var.set(file_path)
            self._show_maquette_thumb(file_path, self.client_maquette_filtre_thumb, "client_maquette_filtre")

    def _show_maquette_thumb(self, path, label, which):
        try:
            pil_img = Image.open(path).convert("RGB")
            pil_img.thumbnail((120, 90))
            tk_img = ImageTk.PhotoImage(pil_img)
            label.config(image=tk_img, text="")
            if which == "client_maquette_eticket":
                self.client_maquette_eticket_photo = tk_img
            else:
                self.client_maquette_filtre_photo = tk_img
        except Exception:
            label.config(image="", text="(preview)")

    def _auto_detect_ref_task(self, photo_path, eticket_style=False):
        """Runs in a background thread â€” Maquette Eticket / Filtre Auto Detect Ref (Tesseract only)."""
        image = load_image_bgr(photo_path)
        if image is None:
            return {"ok": False, "kind": "load", "photo": photo_path}
        if eticket_style:
            try:
                ref = extract_reference_maquette_eticket_tesseract(image)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "kind": "exc", "error": str(exc), "photo": photo_path}
            if ref:
                return {"ok": True, "ref": ref, "photo": photo_path}
            inferred = infer_reference_from_filename(photo_path)
            if inferred:
                return {"ok": True, "ref": inferred, "photo": photo_path, "fallback": "filename"}
            return {"ok": False, "kind": "notfound", "photo": photo_path}
        try:
            ref, _trace = extract_reference_tesseract_full(image, crop_top_fraction=None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "kind": "exc", "error": str(exc), "photo": photo_path}
        if ref:
            return {"ok": True, "ref": ref, "photo": photo_path}
        try:
            ref, raw_ocr = ocr_filter_reference(image)
        except Exception:  # noqa: BLE001
            ref, raw_ocr = "", ""
        if ref:
            ref, _rw = reconcile_ocr_ref_with_known_filename(photo_path, ref, raw_ocr)
            return {"ok": True, "ref": ref, "photo": photo_path, "fallback": "tesseract"}
        inferred = infer_reference_from_filename(photo_path)
        if inferred:
            return {"ok": True, "ref": inferred, "photo": photo_path, "fallback": "filename"}
        return {"ok": False, "kind": "notfound", "photo": photo_path}

    def auto_detect_ref_from_maquette_eticket_async(self, show_success=True):
        if self.auto_detect_eticket_busy:
            return
        photo = self.client_maquette_eticket_var.get().strip()
        if not photo:
            messagebox.showwarning("Missing Photo", "Select a Maquette Eticket photo first.")
            return
        if not os.path.isfile(photo):
            messagebox.showwarning("Missing Photo", "Selected maquette eticket file does not exist on disk.")
            return

        self.auto_detect_eticket_busy = True
        self.btn_auto_detect_eticket.config(state=tk.DISABLED, text="Detecting...")
        self.set_global_status("Running OCR (Maquette Eticket)...")

        def task():
            self.root.after(0, lambda: self.set_global_status("Running OCR (Maquette Eticket)..."))
            return self._auto_detect_ref_task(photo, eticket_style=True)

        def on_success(result):
            self.auto_detect_eticket_busy = False
            self.btn_auto_detect_eticket.config(state=tk.NORMAL, text="Auto Detect Ref")
            self.set_global_status(
                "OCR Ready âœ…"
            )
            if not result.get("ok"):
                k = result.get("kind")
                if k == "load":
                    messagebox.showerror("Image Error", "Could not load the selected maquette eticket photo.")
                elif k == "exc":
                    messagebox.showerror("Tesseract", result.get("error", "Unknown error"))
                else:
                    manual = simpledialog.askstring(
                        "Filter reference",
                        "Could not auto-detect. Enter SA/SN reference (e.g. SA17088):",
                        parent=self.root,
                    )
                    if manual:
                        n = normalize_reference(manual)
                        if n and is_valid_hifi_reference(n):
                            self.client_ref_var.set(n)
                            self.set_client_status(
                                f"Reference entered manually for {os.path.basename(result['photo'])}:\n{n}"
                            )
                        else:
                            messagebox.showwarning(
                                "Invalid reference",
                                "Enter a valid SA##### or SN###### code.",
                            )
                            self.set_client_status(
                                f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}"
                            )
                    else:
                        self.set_client_status(
                            f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}"
                        )
                return
            self.client_ref_var.set(result["ref"])
            fb = result.get("fallback")
            if fb == "filename":
                msg = f"Filename fallback for {os.path.basename(result['photo'])}:\n{result['ref']}"
            else:
                msg = f"Reference detected (Maquette Eticket) for {os.path.basename(result['photo'])}:\n{result['ref']}"
            if show_success or result.get("fallback"):
                self.set_client_status(msg)

        def on_error(exc):
            self.auto_detect_eticket_busy = False
            self.btn_auto_detect_eticket.config(state=tk.NORMAL, text="Auto Detect Ref")
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("Tesseract", str(exc))

        self.run_in_background(task, on_success, on_error)

    def auto_detect_ref_from_maquette_filtre_async(self, show_success=True):
        if self.auto_detect_filtre_busy:
            return
        photo = self.client_maquette_filtre_var.get().strip()
        if not photo:
            messagebox.showwarning("Missing Photo", "Select a Maquette Filtre photo first.")
            return
        if not os.path.isfile(photo):
            messagebox.showwarning("Missing Photo", "Selected maquette filtre file does not exist on disk.")
            return

        self.auto_detect_filtre_busy = True
        self.btn_auto_detect_filtre.config(state=tk.DISABLED, text="Detecting...")
        self.set_global_status("Running OCR (Maquette Filtre)...")

        def task():
            self.root.after(0, lambda: self.set_global_status("Running OCR (Maquette Filtre)..."))
            return self._auto_detect_ref_task(photo, eticket_style=False)

        def on_success(result):
            self.auto_detect_filtre_busy = False
            self.btn_auto_detect_filtre.config(state=tk.NORMAL, text="Auto Detect Ref")
            self.set_global_status(
                "OCR Ready âœ…"
            )
            if not result.get("ok"):
                if result.get("kind") == "load":
                    messagebox.showerror("Image Error", "Could not load the selected maquette filtre photo.")
                elif result.get("kind") == "exc":
                    messagebox.showerror("OCR", result.get("error", "Unknown error"))
                else:
                    messagebox.showwarning(
                        "Auto-detect",
                        "Could not detect a valid SA/SN reference (Tesseract).",
                    )
                    self.set_client_status(f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}")
                return
            self.client_ref_var.set(result["ref"])
            fb = result.get("fallback")
            if fb == "filename":
                msg = f"Filename fallback for {os.path.basename(result['photo'])}:\n{result['ref']}"
            elif fb == "tesseract":
                msg = f"Reference detected with Tesseract (Maquette Filtre) for {os.path.basename(result['photo'])}:\n{result['ref']}"
            else:
                msg = f"Reference detected (Maquette Filtre) for {os.path.basename(result['photo'])}:\n{result['ref']}"
            if show_success or result.get("fallback"):
                self.set_client_status(msg)

        def on_error(exc):
            self.auto_detect_filtre_busy = False
            self.btn_auto_detect_filtre.config(state=tk.NORMAL, text="Auto Detect Ref")
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("OCR", str(exc))

        self.run_in_background(task, on_success, on_error)

    def auto_detect_serial_from_maquette_eticket_async(self, show_success=True):
        if self.auto_detect_serial_busy:
            return
        photo = self.client_maquette_eticket_var.get().strip()
        if not photo:
            messagebox.showwarning("Missing Photo", "Select a Maquette Eticket photo first.")
            return
        if not os.path.isfile(photo):
            messagebox.showwarning("Missing Photo", "Selected maquette eticket file does not exist on disk.")
            return

        self.auto_detect_serial_busy = True
        btn = getattr(self, "btn_auto_detect_serial", None)
        if btn is not None:
            btn.config(state=tk.DISABLED, text="Detecting...")
        self.set_global_status("Running OCR (Maquette Eticket Serial)...")

        def task():
            if not os.path.isfile(photo):
                return {"ok": False, "kind": "load", "photo": photo}
            serial = self._extract_serial_step4_cached(photo)
            if serial:
                return {"ok": True, "serial": serial, "photo": photo}
            return {"ok": False, "kind": "notfound", "photo": photo}

        def on_success(result):
            self.auto_detect_serial_busy = False
            btn2 = getattr(self, "btn_auto_detect_serial", None)
            if btn2 is not None:
                btn2.config(state=tk.NORMAL, text="Auto Detect Serial")
            self.set_global_status("OCR Ready âœ…")
            if not result.get("ok"):
                if result.get("kind") == "load":
                    messagebox.showerror("Image Error", "Could not load the selected maquette eticket photo.")
                else:
                    self.set_client_status(
                        f"No barcode serial detected for:\n{os.path.basename(result['photo'])}"
                    )
                return
            self.client_serial_var.set(result["serial"])
            if show_success:
                self.set_client_status(
                    f"Serial detected (Maquette Eticket) for {os.path.basename(result['photo'])}:\n{result['serial']}"
                )

        def on_error(exc):
            self.auto_detect_serial_busy = False
            btn3 = getattr(self, "btn_auto_detect_serial", None)
            if btn3 is not None:
                btn3.config(state=tk.NORMAL, text="Auto Detect Serial")
            self.set_global_status("OCR Ready âœ…")
            messagebox.showerror("OCR", str(exc))

        self.run_in_background(task, on_success, on_error)

    def _validate_client_form_core(self):
        name = self.client_name_var.get().strip()
        eticket = self.client_maquette_eticket_var.get().strip()
        filtre = self.client_maquette_filtre_var.get().strip()
        ref = normalize_reference(self.client_ref_var.get())
        serial = re.sub(r"[^0-9]", "", (self.client_serial_var.get() or "").strip())
        if not name or not eticket:
            messagebox.showwarning("Missing Data", "Client name and Maquette Eticket path are required.")
            return None
        if not os.path.isfile(eticket):
            messagebox.showwarning("Missing Photo", "Maquette Eticket file does not exist on disk.")
            return None
        if filtre and not os.path.isfile(filtre):
            messagebox.showwarning("Missing Photo", "Maquette Filtre file does not exist on disk.")
            return None
        return name, eticket, filtre or "", ref, serial

    def _start_auto_detect_for_save(self, continuation):
        """If filter reference is empty, run OCR on Maquette Filtre (preferred) or Maquette Eticket, then save/update."""
        if self.auto_detect_for_save_busy:
            return
        if self.auto_detect_filtre_busy or self.auto_detect_eticket_busy:
            messagebox.showinfo("Please wait", "Another auto-detect is in progress.")
            return
        filtre = self.client_maquette_filtre_var.get().strip()
        eticket = self.client_maquette_eticket_var.get().strip()
        if filtre and os.path.isfile(filtre):
            photo = filtre
            src = "Maquette Filtre"
        elif eticket and os.path.isfile(eticket):
            photo = eticket
            src = "Maquette Eticket"
        else:
            messagebox.showwarning(
                "Reference Missing",
                "Browse a Maquette Filtre or Eticket image, or enter the Filter Reference Code manually.",
            )
            return

        self.auto_detect_for_save_busy = True
        self.btn_save_client.config(state=tk.DISABLED)
        self.btn_update_client.config(state=tk.DISABLED)
        self.set_global_status("Running OCR (auto-detect reference)...")
        self.set_client_status(f"Detecting reference from {src}â€¦")

        eticket_only = src == "Maquette Eticket"

        def task():
            self.root.after(0, lambda: self.set_global_status("Running OCR (auto-detect reference)..."))
            return self._auto_detect_ref_task(photo, eticket_style=eticket_only)

        def on_success(result):
            self.auto_detect_for_save_busy = False
            self.btn_save_client.config(state=tk.NORMAL)
            self.btn_update_client.config(state=tk.NORMAL)
            self.set_global_status(
                "OCR Ready âœ…"
            )
            if not result.get("ok"):
                if result.get("kind") == "load":
                    messagebox.showerror("Image Error", "Could not load the selected maquette photo.")
                elif result.get("kind") == "exc":
                    messagebox.showerror("OCR", result.get("error", "Unknown error"))
                else:
                    if eticket_only:
                        manual = simpledialog.askstring(
                            "Filter reference",
                            "Could not auto-detect. Enter SA/SN reference (e.g. SA17088):",
                            parent=self.root,
                        )
                        if manual:
                            n = normalize_reference(manual)
                            if n and is_valid_hifi_reference(n):
                                self.client_ref_var.set(n)
                                self.set_client_status(
                                    f"Reference entered manually for {os.path.basename(result['photo'])}:\n{n}"
                                )
                                if continuation == "save":
                                    self.save_client()
                                else:
                                    self.update_client()
                            else:
                                messagebox.showwarning(
                                    "Invalid reference",
                                    "Enter a valid SA##### or SN###### code.",
                                )
                                self.set_client_status(
                                    f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}"
                                )
                        else:
                            self.set_client_status(
                                f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}"
                            )
                    else:
                        messagebox.showwarning(
                            "Auto-detect",
                            "Could not detect a valid SA/SN reference (Tesseract).",
                        )
                        self.set_client_status(
                            f"No (SA|SN) pattern for:\n{os.path.basename(result['photo'])}"
                        )
                return
            self.client_ref_var.set(result["ref"])
            fb = result.get("fallback")
            if fb == "filename":
                msg = f"Filename fallback for {os.path.basename(result['photo'])}:\n{result['ref']}"
            elif fb == "tesseract":
                msg = f"Reference detected with Tesseract ({src}) for {os.path.basename(result['photo'])}:\n{result['ref']}"
            else:
                msg = f"Reference detected ({src}) for {os.path.basename(result['photo'])}:\n{result['ref']}"
            self.set_client_status(msg)
            if continuation == "save":
                self.save_client()
            else:
                self.update_client()

        def on_error(exc):
            self.auto_detect_for_save_busy = False
            self.btn_save_client.config(state=tk.NORMAL)
            self.btn_update_client.config(state=tk.NORMAL)
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("OCR", str(exc))

        self.run_in_background(task, on_success, on_error)

    def save_client(self):
        if self.auto_detect_for_save_busy:
            return
        data = self._validate_client_form_core()
        if not data:
            return
        name, eticket, filtre, ref, serial = data
        if not ref:
            self._start_auto_detect_for_save("save")
            return
        inserted_id = None
        conn = self.db_connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO clients (name, maquette_eticket_path, maquette_filtre_path, filter_reference_code, maquette_eticket_serial_code)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, eticket, filtre, ref, serial),
            )
            inserted_id = cursor.lastrowid
            conn.commit()
        except sqlite3.IntegrityError:
            messagebox.showerror("Duplicate Client", "A client with this name already exists. Please use a unique name.")
            return
        except Exception as exc:
            messagebox.showerror("Database Error", f"Could not save client:\n{exc}")
            return
        finally:
            conn.close()
        self.refresh_client_list()
        if inserted_id is not None:
            self.select_client_by_id(inserted_id)
        self.clear_client_form()
        self.set_client_status(f"Client '{name}' saved successfully. Running eticket verification...")
        # Auto-verify right after save so matching is fully automatic.
        self.verify_selected_client_with_eticket()

    def refresh_client_list(self, query=""):
        for row in self.client_tree.get_children():
            self.client_tree.delete(row)
        self.clients = []
        conn = self.db_connect()
        try:
            if query:
                like_q = f"%{query}%"
                cur = conn.execute(
                    """
                    SELECT id, name, maquette_eticket_path, maquette_filtre_path, filter_reference_code, maquette_eticket_serial_code, date_added
                    FROM clients
                    WHERE name LIKE ? OR filter_reference_code LIKE ?
                    ORDER BY datetime(date_added) DESC, id DESC
                    """,
                    (like_q, like_q),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT id, name, maquette_eticket_path, maquette_filtre_path, filter_reference_code, maquette_eticket_serial_code, date_added
                    FROM clients
                    ORDER BY datetime(date_added) DESC, id DESC
                    """
                )
            self.clients = cur.fetchall()
        except Exception as exc:
            messagebox.showerror("Database Error", f"Could not load clients:\n{exc}")
            return
        finally:
            conn.close()

        for client in self.clients:
            client_id, name, et_path, fi_path, ref, _serial, date_added = client
            et_ok = bool(et_path and os.path.isfile(et_path))
            fi_ok = bool(fi_path and os.path.isfile(fi_path))
            mq = "âœ…" if (et_ok and fi_ok) else "âŒ"
            self.client_tree.insert("", tk.END, values=(client_id, name, ref, mq, date_added))
        self._refresh_known_references()

    def on_search_clients(self, _event=None):
        self.refresh_client_list(self.client_search_var.get().strip())

    def _get_selected_client_row(self):
        selection = self.client_tree.selection()
        if not selection:
            return None
        values = self.client_tree.item(selection[0], "values")
        if not values:
            return None
        selected_id = int(values[0])
        for row in self.clients:
            if row[0] == selected_id:
                return row
        return None

    def _get_active_client_row(self):
        """Prefer explicit selected_client_id; fallback to current tree selection."""
        sid = getattr(self, "selected_client_id", None)
        if sid is not None:
            for row in self.clients:
                try:
                    if int(row[0]) == int(sid):
                        return row
                except Exception:
                    continue
        try:
            return self._get_selected_client_row()
        except Exception:
            return None

    def _token_exists_in_text(self, raw_text, token, min_ratio=75):
        txt = re.sub(r"[^A-Z0-9]", "", (raw_text or "").upper())
        tok = re.sub(r"[^A-Z0-9]", "", (token or "").upper())
        if not txt or not tok:
            return False
        if tok in txt:
            return True
        # OCR can confuse EAC/HIFI letters on curved/low-contrast print.
        if tok == "EAC":
            for var in ("EAC", "E4C", "EHC", "EAG", "EAT", "FAC"):
                if var in txt or fuzz.partial_ratio(var, txt) >= max(62, int(min_ratio) - 12):
                    return True
        if tok == "HIFI":
            for var in ("HIFI", "HIFI", "H1FI", "HIF1"):
                if var in txt or fuzz.partial_ratio(var, txt) >= max(64, int(min_ratio) - 10):
                    return True
        return fuzz.partial_ratio(tok, txt) >= int(min_ratio)

    def _phrase_exists_in_text(self, raw_text, relaxed=False):
        # Existence-only check (not exact OCR) for "FILTRE A AIR - AIR FILTER" style phrases.
        txt = re.sub(r"[^A-Z0-9]", "", (raw_text or "").upper())
        # Common OCR confusions on small curved print.
        txt = txt.replace("1", "I").replace("4", "A")
        if not txt:
            return False
        has_filtre = ("FILTRE" in txt) or (fuzz.partial_ratio("FILTRE", txt) >= 70)
        has_filter = ("FILTER" in txt) or (fuzz.partial_ratio("FILTER", txt) >= 70)
        has_air = ("AIR" in txt) or (fuzz.partial_ratio("AIR", txt) >= 75)
        has_luft = ("LUFTFILTER" in txt) or (fuzz.partial_ratio("LUFTFILTER", txt) >= 70)
        if relaxed:
            # For eticket/maquette phrase line, accept one strong language token.
            return bool(has_filtre or has_filter or has_luft or (has_air and (has_filtre or has_filter)))
        # Accept one language root + AIR, or both roots together.
        return bool((has_air and (has_filtre or has_filter)) or (has_filtre and has_filter) or has_luft)

    def _extract_presence_profile(self, image_path, kind, known_ref=""):
        """
        Presence profile for requested tokens by image kind.
        kind: live_filter | eticket | maquette_filter | maquette_eticket
        """
        profile = {"eac": False, "hifi": False, "phrase": False, "reference": False, "reference_value": ""}
        if not image_path or not os.path.isfile(image_path):
            return profile
        try:
            mtime = int(os.path.getmtime(image_path))
        except Exception:
            mtime = 0
        cache_key = (
            os.path.abspath(os.path.normpath(image_path)),
            int(mtime),
            str(kind),
            normalize_reference(known_ref or ""),
        )
        cached = self.presence_profile_cache.get(cache_key)
        if isinstance(cached, dict):
            return dict(cached)
        image = load_image_bgr(image_path)
        if image is None or image.size == 0:
            return profile

        raws = []
        ref = normalize_reference(known_ref or "")
        try:
            if kind in ("live_filter", "maquette_filter"):
                r, raw = ocr_filter_reference(image, deadline=time.monotonic() + 4.0)
                if not ref:
                    ref = normalize_reference(r or "")
                if raw:
                    raws.append(raw)
            else:
                r, raw = ocr_eticket_reference_fast(image, timeout_sec=2.4)
                if (not r):
                    r = extract_reference_maquette_eticket_tesseract(image)
                if not ref:
                    ref = normalize_reference(r or "")
                if raw:
                    raws.append(raw)
        except Exception:
            pass

        # Extra cheap OCR snippets to improve presence checks (curved and localized text).
        h, w = image.shape[:2]
        snippets = [image]
        if h > 20 and w > 20:
            snippets.append(image[: max(1, int(h * 0.35)), :])  # top band
            snippets.append(image[max(0, int(h * 0.12)) : min(h, int(h * 0.70)), :])  # central band
            snippets.append(image[:, : max(1, int(w * 0.30))])  # left strip (EAC often vertical)
            snippets.append(image[:, max(0, int(w * 0.62)) :])  # right strip
            # Dedicated phrase stripe under SA/SN banner on eticket/maquette images.
            if kind in ("eticket", "maquette_eticket"):
                py0 = max(0, int(h * 0.22))
                py1 = min(h, int(h * 0.52))
                px0 = max(0, int(w * 0.08))
                px1 = min(w, int(w * 0.92))
                phrase_band = image[py0:py1, px0:px1]
                if phrase_band is not None and phrase_band.size > 0:
                    snippets.append(phrase_band)
                    up_phrase = cv2.resize(phrase_band, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
                    snippets.append(up_phrase)
        for sn in snippets:
            if sn is None or sn.size == 0:
                continue
            for rot in (sn, cv2.rotate(sn, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(sn, cv2.ROTATE_90_COUNTERCLOCKWISE)):
                try:
                    txt = ocr_text_tesseract(rot)
                except Exception:
                    txt = ""
                if txt:
                    raws.append(txt)
                try:
                    gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
                    txt2 = pytesseract.image_to_string(clahe, config=_tesseract_cfg_psm(11))
                except Exception:
                    txt2 = ""
                if txt2:
                    raws.append(txt2)

        # Phrase-dedicated OCR under dark SA/SN banner for eticket and maquette_eticket.
        if kind in ("eticket", "maquette_eticket"):
            try:
                rect = _eticket_largest_dark_rectangle_roi(image)
            except Exception:
                rect = None
            if rect is not None:
                x, y, bw, bh = rect
                px = max(0, x - int(0.03 * bw))
                pw = min(image.shape[1], x + bw + int(0.03 * bw)) - px
                py1 = min(image.shape[0], y + bh + int(0.06 * bh))
                py2 = min(image.shape[0], y + bh + int(1.25 * bh))
                if pw > 20 and py2 - py1 > 8:
                    band = image[py1:py2, px : px + pw]
                    try:
                        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                        g = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(g)
                        up = cv2.resize(g, None, fx=2.8, fy=2.8, interpolation=cv2.INTER_CUBIC)
                    except Exception:
                        up = None
                    if up is not None and up.size > 0:
                        for src in (up, cv2.bitwise_not(up)):
                            for psm in (7, 6, 11):
                                try:
                                    t = pytesseract.image_to_string(src, config=_tesseract_cfg_psm(psm))
                                except Exception:
                                    t = ""
                                if t:
                                    raws.append(t)

        raw_all = "\n".join(r for r in raws if r)

        profile["reference_value"] = ref
        profile["reference"] = bool(ref)
        profile["eac"] = self._token_exists_in_text(raw_all, "EAC", min_ratio=66)
        if kind in ("live_filter", "eticket", "maquette_filter"):
            profile["hifi"] = self._token_exists_in_text(raw_all, "HIFI", min_ratio=68)
        profile["phrase"] = self._phrase_exists_in_text(
            raw_all,
            relaxed=bool(kind in ("eticket", "maquette_eticket")),
        )
        self.presence_profile_cache[cache_key] = dict(profile)
        return profile

    def _warm_presence_cache_context(self):
        sel_path = os.path.abspath(os.path.normpath(getattr(self, "matcher_selected_eticket_path", "") or ""))
        sel_ref = normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or "")
        row = self._get_active_client_row()
        mq_et_path = os.path.abspath(os.path.normpath((row[2] if row and len(row) > 2 else "") or ""))
        mq_fi_path = os.path.abspath(os.path.normpath((row[3] if row and len(row) > 3 else "") or ""))
        mq_fi_ref = normalize_reference((row[4] if row and len(row) > 4 else "") or "")

        def _task():
            if sel_path and os.path.isfile(sel_path):
                self._extract_presence_profile(sel_path, "eticket", known_ref=sel_ref)
                self._extract_serial_step4_cached(sel_path)
            if mq_et_path and os.path.isfile(mq_et_path):
                self._extract_presence_profile(mq_et_path, "maquette_eticket", known_ref="")
                self._extract_serial_step4_cached(mq_et_path)
            if mq_fi_path and os.path.isfile(mq_fi_path):
                self._extract_presence_profile(mq_fi_path, "maquette_filter", known_ref=mq_fi_ref)

        threading.Thread(target=_task, daemon=True).start()

    def _serial_cache_key(self, image_path):
        if not image_path:
            return None
        p = os.path.abspath(os.path.normpath(image_path))
        if not os.path.isfile(p):
            return None
        try:
            return (p, int(os.path.getmtime(p)))
        except Exception:
            return None

    def _extract_serial_step4_cached(self, image_path, image_bgr=None, force_refresh=False):
        key = self._serial_cache_key(image_path)
        if (not force_refresh) and key is not None and key in self.step4_serial_cache:
            return self.step4_serial_cache.get(key, "") or ""
        img = image_bgr
        if img is None or getattr(img, "size", 0) == 0:
            img = load_image_bgr(image_path)
        serial = ""
        if img is not None and img.size > 0:
            try:
                serial = extract_eticket_serial_number_step4_strict(img) or ""
            except Exception:
                serial = ""
        if key is not None:
            if serial:
                self.step4_serial_cache[key] = serial
            else:
                # Do not pin empty OCR forever; allow next attempt to re-read.
                self.step4_serial_cache.pop(key, None)
        return serial

    def _build_client_auto_pipeline(
        self,
        filter_ref,
        selected_eticket_ref,
        client_row,
        live_filter_path="",
        live_presence_profile=None,
    ):
        """
        Build automatic 3-step comparison summary:
        1) live filter ref vs selected eticket ref
        2) selected eticket ref vs client maquette-eticket ref
        3) live filter ref vs client maquette-filter ref
        """
        fr = normalize_reference(filter_ref or "")
        sel = normalize_reference(selected_eticket_ref or "")
        if not client_row:
            return "", None
        _cid, cname, mq_et_path, mq_fi_path, stored_filter_ref, stored_mq_et_serial, _date = client_row

        # Step 2: resolve maquette eticket reference (quick OCR fallback if needed).
        mq_et_ref = ""
        mq_et_src = ""
        if mq_et_path and os.path.isfile(mq_et_path):
            inf = infer_reference_from_filename(mq_et_path)
            if inf and is_valid_hifi_reference(inf):
                mq_et_ref = normalize_reference(inf)
                mq_et_src = "filename"
            else:
                img = load_image_bgr(mq_et_path)
                if img is not None and img.size > 0:
                    try:
                        r_fast, _raw_fast = ocr_eticket_reference_fast(img, timeout_sec=2.2)
                    except Exception:
                        r_fast = ""
                    if r_fast:
                        mq_et_ref = normalize_reference(r_fast)
                        mq_et_src = "fast_ocr"
                    else:
                        try:
                            r_full = extract_reference_maquette_eticket_tesseract(img)
                        except Exception:
                            r_full = ""
                        if r_full:
                            mq_et_ref = normalize_reference(r_full)
                            mq_et_src = "banner_ocr"
                        else:
                            # Last-resort robust pass: full eticket OCR + raw-text strict scan.
                            try:
                                r_any, raw_any = ocr_eticket_reference(img)
                            except Exception:
                                r_any, raw_any = "", ""
                            if r_any:
                                mq_et_ref = normalize_reference(r_any)
                                mq_et_src = "full_eticket_ocr"
                            elif raw_any:
                                strict_hits = _strict_refs_from_raw_text(raw_any)
                                if strict_hits:
                                    mq_et_ref = normalize_reference(strict_hits[0])
                                    mq_et_src = "raw_strict_scan"
                                else:
                                    # Extra OCR sweep on cropped/rotated variants for stubborn maquettes.
                                    for crop in build_eticket_raw_crops(img) + [img]:
                                        found = ""
                                        for src in (crop, cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)):
                                            try:
                                                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                                                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
                                            except Exception:
                                                continue
                                            for psm in (7, 6, 11):
                                                try:
                                                    txt = pytesseract.image_to_string(clahe, config=_tesseract_cfg_psm(psm))
                                                except Exception:
                                                    txt = ""
                                                found = _eticket_ref_from_ocr_text(txt) or extract_reference_from_text(txt)
                                                if found:
                                                    break
                                            if found:
                                                break
                                        if found:
                                            mq_et_ref = normalize_reference(found)
                                            mq_et_src = "sweep_ocr"
                                            break

        # Step 3: maquette filter reference from stored code first, OCR fallback.
        mq_fi_ref = normalize_reference(stored_filter_ref or "")
        mq_fi_src = "stored_client_ref" if mq_fi_ref else ""
        if (not mq_fi_ref) and mq_fi_path and os.path.isfile(mq_fi_path):
            imgf = load_image_bgr(mq_fi_path)
            if imgf is not None and imgf.size > 0:
                try:
                    rfi, _rawfi = ocr_filter_reference(imgf, deadline=time.monotonic() + 5.0)
                except Exception:
                    rfi = ""
                mq_fi_ref = normalize_reference(rfi or "")
                mq_fi_src = "ocr_maquette_filter" if mq_fi_ref else ""

        step1_ok = bool(fr and sel and fr == sel)
        step2_ok = bool(sel and mq_et_ref and sel == mq_et_ref)
        step3_ok = bool(fr and mq_fi_ref and fr == mq_fi_ref)
        # Step 4: serial number compare (selected eticket vs maquette eticket).
        mq_et_serial = re.sub(r"[^0-9]", "", (stored_mq_et_serial or ""))
        sel_serial = ""
        sel_path = getattr(self, "matcher_selected_eticket_path", "") or ""
        sel_serial_path = getattr(self, "matcher_selected_eticket_serial_path", "") or ""
        if sel_serial_path and os.path.isfile(sel_serial_path):
            sel_serial = self._extract_serial_step4_cached(sel_serial_path, force_refresh=True)
        elif sel_path and os.path.isfile(sel_path):
            sel_serial = self._extract_serial_step4_cached(sel_path)
        if (not mq_et_serial) and mq_et_path and os.path.isfile(mq_et_path):
            mq_et_serial = self._extract_serial_step4_cached(mq_et_path)
        sel_serial_cmp = re.sub(r"\s+", "", (sel_serial or ""))
        mq_et_serial_cmp = re.sub(r"\s+", "", (mq_et_serial or ""))
        step4_ok = bool(sel_serial_cmp and mq_et_serial_cmp and sel_serial_cmp == mq_et_serial_cmp)

        # Presence checks requested by user for each image type.
        live_prof = (
            dict(live_presence_profile)
            if isinstance(live_presence_profile, dict)
            else self._extract_presence_profile(live_filter_path, "live_filter", known_ref=fr)
        )
        sel_prof = self._extract_presence_profile(sel_path, "eticket", known_ref=sel)
        mq_et_prof = self._extract_presence_profile(mq_et_path, "maquette_eticket", known_ref=mq_et_ref)
        mq_fi_prof = self._extract_presence_profile(mq_fi_path, "maquette_filter", known_ref=mq_fi_ref)
        # User preference: always treat maquette eticket phrase presence as YES.
        mq_et_prof["phrase"] = True

        lines = [
            "",
            "--- Client auto pipeline ---",
            f"Client: {cname}",
            "Step 1 (live filter vs selected eticket): "
            + ("MATCH ✅" if step1_ok else "MISMATCH ❌")
            + f"  [{fr or 'N/A'} vs {sel or 'N/A'}]",
            "Step 2 (selected eticket vs maquette eticket): "
            + ("MATCH ✅" if step2_ok else "MISMATCH ❌")
            + f"  [{sel or 'N/A'} vs {mq_et_ref or 'N/A'}]"
            + (f" (source: {mq_et_src})" if mq_et_src else ""),
            "Step 3 (live filter vs maquette filter): "
            + ("MATCH ✅" if step3_ok else "MISMATCH ❌")
            + f"  [{fr or 'N/A'} vs {mq_fi_ref or 'N/A'}]"
            + (f" (source: {mq_fi_src})" if mq_fi_src else ""),
            "Step 4 (selected eticket serial vs maquette eticket serial): "
            + ("MATCH ✅" if step4_ok else "MISMATCH ❌")
            + f"  [{sel_serial or 'N/A'} vs {mq_et_serial or 'N/A'}]",
            "",
            "Presence checks (requested):",
            f"- Live filter: EAC={'YES' if live_prof['eac'] else 'NO'}, HIFI={'YES' if live_prof['hifi'] else 'NO'}, REF={'YES' if live_prof['reference'] else 'NO'}, PHRASE={'YES' if live_prof['phrase'] else 'NO'}",
            f"- Selected eticket: EAC={'YES' if sel_prof['eac'] else 'NO'}, HIFI={'YES' if sel_prof['hifi'] else 'NO'}, REF={'YES' if sel_prof['reference'] else 'NO'}, PHRASE={'YES' if sel_prof['phrase'] else 'NO'}",
            f"- Maquette eticket: EAC={'YES' if mq_et_prof['eac'] else 'NO'}, REF={'YES' if mq_et_prof['reference'] else 'NO'}, PHRASE={'YES' if mq_et_prof['phrase'] else 'NO'}",
            f"- Maquette filter: EAC={'YES' if mq_fi_prof['eac'] else 'NO'}, HIFI={'YES' if mq_fi_prof['hifi'] else 'NO'}, REF={'YES' if mq_fi_prof['reference'] else 'NO'}, PHRASE={'YES' if mq_fi_prof['phrase'] else 'NO'}",
        ]
        all_known = bool(sel and mq_et_ref and fr and mq_fi_ref and sel_serial and mq_et_serial)
        all_ok = bool(all_known and step1_ok and step2_ok and step3_ok and step4_ok)
        if all_known:
            lines.append("Overall client pipeline: " + ("MATCH ✅" if all_ok else "MISMATCH ❌"))
            tone = "match" if all_ok else "mismatch"
        else:
            lines.append("Overall client pipeline: PARTIAL (some references missing)")
            tone = "mismatch"
        return "\n".join(lines), tone

    def on_client_select(self, _event=None):
        row = self._get_selected_client_row()
        if not row:
            return
        client_id, name, eticket, filtre, ref, serial, _date_added = row
        self.selected_client_id = client_id
        self.client_name_var.set(name)
        self.client_maquette_eticket_var.set(eticket)
        self.client_maquette_filtre_var.set(filtre or "")
        self.client_ref_var.set(ref)
        self.client_serial_var.set(re.sub(r"[^0-9]", "", (serial or "")))
        if eticket and os.path.isfile(eticket):
            self.show_image(eticket, self.client_maquette_preview, "client_filter")
            self._show_maquette_thumb(eticket, self.client_maquette_eticket_thumb, "client_maquette_eticket")
            serial_clean = re.sub(r"[^0-9]", "", (serial or ""))
            if serial_clean:
                key = self._serial_cache_key(eticket)
                if key is not None:
                    self.step4_serial_cache[key] = serial_clean
            else:
                # Keep serial field auto-synced like reference detect (only when missing).
                self.root.after(
                    0,
                    lambda: self.auto_detect_serial_from_maquette_eticket_async(show_success=False),
                )
        else:
            self.client_maquette_eticket_thumb.config(image="", text="")
        if filtre and os.path.isfile(filtre):
            self._show_maquette_thumb(filtre, self.client_maquette_filtre_thumb, "client_maquette_filtre")
        else:
            self.client_maquette_filtre_thumb.config(image="", text="")
        if eticket and not os.path.isfile(eticket):
            self.set_client_status(
                "Maquette Eticket file not found on disk.\n"
                f"Stored path:\n{eticket}\n\n"
                "Use Browse next to Maquette Eticket to pick the file again, then Save or Update Selected."
            )
        else:
            self.set_client_status("Client loaded. Paths look valid.")
        self._warm_presence_cache_context()

    def on_client_double_click_verify(self, _event=None):
        self.verify_selected_client_with_eticket()

    def update_client(self):
        if self.auto_detect_for_save_busy:
            return
        row = self._get_selected_client_row()
        if not row:
            messagebox.showwarning("No Selection", "Select a client to update.")
            return
        data = self._validate_client_form_core()
        if not data:
            return
        name, eticket, filtre, ref, serial = data
        if not ref:
            self._start_auto_detect_for_save("update")
            return
        conn = self.db_connect()
        try:
            conn.execute(
                """
                UPDATE clients SET name=?, maquette_eticket_path=?, maquette_filtre_path=?, filter_reference_code=?, maquette_eticket_serial_code=?
                WHERE id=?
                """,
                (name, eticket, filtre, ref, serial, row[0]),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            messagebox.showerror("Duplicate Client", "A client with this name already exists. Please use a unique name.")
            return
        except Exception as exc:
            messagebox.showerror("Database Error", f"Could not update client:\n{exc}")
            return
        finally:
            conn.close()
        self.refresh_client_list(self.client_search_var.get().strip())
        self.set_client_status(f"Client '{name}' updated successfully.")

    def delete_client(self):
        row = self._get_selected_client_row()
        if not row:
            messagebox.showwarning("No Selection", "Select a client to delete.")
            return
        client_id, name, _et, _fi, _ref, _serial, _date_added = row
        if not messagebox.askyesno("Confirm Delete", f"Delete client '{name}'?"):
            return
        conn = self.db_connect()
        try:
            conn.execute("DELETE FROM clients WHERE id=?", (client_id,))
            conn.commit()
        except Exception as exc:
            messagebox.showerror("Database Error", f"Could not delete client:\n{exc}")
            return
        finally:
            conn.close()
        self.refresh_client_list(self.client_search_var.get().strip())
        self.clear_client_form()
        self.client_maquette_preview.config(image="", text="Client maquette preview")
        self.client_eticket_preview.config(image="", text="Matched eticket preview")
        self.client_maquette_eticket_thumb.config(image="", text="")
        self.client_maquette_filtre_thumb.config(image="", text="")
        self.set_client_status(f"Client '{name}' deleted.")

    def clear_client_form(self):
        self.selected_client_id = None
        self.client_name_var.set("")
        self.client_maquette_eticket_var.set("")
        self.client_maquette_filtre_var.set("")
        self.client_ref_var.set("")
        self.client_serial_var.set("")
        self.client_maquette_eticket_thumb.config(image="", text="")
        self.client_maquette_filtre_thumb.config(image="", text="")

    def select_client_by_id(self, client_id):
        for item_id in self.client_tree.get_children():
            values = self.client_tree.item(item_id, "values")
            if values and int(values[0]) == int(client_id):
                self.client_tree.selection_set(item_id)
                self.client_tree.focus(item_id)
                self.client_tree.see(item_id)
                self.on_client_select()
                return

    def _show_verification_popup(
        self,
        left_path,
        right_path,
        header_lines,
        left_caption,
        right_caption,
        default_filename,
    ):
        """Large popup (>=900x600) with sharpened images, zoom on click, Save Result as JPG."""
        left_bgr = load_image_bgr(left_path)
        right_bgr = load_image_bgr(right_path)
        if left_bgr is None or right_bgr is None:
            messagebox.showerror("Image Error", "Could not load one of the images for comparison.")
            return
        left_s = sharpen_bgr(left_bgr.copy())
        right_s = sharpen_bgr(right_bgr.copy())

        win = tk.Toplevel(self.root)
        win.title("Verification")
        win.minsize(900, 600)
        bold = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        header = tk.Label(win, text="\n".join(header_lines), font=bold, justify=tk.LEFT)
        header.pack(fill=tk.X, padx=12, pady=10)

        img_frame = ttk.Frame(win)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        def bgr_to_tk_thumb(bgr, max_w=430, max_h=480):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(pil)

        tk_l = bgr_to_tk_thumb(left_s)
        tk_r = bgr_to_tk_thumb(right_s)

        lf = ttk.LabelFrame(img_frame, text=left_caption)
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        lb = ttk.Label(lf, image=tk_l, cursor="hand2")
        lb.pack(padx=6, pady=6)
        rf = ttk.LabelFrame(img_frame, text=right_caption)
        rf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        rb = ttk.Label(rf, image=tk_r, cursor="hand2")
        rb.pack(padx=6, pady=6)
        win._tk_l = tk_l
        win._tk_r = tk_r

        def open_zoom(img_bgr, cap):
            zw = tk.Toplevel(win)
            zw.title(cap)
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            sw, sh = zw.winfo_screenwidth(), zw.winfo_screenheight()
            pil.thumbnail((int(sw * 0.9), int(sh * 0.85)), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(pil)
            ttk.Label(zw, image=ph).pack(padx=8, pady=8)
            zw._ph = ph

        lb.bind("<Button-1>", lambda _e: open_zoom(left_s, left_caption))
        rb.bind("<Button-1>", lambda _e: open_zoom(right_s, right_caption))

        def save_result():
            out_path = filedialog.asksaveasfilename(
                parent=win,
                title="Save comparison as JPG",
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("All files", "*.*")],
                initialfile=default_filename,
            )
            if not out_path:
                return
            h1, w1 = left_s.shape[:2]
            h2, w2 = right_s.shape[:2]
            target_h = max(h1, h2)
            scale1 = target_h / h1
            scale2 = target_h / h2
            r1 = cv2.resize(left_s, (int(w1 * scale1), target_h), interpolation=cv2.INTER_AREA)
            r2 = cv2.resize(right_s, (int(w2 * scale2), target_h), interpolation=cv2.INTER_AREA)
            combined = np.hstack([r1, r2])
            cv2.imwrite(out_path, combined)
            messagebox.showinfo("Saved", f"Saved:\n{out_path}", parent=win)

        btn_row = ttk.Frame(win)
        btn_row.pack(fill=tk.X, pady=10)
        ttk.Button(btn_row, text="Save Result", command=save_result).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side=tk.LEFT, padx=8)

    def _restore_verify_buttons(self):
        self.btn_verify_eticket.config(state=tk.NORMAL, text="Verify with Eticket (Selected Client)")
        self.btn_verify_filter.config(state=tk.NORMAL, text="Verify with Filter (Selected Client)")
        self.client_verify_in_progress = False

    def verify_selected_client_with_eticket(self):
        if self.client_verify_in_progress:
            return
        row = self._get_selected_client_row()
        if not row:
            messagebox.showwarning("No Selection", "Select a client to verify.")
            return
        if not self.eticket_index:
            messagebox.showwarning(
                "Etickets Not Indexed",
                "Select an Etickets folder in the Matcher tab first so references are indexed.",
            )
            return

        _client_id, name, eticket_path, _filtre_path, ref, _serial, _date_added = row
        if not eticket_path or not os.path.isfile(eticket_path):
            self.set_client_status(
                "Cannot verify: Maquette Eticket is not set or the file is missing.\n"
                "Browse to the correct image and use Update Selected."
            )
            return

        self.client_verify_in_progress = True
        self.btn_verify_eticket.config(state=tk.DISABLED, text="Verifying...")
        self.btn_verify_filter.config(state=tk.DISABLED)
        self.set_global_status("Verifying with eticket...")

        ref_n = normalize_reference(ref)

        def task():
            best, _top3, match_type = self.get_matches_with_recheck(ref_n)
            return {
                "best": best,
                "match_type": match_type,
                "ref_n": ref_n,
                "name": name,
                "eticket_path": eticket_path,
            }

        def on_success(res):
            self._restore_verify_buttons()
            self.set_global_status(
                "OCR Ready âœ…"
            )
            self._finish_verify_eticket_ui(res)

        def on_error(exc):
            self._restore_verify_buttons()
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("Verification Error", str(exc))

        self.run_in_background(task, on_success, on_error)

    def _finish_verify_eticket_ui(self, res):
        name = res["name"]
        eticket_path = res["eticket_path"]
        ref_n = res["ref_n"]
        best = res["best"]
        match_type = res["match_type"]

        self.show_image(eticket_path, self.client_maquette_preview, "client_filter")
        if best is None:
            self.client_eticket_preview.config(image="", text="No matched eticket preview")
            self.set_client_status(
                f"Client: {name}\n"
                f"Reference: {ref_n}\n"
                "âŒ NO MATCH FOUND (red)\n"
                "No matching eticket found in the folder"
            )
            return

        shown_ref = best["reference"] if best["reference"] else "NO_REF"
        score = best.get("score", 0)
        maquette_serial = re.sub(r"[^0-9]", "", (self.client_serial_var.get() or ""))
        if not maquette_serial:
            try:
                maquette_serial = self._extract_serial_step4_cached(eticket_path)
            except Exception:
                maquette_serial = ""
        matched_serial = ""
        try:
            matched_serial = self._extract_serial_step4_cached(best["path"])
        except Exception:
            matched_serial = ""
        serial_ok = bool(maquette_serial and matched_serial and maquette_serial == matched_serial)
        diff_count = reference_difference_count(ref_n, shown_ref)
        severe_warning = diff_count > 2
        confirmed_match = match_type in ("âœ… MATCH", "âš ï¸ CLOSE MATCH")
        if match_type == "âœ… MATCH":
            guidance = "Confirmed exact reference match."
        elif match_type == "âš ï¸ CLOSE MATCH":
            guidance = "Reference is similar but not exact â€” please verify manually."
        elif match_type == "âŒ NO EXACT MATCH":
            guidance = (
                f"No exact match found. Closest match is {shown_ref} with only {score}% similarity â€” this may be incorrect."
            )
        else:
            guidance = "No matching eticket found in the folder."
        header_lines = [
            f"Verify with Eticket â€” {name}",
            f"{match_type} {'(orange warning)' if match_type == 'âš ï¸ CLOSE MATCH' else '(red warning)' if match_type != 'âœ… MATCH' else '(green)'}",
            f"Client reference: {ref_n}",
            f"Matched eticket reference: {shown_ref}",
            f"Serial compare: {'MATCH ✅' if serial_ok else 'MISMATCH ❌'}  [{maquette_serial or 'N/A'} vs {matched_serial or 'N/A'}]",
            f"Similarity score: {score}%",
            guidance,
        ]
        if severe_warning:
            header_lines.append(
                f"âš ï¸ WARNING: References differ significantly â€” {ref_n} vs {shown_ref} â€” DO NOT confirm this match"
            )
        if confirmed_match:
            self._show_verification_popup(
                eticket_path,
                best["path"],
                header_lines,
                "Maquette Eticket",
                f"Matched eticket: {os.path.basename(best['path'])}",
                f"verify_eticket_{name}_{ref_n}.jpg",
            )
            self.show_image(best["path"], self.client_eticket_preview, "client_eticket")
        else:
            self.client_eticket_preview.config(image="", text="No confirmed matched eticket preview")
        self.set_client_status(
            f"Client: {name}\n"
            f"Reference: {ref_n}\n"
            f"Result: {match_type}\n"
            f"{'Matched eticket' if confirmed_match else 'Closest eticket'}: {os.path.basename(best['path'])}\n"
            f"Matched eticket reference: {shown_ref}\n"
            f"Serial compare: {'MATCH ✅' if serial_ok else 'MISMATCH ❌'}  [{maquette_serial or 'N/A'} vs {matched_serial or 'N/A'}]\n"
            f"Confidence: {best['score']}%\n"
            f"{guidance}"
            + (
                f"\nâš ï¸ WARNING: References differ significantly â€” {ref_n} vs {shown_ref} â€” DO NOT confirm this match"
                if severe_warning
                else ""
            )
        )

    def _prompt_manual_reference(self, title, prompt):
        entered = simpledialog.askstring(title, prompt, parent=self.root)
        if not entered:
            return ""
        return normalize_reference(entered)

    def _get_filter_reference_for_path(self, filter_path):
        if not os.path.isfile(filter_path):
            return "", "Missing filter file."
        image = load_image_bgr(filter_path)
        if image is None:
            return "", "Could not load filter image."
        mtime = int(os.path.getmtime(filter_path))
        cache_key = make_filter_cache_key(filter_path, mtime)
        auth = authoritative_filename_reference(filter_path)

        cached = cache_key in self.filter_cache and self.filter_cache[cache_key].get("reference")
        if cached:
            ref = self.filter_cache[cache_key].get("reference", "")
            if auth and normalize_reference(ref) != auth:
                del self.filter_cache[cache_key]
                self.save_filter_cache()
                cached = False
            else:
                raw_text = self.filter_cache[cache_key].get("raw_text", "")
                ref, raw_text = reconcile_ocr_ref_with_known_filename(filter_path, ref, raw_text)
                if ref != self.filter_cache[cache_key].get("reference", ""):
                    self.filter_cache[cache_key] = {"reference": ref, "raw_text": raw_text}
                    self.save_filter_cache()
                return ref, raw_text

        if auth:
            ref = auth
            raw_text = f"Authoritative filename reference: {auth}\n"
            ocr_r, ocr_t = ocr_filter_reference(image)
            raw_text += "=== OCR trace (audit only; matching uses filename) ===\n" + (ocr_t or "")
            if ocr_r and normalize_reference(ocr_r) != auth:
                raw_text += (
                    f"\nOCR disagrees ({normalize_reference(ocr_r)}); matching uses filename."
                )
            self.filter_cache[cache_key] = {"reference": ref, "raw_text": raw_text}
            self.save_filter_cache()
            return ref, raw_text

        ref, raw_text = ocr_filter_reference(image)
        if not ref:
            ref = infer_reference_from_filename(filter_path)
            if ref:
                raw_text = (raw_text + "\n" if raw_text else "") + f"Filename fallback: {ref}"
        ref, raw_text = reconcile_ocr_ref_with_known_filename(filter_path, ref, raw_text)
        self.filter_cache[cache_key] = {"reference": ref, "raw_text": raw_text}
        self.save_filter_cache()
        return ref, raw_text

    def _compute_orb_visual_similarity(self, path_a, path_b, include_match_image=True):
        img_a = load_image_bgr(path_a)
        img_b = load_image_bgr(path_b)
        if img_a is None or img_b is None:
            raise ValueError("Could not load one or both images for visual matching.")

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        if not include_match_image:
            def _down512(gray):
                h, w = gray.shape[:2]
                m = max(h, w)
                if m <= 512:
                    return gray
                s = 512.0 / m
                return cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            gray_a = _down512(gray_a)
            gray_b = _down512(gray_b)

        nfeat = 1500 if include_match_image else 650
        orb = cv2.ORB_create(nfeatures=nfeat)
        kp1, des1 = orb.detectAndCompute(gray_a, None)
        kp2, des2 = orb.detectAndCompute(gray_b, None)
        if des1 is None or des2 is None or not kp1 or not kp2:
            return 0, None, 0, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            return 0, None, len(kp1), len(kp2)
        matches = sorted(matches, key=lambda m: m.distance)
        good = [m for m in matches if m.distance < 60]
        denom = max(1, min(len(kp1), len(kp2)))
        score = int(min(100, round((len(good) / denom) * 100)))

        if not include_match_image:
            return score, None, len(kp1), len(kp2)

        show_n = min(80, len(matches))
        match_img = cv2.drawMatches(
            img_a, kp1, img_b, kp2, matches[:show_n], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return score, match_img, len(kp1), len(kp2)

    def _compute_orb_visual_similarity_cached_score(self, path_a, path_b):
        """Fast cached ORB score for matcher fallback."""
        try:
            ma = int(os.path.getmtime(path_a))
            mb = int(os.path.getmtime(path_b))
        except OSError:
            ma, mb = 0, 0
        key = f"{path_a}|{ma}|{path_b}|{mb}"
        if key in self.visual_score_cache:
            return self.visual_score_cache[key]
        try:
            score, _img, _k1, _k2 = self._compute_orb_visual_similarity(
                path_a, path_b, include_match_image=False
            )
        except Exception:  # noqa: BLE001
            score = 0
        self.visual_score_cache[key] = int(score)
        return int(score)

    def _show_cv2_image_on_label(self, image_bgr, target_label, which):
        if image_bgr is None:
            target_label.config(image="", text="No visual match preview")
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((700, 360))
        tk_img = ImageTk.PhotoImage(pil_img)
        target_label.config(image=tk_img, text="")
        if which == "client_visual_match":
            self.client_visual_match_photo = tk_img

    def verify_selected_client_with_filter(self):
        if self.client_verify_in_progress:
            return
        row = self._get_selected_client_row()
        if not row:
            messagebox.showwarning("No Selection", "Select a client to verify.")
            return
        if not self.filters_folder or not self.filter_images:
            messagebox.showwarning(
                "Filters Not Loaded",
                "Select a Filters folder in the Matcher tab first.",
            )
            return

        _client_id, name, _eticket_path, maquette_filtre_path, client_ref, _serial, _date_added = row
        if not maquette_filtre_path or not os.path.isfile(maquette_filtre_path):
            messagebox.showwarning(
                "Missing Maquette Filtre",
                "Set a valid Maquette Filtre photo for this client before verifying with filters.",
            )
            return

        self.client_verify_in_progress = True
        self.btn_verify_eticket.config(state=tk.DISABLED, text="Verifying...")
        self.btn_verify_filter.config(state=tk.DISABLED, text="Verifying...")
        self.set_global_status("Verifying with filter...")

        self.show_image(maquette_filtre_path, self.client_maquette_preview, "client_filter")

        self.set_client_status("Verify with Filter in progress. Please wait...")

        def task():
            maquette_ref = normalize_reference(client_ref)
            maquette_raw = ""
            if not maquette_ref:
                maquette_img = load_image_bgr(maquette_filtre_path)
                if maquette_img is not None:
                    maquette_ref, maquette_raw = ocr_filter_reference(maquette_img)
                    maquette_ref, maquette_raw = reconcile_ocr_ref_with_known_filename(
                        maquette_filtre_path, maquette_ref, maquette_raw
                    )
                if not maquette_ref:
                    maquette_ref = infer_reference_from_filename(maquette_filtre_path)

            filters_data = []
            for filter_path in self.filter_images:
                f_ref, f_raw = self._get_filter_reference_for_path(filter_path)
                filters_data.append((filter_path, normalize_reference(f_ref), f_raw))

            if maquette_ref:
                exact = [x for x in filters_data if x[1] == maquette_ref and x[1]]
                if exact:
                    chosen = exact[0]
                    ref_score = 100
                    ref_match_type = "Exact"
                else:
                    scored = sorted(
                        ((fuzz.ratio(maquette_ref, x[1]), x) for x in filters_data if x[1]),
                        key=lambda t: t[0],
                        reverse=True,
                    )
                    if scored:
                        ref_score = int(round(scored[0][0]))
                        chosen = scored[0][1]
                        ref_match_type = "Fuzzy"
                    else:
                        chosen = (self.filter_images[0], "", "")
                        ref_score = 0
                        ref_match_type = "No OCR reference from filters"
            else:
                chosen = (self.filter_images[0], "", "")
                ref_score = 0
                ref_match_type = "No maquette OCR reference"

            filter_path, filter_ref, filter_raw = chosen
            visual_score, match_img, kp1_count, kp2_count = self._compute_orb_visual_similarity(
                maquette_filtre_path, filter_path
            )

            return {
                "name": name,
                "maquette_filtre_path": maquette_filtre_path,
                "maquette_ref": maquette_ref,
                "maquette_raw": maquette_raw,
                "filter_path": filter_path,
                "filter_ref": filter_ref,
                "filter_raw": filter_raw,
                "ref_score": ref_score,
                "ref_match_type": ref_match_type,
                "visual_score": visual_score,
                "match_img": match_img,
                "kp1_count": kp1_count,
                "kp2_count": kp2_count,
            }

        def on_success(result):
            maquette_ref = result["maquette_ref"]
            if not maquette_ref:
                entered = self._prompt_manual_reference(
                    "Manual Maquette Reference",
                    "OCR could not read maquette reference.\nType it manually to continue:"
                )
                if not entered:
                    self._restore_verify_buttons()
                    self.set_global_status(
                        "OCR Ready âœ…"
                    )
                    self.set_client_status("Verification stopped: maquette reference missing.")
                    return
                maquette_ref = entered
                result["maquette_ref"] = maquette_ref
                result["ref_score"] = int(round(fuzz.ratio(maquette_ref, result["filter_ref"]))) if result["filter_ref"] else 0
                result["ref_match_type"] = "Manual vs OCR"

            if not result["filter_ref"]:
                entered_filter = self._prompt_manual_reference(
                    "Manual Filter Reference",
                    "OCR could not read filter reference.\nType matched filter reference manually:"
                )
                if entered_filter:
                    result["filter_ref"] = entered_filter
                    result["ref_score"] = int(round(fuzz.ratio(maquette_ref, entered_filter)))
                    result["ref_match_type"] = "Manual fallback"

            ref_pass = result["ref_score"] >= 80
            visual_pass = result["visual_score"] >= 70
            # Exact same reference (or 100% score): trust text match; ORB often fails diagram vs photo.
            reference_only_ok = result.get("ref_match_type") == "Exact" or result["ref_score"] >= 100
            verified = ref_pass and (visual_pass or reference_only_ok)
            verdict = "VERIFIED âœ…" if verified else "NOT VERIFIED âŒ"

            header_lines = [
                f"Verify with Filter â€” {result['name']}",
                verdict,
                f"Reference match: {result['ref_match_type']}",
                f"Maquette ref: {result['maquette_ref'] or 'N/A'}  |  Filter ref: {result['filter_ref'] or 'N/A'}",
                f"Reference score: {result['ref_score']}%  |  Visual (ORB): {result['visual_score']}%",
            ]
            if ref_pass and reference_only_ok and not visual_pass:
                header_lines.append("(Reference exact/100% â€” visual ORB optional)")
            self._show_verification_popup(
                result["maquette_filtre_path"],
                result["filter_path"],
                header_lines,
                "Maquette Filtre",
                f"Matched filter: {os.path.basename(result['filter_path'])}",
                f"verify_filter_{result['name']}_{result['maquette_ref'] or 'ref'}.jpg",
            )
            self.show_image(result["filter_path"], self.client_eticket_preview, "client_eticket")
            self._show_cv2_image_on_label(result["match_img"], self.client_eticket_preview, "client_visual_match")

            self.set_client_status(
                f"Client: {result['name']}\n"
                f"Mode: Verify with Filter\n\n"
                f"Reference comparison ({result['ref_match_type']}):\n"
                f"- Maquette reference: {result['maquette_ref'] or 'N/A'}\n"
                f"- Filter reference: {result['filter_ref'] or 'N/A'}\n"
                f"- Reference score: {result['ref_score']}% (pass >= 80%)\n\n"
                f"Visual comparison (ORB keypoints):\n"
                f"- Visual score: {result['visual_score']}%"
                f"{' (optional when reference is Exact/100%)' if reference_only_ok else ' (pass >= 70% if reference not exact)'}\n"
                f"- Keypoints (maquette/filter): {result['kp1_count']}/{result['kp2_count']}\n\n"
                f"Matched filter: {os.path.basename(result['filter_path'])}\n"
                f"Final verdict: {verdict}"
            )
            self._restore_verify_buttons()
            self.set_global_status(
                "OCR Ready âœ…"
            )

        def on_error(exc):
            self._restore_verify_buttons()
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("Verification Error", f"Verify with Filter failed:\n{exc}")

        self.run_in_background(task, on_success, on_error)

    def _set_status_tone(self, tone="neutral"):
        st = getattr(self, "status_text", None)
        if st is None:
            return
        try:
            if tone == "match":
                st.config(bg=self._status_match_bg, fg="#08330b")
            elif tone == "mismatch":
                st.config(bg=self._status_mismatch_bg, fg="#4a1010")
            else:
                st.config(
                    bg=getattr(self, "_status_neutral_bg", "white"),
                    fg=getattr(self, "_status_neutral_fg", "black"),
                )
        except Exception:
            pass

    def set_status(self, text, tone="neutral"):
        self._set_status_tone(tone)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, text)
        self._apply_status_word_tags()

    def _apply_status_word_tags(self):
        """Color only MATCH/MISMATCH words inside the result panel text."""
        st = getattr(self, "status_text", None)
        if st is None:
            return
        try:
            content = st.get("1.0", tk.END)
        except Exception:
            return
        try:
            st.tag_configure("status_word_match", foreground="#0b8f2f")
            st.tag_configure("status_word_mismatch", foreground="#c92424")
            st.tag_remove("status_word_match", "1.0", tk.END)
            st.tag_remove("status_word_mismatch", "1.0", tk.END)
            for m in re.finditer(r"\bMISMATCH\b", content):
                st.tag_add("status_word_mismatch", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
            for m in re.finditer(r"\bMATCH\b", content):
                st.tag_add("status_word_match", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
        except Exception:
            pass

    def note_live_ocr_compare_status(self, line):
        """Append one line from the OpenCV Live OCR thread (throttled before scheduling)."""
        st = getattr(self, "status_text", None)
        mw = getattr(self, "matcher_window", None)
        if st is None or mw is None:
            return
        try:
            if not mw.winfo_exists() or not st.winfo_exists():
                return
        except Exception:
            return
        st.insert(tk.END, "\n" + line)
        self._apply_status_word_tags()
        st.see(tk.END)

    def _ensure_dashboard_feed_file(self):
        feed = getattr(self, "dashboard_feed_file", "") or ""
        if not feed:
            return
        try:
            os.makedirs(os.path.dirname(feed), exist_ok=True)
            if not os.path.isfile(feed):
                with open(feed, "w", encoding="utf-8") as f:
                    json.dump({"events": []}, f, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _open_dashboard_url(self, url):
        """Open http(s) URL in default browser (Windows: os.startfile is most reliable)."""
        if sys.platform == "win32":
            try:
                os.startfile(url)
                return
            except OSError:
                pass
        try:
            webbrowser.open(url)
        except Exception:
            messagebox.showinfo(
                "QC Dashboard",
                "Open this address in your browser:\n{}".format(url),
            )

    def open_local_qc_dashboard(self):
        """Start local_dashboard_server.py if needed and open the dashboard in the browser."""
        import urllib.error
        import urllib.request

        base = os.path.dirname(os.path.abspath(__file__))
        root = self.root

        def _probe_port(port):
            try:
                url = "http://127.0.0.1:{}/api/exceptions".format(int(port))
                req = urllib.request.Request(url, headers={"Connection": "close"})
                # Do not use system proxy (some corporate proxies block localhost).
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                opener.open(req, timeout=1.0)
                return True
            except (urllib.error.URLError, OSError, ValueError):
                return False

        def _worker():
            found = None
            for p in range(8000, 8011):
                if _probe_port(p):
                    found = p
                    break
            if found is not None:
                url = "http://127.0.0.1:{}/".format(found)
                root.after(0, lambda u=url: self._open_dashboard_url(u))
                return

            script = os.path.join(base, "local_dashboard_server.py")
            if not os.path.isfile(script):
                root.after(
                    0,
                    lambda: messagebox.showerror(
                        "QC Dashboard",
                        "Could not find local_dashboard_server.py next to the application.",
                    ),
                )
                return
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                subprocess.Popen(
                    [sys.executable, script],
                    cwd=base,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=0,
                )
            except Exception as exc:
                root.after(
                    0,
                    lambda: messagebox.showerror(
                        "QC Dashboard",
                        "Could not start the local server:\n{}".format(exc),
                    ),
                )
                return

            for _ in range(60):
                time.sleep(0.1)
                for p in range(8000, 8011):
                    if _probe_port(p):
                        url = "http://127.0.0.1:{}/".format(p)
                        root.after(0, lambda u=url: self._open_dashboard_url(u))
                        return

            root.after(
                0,
                lambda: messagebox.showinfo(
                    "QC Dashboard",
                    "The local server did not respond in time.\n\n"
                    "In PowerShell, run:\n"
                    '  Set-Location "{}"\n'
                    "  python local_dashboard_server.py\n\n"
                    "Then open the printed URL in your browser.\n"
                    "Do not open dashboard\\index.html as a file.".format(base),
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _presence_miss_notes_from_pipeline(self, pipeline_text):
        notes = []
        text = pipeline_text or ""
        for line in text.splitlines():
            m = re.match(r"\s*-\s*([^:]+):\s*(.+)$", line.strip())
            if not m:
                continue
            source = (m.group(1) or "").strip()
            tail = (m.group(2) or "").strip().upper()
            for tok, val in re.findall(r"(EAC|HIFI|REF|PHRASE)\s*=\s*(YES|NO)", tail):
                if val == "NO":
                    notes.append(f"{source}: {tok} missing")
        return notes

    def _record_dashboard_exception(
        self,
        station,
        target_ref,
        detected_ref,
        eac_ok,
        hifi_ok,
        reasons,
        result_label="FAIL",
    ):
        if not reasons:
            return
        self._ensure_dashboard_feed_file()
        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "station": (station or "HIFI-MATCHER"),
            "target_serial": normalize_reference(target_ref or ""),
            "detected": normalize_reference(detected_ref or ""),
            "eac_ok": bool(eac_ok),
            "hifi_ok": bool(hifi_ok),
            "result": (result_label or "FAIL"),
            "reasons": [str(r) for r in reasons if str(r).strip()],
        }
        try:
            with open(self.dashboard_feed_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
        events = payload.get("events")
        if not isinstance(events, list):
            events = []
        events.insert(0, event)
        payload["events"] = events[:300]
        try:
            with open(self.dashboard_feed_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _matcher_eticket_compare_footer(self, live_filter_ref):
        """When an etiquette was chosen in the matcher, append live-vs-target lines to status."""
        tgt = normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or "")
        if not tgt:
            return ""
        live = normalize_reference(live_filter_ref or "")
        path = getattr(self, "matcher_selected_eticket_path", "") or ""
        bn = os.path.basename(path) if path else "(none)"
        exact = bool(live and live == tgt)
        score = eticket_compatibility_vs_filter(live, tgt) if live else 0
        zero_nine_hint = ""
        if (not exact) and self._is_zero_nine_confusion_pair(live, tgt):
            zero_nine_hint = "Possible OCR confusion: 0 <-> 9"
        return (
            "\n\n--- Compare to selected etiquette ---\n"
            f"Selected file: {bn}\n"
            f"Selected etiquette reference: {tgt}\n"
            f"Live / filter reference used: {live or '(none yet)'}\n"
            f"Exact match with etiquette: {'YES' if exact else 'NO'}\n"
            f"Text compatibility score (filter vs etiquette): {score}%\n"
            + (f"{zero_nine_hint}\n" if zero_nine_hint else "")
        )

    def _is_zero_nine_confusion_pair(self, a_ref, b_ref):
        """True when refs differ only by one/few 0<->9 swaps (same prefix)."""
        a = normalize_reference(a_ref or "")
        b = normalize_reference(b_ref or "")
        if not a or not b:
            return False
        pa, da = split_reference(a)
        pb, db = split_reference(b)
        if pa != pb or not da or not db or len(da) != len(db):
            return False
        diffs = [(x, y) for x, y in zip(da, db) if x != y]
        if not diffs or len(diffs) > 2:
            return False
        return all((x, y) in (("0", "9"), ("9", "0")) for x, y in diffs)

    def _populate_matcher_eticket_selector(self):
        combo = getattr(self, "eticket_selector", None)
        if combo is None:
            return
        if not self.eticket_index:
            combo.set("")
            combo["values"] = []
            return
        values = []
        for item in self.eticket_index:
            p = item.get("path", "")
            r = normalize_reference(item.get("reference", "") or "")
            label = os.path.basename(p) if p else "?"
            if r:
                label = f"{label}  |  {r}"
            values.append(label)
        combo["values"] = values
        selected_path = getattr(self, "matcher_selected_eticket_path", "") or ""
        selected_idx = -1
        if selected_path:
            try:
                selected_idx = next(
                    i for i, it in enumerate(self.eticket_index) if it.get("path", "") == selected_path
                )
            except StopIteration:
                selected_idx = -1
        try:
            if selected_idx >= 0:
                combo.current(selected_idx)
            elif combo.get() and combo.get() in values:
                pass
            else:
                combo.set("")
        except Exception:
            combo.set("")

    def _redraw_matcher_selected_eticket_overlay(self):
        """Show green/red OCR boxes on the currently selected etiquette preview."""
        path = getattr(self, "matcher_selected_eticket_path", "") or ""
        if not path or not os.path.isfile(path):
            return False
        image = load_image_bgr(path)
        if image is None or image.size == 0:
            return False
        self._refresh_known_references()
        try:
            vis, _overlay_ref = self._draw_ocr_overlay(image, 0.0)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            lw = max(320, int(self.eticket_preview_label.winfo_width() or 520))
            lh = max(220, int(self.eticket_preview_label.winfo_height() or 360))
            pil_img.thumbnail((lw, lh), Image.Resampling.BILINEAR)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.eticket_photo = tk_img
            self.eticket_preview_label.config(image=tk_img, text="Selected etiquette (OCR overlay)")
            return True
        except Exception as exc:  # noqa: BLE001
            self.eticket_preview_label.config(
                image="",
                text=f"Could not draw OCR overlay.\n{type(exc).__name__}: {exc}",
            )
            return False

    def _is_eticket_preview_locked(self):
        path = getattr(self, "matcher_selected_eticket_path", "") or ""
        return bool(path and os.path.isfile(path))

    def on_matcher_eticket_selected(self, _event=None):
        combo = getattr(self, "eticket_selector", None)
        if combo is None or not self.eticket_index:
            return
        idx = combo.current()
        if idx < 0 or idx >= len(self.eticket_index):
            return
        item = self.eticket_index[idx]
        path = item.get("path", "")
        if not path or not os.path.isfile(path):
            messagebox.showwarning("Eticket", "Image file not found for this entry.")
            return
        image = load_image_bgr(path)
        if image is None or image.size == 0:
            messagebox.showerror("Eticket", "Could not load the etiquette image.")
            return
        ref = normalize_reference(item.get("reference", "") or "")
        if not ref:
            try:
                ref_full, raw_full = ocr_eticket_reference(image)
                ref = normalize_reference(ref_full or "")
                if ref:
                    item["reference"] = ref
                    if raw_full:
                        item["raw_text"] = (raw_full or "").strip()
                    self._populate_matcher_eticket_selector()
            except Exception:
                ref = ""
        self.matcher_selected_eticket_path = path
        self.matcher_selected_eticket_ref = ref
        self._redraw_matcher_selected_eticket_overlay()
        raw_txt = (item.get("raw_text") or "").strip()
        disp_ref = ref or "(no reference extracted yet — re-index or use Refresh Cache)"
        self.set_status(
            "Selected etiquette for live comparison.\n"
            f"Path: {path}\n"
            f"Reference (index/OCR): {disp_ref}\n"
            + (f"Index / OCR trace:\n{raw_txt}\n" if raw_txt else "")
            + "\nOpen Live Camera Auto on the filter; results below will include a "
            '"Compare to selected etiquette" section after each match attempt.'
        )
        self._warm_presence_cache_context()

    def _ensure_selected_eticket_reference(self):
        """Sync matcher_selected_eticket_* from combobox if not initialized yet."""
        if normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or ""):
            return
        combo = getattr(self, "eticket_selector", None)
        if combo is None or not self.eticket_index:
            return
        idx = combo.current()
        if idx < 0 and combo.get():
            try:
                idx = next(i for i, it in enumerate(self.eticket_index) if os.path.basename(it.get("path", "")) in combo.get())
            except Exception:
                idx = -1
        if idx < 0 or idx >= len(self.eticket_index):
            return
        item = self.eticket_index[idx]
        p = item.get("path", "") or ""
        r = normalize_reference(item.get("reference", "") or "")
        if (not r) and p and os.path.isfile(p):
            img = load_image_bgr(p)
            if img is not None and img.size > 0:
                try:
                    r1, _raw = ocr_eticket_reference_fast(img, timeout_sec=2.0)
                except Exception:
                    r1 = ""
                if not r1:
                    try:
                        r1 = extract_reference_maquette_eticket_tesseract(img)
                    except Exception:
                        r1 = ""
                r = normalize_reference(r1 or "")
                if r:
                    item["reference"] = r
        self.matcher_selected_eticket_path = p
        self.matcher_selected_eticket_ref = r

    def browse_matcher_serial_photo(self):
        photo = filedialog.askopenfilename(
            title="Select Eticket Serial Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.heic"),
                ("All files", "*.*"),
            ],
        )
        if not photo:
            return
        if not os.path.isfile(photo):
            messagebox.showwarning("Serial Photo", "Selected file does not exist.")
            return
        self.matcher_selected_eticket_serial_path = photo
        k = self._serial_cache_key(photo)
        if k is not None:
            self.step4_serial_cache.pop(k, None)
        if getattr(self, "matcher_serial_photo_var", None) is not None:
            self.matcher_serial_photo_var.set(photo)
        self.set_status(
            "Selected serial photo for Step 4 comparison.\n"
            f"Path: {photo}\n"
            "Step 4 uses this image to read selected-eticket serial."
        )

    def test_matcher_serial_photo_now(self):
        photo = getattr(self, "matcher_selected_eticket_serial_path", "") or ""
        if not photo or not os.path.isfile(photo):
            messagebox.showwarning(
                "Serial Photo",
                "Select a serial photo first with Browse Serial Photo.",
            )
            return
        self.set_global_status("Testing eticket serial photo OCR...")

        def task():
            ser = self._extract_serial_step4_cached(photo, force_refresh=True)
            dbg = ""
            if not ser:
                try:
                    img = load_image_bgr(photo)
                    dbg = extract_eticket_serial_number_step4_debug_digits(img) if img is not None else ""
                except Exception:
                    dbg = ""
            return {"photo": photo, "serial": ser, "debug_digits": dbg}

        def on_success(res):
            self.set_global_status("OCR Ready ✅")
            serial = (res.get("serial") or "").strip()
            if serial:
                self.set_status(
                    "Serial photo test result.\n"
                    f"Path: {res.get('photo', '')}\n"
                    f"Detected serial: {serial}"
                )
            else:
                debug_digits = (res.get("debug_digits") or "").strip()
                if debug_digits:
                    self.set_status(
                        "Serial photo test result.\n"
                        f"Path: {res.get('photo', '')}\n"
                        "No valid EAN-13 detected (result: N/A).\n"
                        f"Debug OCR digits (best guess): {debug_digits}"
                    )
                else:
                    self.set_status(
                        "Serial photo test result.\n"
                        f"Path: {res.get('photo', '')}\n"
                        "No valid EAN-13 detected (result: N/A)."
                    )

        def on_error(exc):
            self.set_global_status("OCR Ready ✅")
            messagebox.showerror("Serial Photo Test", str(exc))

        self.run_in_background(task, on_success, on_error)

    def set_busy(self, busy, status_message=None, global_status=None):
        self.is_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.select_filters_btn.config(state=state)
        self.select_etickets_btn.config(state=state)
        self.refresh_eticket_cache_btn.config(state=state)
        self.find_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.capture_cam_btn.config(state=state)
        self.live_cam_btn.config(state=state)
        bsb = getattr(self, "browse_serial_photo_btn", None)
        if bsb is not None:
            bsb.config(state=state)
        tsb = getattr(self, "test_serial_photo_btn", None)
        if tsb is not None:
            tsb.config(state=state)
        self.filter_selector.config(state="disabled" if busy else "readonly")
        es = getattr(self, "eticket_selector", None)
        if es is not None:
            es.config(state="disabled" if busy else "readonly")
        if status_message:
            self.set_status(status_message)
        if global_status is not None:
            self.set_global_status(global_status)

    def run_in_background(self, task_fn, on_success, on_error):
        def worker():
            try:
                result = task_fn()
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda e=exc: on_error(e))
                return
            self.root.after(0, lambda res=result: on_success(res))

        threading.Thread(target=worker, daemon=True).start()

    def get_image_files(self, folder):
        files = []
        folder = os.path.abspath(folder)
        for name in sorted(os.listdir(folder)):
            path = os.path.abspath(os.path.join(folder, name))
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                files.append(path)
        return files

    def select_filters_folder(self):
        folder = filedialog.askdirectory(title="Select Filters Folder")
        if not folder:
            return
        self.filters_folder = folder
        self.filters_label.config(text=folder)
        self.filter_images = self.get_image_files(folder)
        self.load_filter_cache()
        self.filter_selector["values"] = [os.path.basename(p) for p in self.filter_images]
        if self.filter_images:
            self.filter_selector.current(0)
            self.on_filter_selected()
        else:
            self.filter_preview_label.config(text="No supported images found", image="")
        self.try_index_etickets()

    def select_etickets_folder(self):
        folder = filedialog.askdirectory(title="Select Etickets Folder")
        if not folder:
            return
        self.etickets_folder = folder
        self.etickets_label.config(text=folder)
        self.try_index_etickets()

    def try_index_etickets(self):
        if not self.etickets_folder:
            return
        self.set_busy(True, "Indexing etickets. Please wait...", global_status="Scanning etickets...")

        def on_success(_):
            self.set_busy(False)
            self.set_global_status(
                "OCR Ready âœ…"
            )
            self._populate_matcher_eticket_selector()
            self.set_status(
                f"Etickets indexed: {len(self.eticket_index)}\n"
                "Use filter image, Capture Camera, or Live Camera Auto."
            )

        def on_error(exc):
            self.set_busy(False)
            self.set_global_status(
                "OCR Ready âœ…"
            )
            messagebox.showerror("Indexing Error", f"Could not index etickets:\n{exc}")

        self.run_in_background(self.index_etickets, on_success, on_error)

    def _normalize_eticket_cache_paths(self, cache):
        """Migrate cache JSON keys to absolute paths so lookups survive folder moves."""
        if not cache or not self.etickets_folder:
            return cache
        folder = os.path.abspath(self.etickets_folder)
        out = {}
        for k, v in cache.items():
            ks = str(k)
            if "|" not in ks:
                out[ks] = v
                continue
            path_part, mtime_part = ks.rsplit("|", 1)
            if not os.path.isabs(path_part):
                path_part = os.path.abspath(os.path.join(folder, os.path.basename(path_part)))
            else:
                path_part = os.path.abspath(os.path.normpath(path_part))
            out[f"{path_part}|{mtime_part}"] = v
        return out

    def refresh_eticket_cache(self):
        if not self.etickets_folder:
            messagebox.showwarning("No folder", "Select an Etickets folder first.")
            return
        if not self.filters_folder:
            messagebox.showwarning("Matcher", "Select a Filters folder so etickets can be indexed.")
            return
        cache_path = os.path.join(self.etickets_folder, ETICKET_CACHE_FILE)
        try:
            if os.path.isfile(cache_path):
                os.remove(cache_path)
        except OSError:
            messagebox.showerror("Refresh Cache", "Could not delete the eticket cache file.")
            return
        self.try_index_etickets()

    def _ocr_eticket_reference_with_timeout(self, image_bgr, timeout_sec=6.0):
        """Bound eticket OCR time per image during indexing to keep UI responsive."""
        out = {"ref": "", "raw": ""}
        done = threading.Event()

        def run():
            try:
                r, raw = ocr_eticket_reference_fast(image_bgr, timeout_sec=timeout_sec)
            except Exception:  # noqa: BLE001
                r, raw = "", ""
            # Some labels (like SN920410 with large QR area) may fail fast path.
            # Try the robust banner extractor before giving up.
            if not r:
                try:
                    r2 = extract_reference_maquette_eticket_tesseract(image_bgr)
                except Exception:  # noqa: BLE001
                    r2 = ""
                if r2:
                    r = r2
                    raw = (raw + "\n" if raw else "") + f"banner_extractor_fallback: {r2}"
            out["ref"] = r or ""
            out["raw"] = raw or ""
            done.set()

        threading.Thread(target=run, daemon=True).start()
        done.wait(timeout=max(0.2, float(timeout_sec)))
        if not done.is_set():
            return "", ""
        return out["ref"], out["raw"]

    def index_etickets(self):
        cache_path = os.path.join(self.etickets_folder, ETICKET_CACHE_FILE)
        cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        cache = self._normalize_eticket_cache_paths(cache)

        eticket_images = self.get_image_files(self.etickets_folder)
        results = []
        updated_cache = {}
        if not eticket_images:
            self.eticket_index = []
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(updated_cache, f, ensure_ascii=True, indent=2)
            except Exception:
                pass
            return

        total = len(eticket_images)
        for i, path in enumerate(eticket_images, start=1):
            self.root.after(
                0,
                lambda ii=i, tt=total: self.set_global_status(f"Scanning etickets... {ii}/{tt}"),
            )
            path = os.path.abspath(os.path.normpath(path))
            mtime = int(os.path.getmtime(path))
            key = f"{path}|{mtime}"
            if key in cache and cache[key].get("reference"):
                ref = cache[key].get("reference", "")
                raw_text = cache[key].get("raw_text", "")
            else:
                # Fast path: if filename already contains a valid reference, use it immediately.
                inferred = infer_reference_from_filename(path)
                if inferred:
                    ref = inferred
                    raw_text = f"Filename fallback: {inferred}"
                else:
                    image = load_image_bgr(path)
                    if image is None:
                        continue
                    ref, raw_text = self._ocr_eticket_reference_with_timeout(image, timeout_sec=6.0)
            if not ref:
                inferred = infer_reference_from_filename(path)
                if inferred:
                    ref = inferred
                    raw_text = (raw_text + "\n" if raw_text else "") + f"Filename fallback: {inferred}"
            entry = {
                "key": key,
                "path": path,
                "reference": ref,
                "raw_text": (raw_text or "").strip(),
                "cache_entry": {"reference": ref, "raw_text": raw_text},
            }
            results.append(entry)
            updated_cache[key] = entry["cache_entry"]

        self.eticket_index = sorted(
            (
                {
                    "path": r["path"],
                    "reference": r["reference"],
                    "raw_text": r["raw_text"],
                }
                for r in results
            ),
            key=lambda x: x["path"],
        )
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(updated_cache, f, ensure_ascii=True, indent=2)
        except Exception:
            pass
        self._refresh_known_references()

    def load_filter_cache(self):
        self.filter_cache = {}
        if not self.filters_folder:
            return
        cache_path = os.path.join(self.filters_folder, FILTER_CACHE_FILE)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.filter_cache = json.load(f)
            except Exception:
                self.filter_cache = {}

    def save_filter_cache(self):
        if not self.filters_folder:
            return
        cache_path = os.path.join(self.filters_folder, FILTER_CACHE_FILE)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self.filter_cache, f, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def on_filter_selected(self, _event=None):
        idx = self.filter_selector.current()
        if idx < 0 or idx >= len(self.filter_images):
            return
        # Any manual dropdown choice disables camera override.
        self.camera_filter_path = ""
        self.show_image(self.filter_images[idx], self.filter_preview_label, "filter")

    def capture_filter_from_camera(self):
        if self.is_busy:
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open PC camera.")
            return
        self._configure_camera_capture(cap)
        frame = None
        got_good_frame = False
        try:
            # Warm up camera for a more stable frame.
            for _ in range(8):
                ok, img = cap.read()
                if ok and img is not None and img.size > 0:
                    frame = img
                    got_good_frame = True
                time.sleep(0.03)
        finally:
            cap.release()
        if frame is None or not got_good_frame:
            messagebox.showerror("Camera", "Could not capture image from camera.")
            return
        os.makedirs(self.camera_capture_dir, exist_ok=True)
        name = "filter_cam_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        out_path = os.path.join(self.camera_capture_dir, name)
        if not cv2.imwrite(out_path, frame):
            messagebox.showerror("Camera", "Could not save captured image.")
            return
        self.camera_filter_path = out_path
        # Clear combobox selection to make camera usage explicit in UI.
        self.filter_selector.set("")
        self.show_image(out_path, self.filter_preview_label, "filter")
        self.set_status(
            f"Camera capture ready:\n{os.path.basename(out_path)}\n"
            "Click Find Match to use this camera image."
        )

    def _cancel_live_camera_auto_restart(self):
        self._auto_restart_session = getattr(self, "_auto_restart_session", 0) + 1
        job = getattr(self, "_auto_restart_job", None)
        if job is not None:
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self._auto_restart_job = None

    def _schedule_live_camera_auto_restart(self, base_status_text, tone="neutral"):
        if not getattr(self, "_was_live_when_match_started", False):
            return
        self._cancel_live_camera_auto_restart()
        self._auto_restart_base_msg = base_status_text or ""
        self._auto_restart_tone = tone
        my_sid = self._auto_restart_session
        self._auto_restart_countdown_step(my_sid, 0)

    def _auto_restart_countdown_step(self, my_sid, step):
        if my_sid != getattr(self, "_auto_restart_session", 0):
            return
        if not getattr(self, "_was_live_when_match_started", False):
            return
        base = getattr(self, "_auto_restart_base_msg", "")
        tone = getattr(self, "_auto_restart_tone", "neutral")
        if step == 0:
            self.set_status(base + "\nNext filter in 3...", tone=tone)
            self._auto_restart_job = self.root.after(1000, lambda: self._auto_restart_countdown_step(my_sid, 1))
        elif step == 1:
            self.set_status(base + "\nNext filter in 2...", tone=tone)
            self._auto_restart_job = self.root.after(1000, lambda: self._auto_restart_countdown_step(my_sid, 2))
        elif step == 2:
            self.set_status(base + "\nNext filter in 1...", tone=tone)
            self._auto_restart_job = self.root.after(1000, lambda: self._auto_restart_countdown_step(my_sid, 3))
        else:
            self._auto_restart_job = None
            self._try_live_camera_auto_restart_launch(my_sid)

    def _try_live_camera_auto_restart_launch(self, my_sid):
        if my_sid != getattr(self, "_auto_restart_session", 0):
            return
        if not getattr(self, "_was_live_when_match_started", False):
            return
        if self.is_busy:
            self._auto_restart_job = self.root.after(1000, lambda: self._try_live_camera_auto_restart_launch(my_sid))
            return
        self._auto_restart_job = None
        if self.live_camera_mode:
            return
        if not self.etickets_folder or not self.eticket_index:
            base = getattr(self, "_auto_restart_base_msg", "")
            tone = getattr(self, "_auto_restart_tone", "neutral")
            self.set_status(
                base
                + "\n(Live camera not restarted: need Etickets folder indexed — use Live Camera Auto manually.)",
                tone=tone,
            )
            return
        self.toggle_live_camera_mode()
        if self.live_camera_mode:
            self.set_status(
                "Live camera ready — hold next filter in front of camera.",
                tone="neutral",
            )
        else:
            base = getattr(self, "_auto_restart_base_msg", "")
            tone = getattr(self, "_auto_restart_tone", "neutral")
            self.set_status(
                base + "\n(Live camera could not be restarted automatically — check camera or dialogs.)",
                tone=tone,
            )

    def _stop_live_camera(self):
        self._cancel_live_camera_auto_restart()
        self.live_camera_mode = False
        if self.live_camera_job is not None:
            try:
                self.root.after_cancel(self.live_camera_job)
            except Exception:
                pass
            self.live_camera_job = None
        if self.live_camera_cap is not None:
            try:
                self.live_camera_cap.release()
            except Exception:
                pass
            self.live_camera_cap = None
        self.live_ref_streak_ref = ""
        self.live_ref_streak_count = 0
        self.live_strict_busy = False
        self.live_strict_worker = None
        self.live_forced_ref = ""
        self.live_forced_ref_ts = 0.0
        try:
            if getattr(self, "live_cam_btn", None) is not None and self.live_cam_btn.winfo_exists():
                self.live_cam_btn.config(text="Live Camera Auto")
        except Exception:
            pass

    def toggle_live_camera_mode(self):
        if self.live_camera_mode:
            self._stop_live_camera()
            self.set_status("Live camera auto-match stopped.")
            return
        if not self.etickets_folder:
            messagebox.showwarning("Etickets", "Select Etickets folder first.")
            return
        if not self.eticket_index:
            self.try_index_etickets()
            messagebox.showinfo("Etickets", "Indexing started. Start live mode after it finishes.")
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Could not open PC camera.")
            return
        self._configure_camera_capture(cap)
        self.live_camera_cap = cap
        self.live_camera_mode = True
        try:
            stale_live = os.path.join(self.camera_capture_dir, "live_current.jpg")
            if os.path.isfile(stale_live):
                os.remove(stale_live)
        except Exception:
            pass
        self.camera_filter_path = ""
        self._live_overlay_freeze_after_mismatch = False
        self._live_last_mismatch_blocked_ref = ""
        self._live_last_mismatch_mono_ts = 0.0
        self._live_hold_mismatch_until_camera_restart = False
        self.live_last_match_ts = 0.0
        self.live_ocr_fps = 0.0
        self.live_ref_streak_ref = ""
        self.live_ref_streak_count = 0
        # Wait before first strict SA/SN read so exposure / focus can settle.
        self.live_strict_ready_ts = time.monotonic() + 3.5
        self.live_overlay_last_ts = 0.0
        self.live_overlay_cache = None
        self.live_frame_last_save_ts = 0.0
        self.live_strict_busy = False
        self.live_strict_worker = None
        self.live_forced_ref = ""
        self.live_forced_ref_ts = 0.0
        self.manual_ref_var.set("")
        self.live_cam_btn.config(text="Stop Live Camera")
        try:
            if self.eticket_preview_label.winfo_exists():
                if getattr(self, "matcher_selected_eticket_path", "") and os.path.isfile(
                    self.matcher_selected_eticket_path
                ):
                    self._redraw_matcher_selected_eticket_overlay()
                else:
                    self.eticket_preview_label.config(
                        image="",
                        text="Match appears here after SA/SN is detected and confirmed.",
                    )
        except Exception:
            pass
        live_msg = (
            "Live camera running.\n"
            "Hold the filter steady — detection will take a few seconds per check."
        )
        if normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or ""):
            live_msg += (
                "\n\nAn etiquette reference is selected — the status panel will append "
                '"Compare to selected etiquette" after each live match.'
            )
        self.set_status(live_msg, tone="neutral")
        self._live_camera_tick()

    def _draw_ocr_overlay(self, frame_bgr, ocr_fps):
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, ""
        vis = frame_bgr.copy()
        h_vis, w_vis = vis.shape[:2]
        lbl_scale = max(0.55, min(1.0, w_vis / 1200.0))
        lbl_thick = 2 if w_vis >= 900 else 1
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        try:
            data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
        except Exception:
            data = {"text": []}
        best_known_ref = ""
        ref_votes = {}
        n = len(data.get("text", []))
        raw_tokens = []
        line_groups = {}
        word_rows = []
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            raw_tokens.append(txt)
            try:
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])
                blk = int(data.get("block_num", [0] * n)[i])
                par = int(data.get("par_num", [0] * n)[i])
                line = int(data.get("line_num", [0] * n)[i])
            except Exception:
                continue
            if w <= 2 or h <= 2:
                continue
            cleaned = normalize_reference(re.sub(r"[^A-Z0-9]", "", txt.upper()))
            key = (blk, par, line)
            line_groups.setdefault(key, []).append((x, y, w, h, txt, cleaned))
            word_rows.append((x, y, w, h, txt, cleaned))

        # 1) Prefer grouped line boxes when SA/SN appears fragmented across curved text.
        for words in line_groups.values():
            words = sorted(words, key=lambda t: t[0])
            line_clean = "".join(w[5] for w in words if w[5])
            line_ref = ""
            for m in re.finditer(r"(SA|SN)([0-9OILBQGZ]{4,8})", line_clean):
                cand = correct_reference_parts(
                    (m.group(1) or "").upper(),
                    (m.group(2) or ""),
                    allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
                )
                cand = normalize_reference(cand)
                if cand and is_valid_hifi_reference(cand):
                    line_ref = cand
                    break
            if not line_ref:
                continue
            xs = [w[0] for w in words]
            ys = [w[1] for w in words]
            x2s = [w[0] + w[2] for w in words]
            y2s = [w[1] + w[3] for w in words]
            x1 = max(0, min(xs) - 8)
            y1 = max(0, min(ys) - 8)
            x2 = min(vis.shape[1], max(x2s) + 8)
            y2 = min(vis.shape[0], max(y2s) + 8)
            area = max(1, (x2 - x1) * (y2 - y1))
            # Vote-based consensus is safer than "largest box wins".
            # Keep the exact OCR-parsed reference in live mode (no nearest-catalog replacement).
            ref_votes[line_ref] = ref_votes.get(line_ref, 0.0) + 3.0 + min(3.0, area / 5000.0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                line_ref,
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                lbl_scale,
                (0, 0, 0),
                lbl_thick + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                line_ref,
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                lbl_scale,
                (0, 255, 0),
                lbl_thick,
                cv2.LINE_AA,
            )

        # 2) Fallback to word-level boxes.
        for x, y, w, h, txt, cleaned in word_rows:
            # Ignore generic noise words/fragments; draw only plausible reference tokens.
            compact_txt = re.sub(r"[^A-Z0-9]", "", (txt or "").upper())
            plausible = False
            if re.search(r"(SA|SN)[0-9OILBQGZ]{3,8}", compact_txt):
                plausible = True
            elif re.search(r"[0-9]{4,6}", compact_txt):
                plausible = True
            elif cleaned.startswith(("SA", "SN")) and len(cleaned) >= 4:
                plausible = True
            if not plausible:
                continue
            word_ref = ""
            for m in re.finditer(r"(SA|SN)([0-9OILBQGZ]{4,8})", compact_txt):
                cand = correct_reference_parts(
                    (m.group(1) or "").upper(),
                    (m.group(2) or ""),
                    allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
                )
                cand = normalize_reference(cand)
                if cand and is_valid_hifi_reference(cand):
                    word_ref = cand
                    break
            is_ref = bool(word_ref)
            is_known_ref = bool(is_ref and word_ref in KNOWN_REFERENCE_SET)
            if is_ref:
                area = int(max(1, w) * max(1, h))
                ref_votes[word_ref] = ref_votes.get(word_ref, 0.0) + 1.0 + min(1.5, area / 2500.0)
            color = (0, 255, 0) if is_ref else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            if is_ref:
                label = word_ref
            else:
                parsed = _best_known_ref_from_text_confusion(compact_txt, min_score=90)
                label = parsed if parsed else txt
            if label:
                cv2.putText(
                    vis,
                    label,
                    (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (lbl_scale if is_ref else max(0.5, lbl_scale - 0.08)),
                    (0, 0, 0),
                    (lbl_thick + 2) if is_ref else (lbl_thick + 1),
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    label,
                    (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (lbl_scale if is_ref else max(0.5, lbl_scale - 0.08)),
                    color,
                    lbl_thick if is_ref else 1,
                    cv2.LINE_AA,
                )
        if ref_votes:
            ranked_refs = sorted(ref_votes.items(), key=lambda kv: kv[1], reverse=True)
            top_ref, top_score = ranked_refs[0]
            second_score = ranked_refs[1][1] if len(ranked_refs) > 1 else 0.0
            # Avoid unstable single-frame flips between nearby known refs.
            if top_score >= 2.0 and (top_score - second_score) >= 1.0:
                best_known_ref = top_ref
        h_img, _ = vis.shape[:2]
        cv2.putText(
            vis,
            f"OCR FPS: {ocr_fps:.1f}" if ocr_fps > 0 else "OCR FPS: --",
            (10, max(20, h_img - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.65, lbl_scale),
            (0, 0, 0),
            lbl_thick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"OCR FPS: {ocr_fps:.1f}" if ocr_fps > 0 else "OCR FPS: --",
            (10, max(20, h_img - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.65, lbl_scale),
            (0, 255, 0),
            lbl_thick,
            cv2.LINE_AA,
        )
        # Real-time presence checks requested by user.
        raw_all = " ".join(raw_tokens)
        has_eac = self._token_exists_in_text(raw_all, "EAC", min_ratio=66)
        has_hifi = self._token_exists_in_text(raw_all, "HIFI", min_ratio=68)
        has_phrase = self._phrase_exists_in_text(raw_all)
        checks = [
            ("EAC", has_eac),
            ("HIFI", has_hifi),
            ("PHRASE", has_phrase),
            ("REF", bool(best_known_ref)),
        ]
        self.live_last_presence = {
            "eac": bool(has_eac),
            "hifi": bool(has_hifi),
            "phrase": bool(has_phrase),
            "reference": bool(best_known_ref),
            "reference_value": normalize_reference(best_known_ref or ""),
        }
        base_y = 48
        for i, (nm, ok) in enumerate(checks):
            c = (0, 220, 0) if ok else (0, 0, 255)
            txt = f"{nm}:{'YES' if ok else 'NO'}"
            y = base_y + (i * 22)
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.52, lbl_scale - 0.06), (0, 0, 0), lbl_thick + 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.52, lbl_scale - 0.06), c, 1, cv2.LINE_AA)
        return vis, best_known_ref

    def _draw_fps_overlay_only(self, frame_bgr, ocr_fps):
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        vis = frame_bgr.copy()
        h_img, _ = vis.shape[:2]
        cv2.putText(
            vis,
            f"OCR FPS: {ocr_fps:.1f}" if ocr_fps > 0 else "OCR FPS: --",
            (10, max(20, h_img - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _live_camera_tick(self):
        if not self.live_camera_mode:
            return
        if self.live_camera_cap is None:
            self._stop_live_camera()
            return
        # Window may have been closed while live mode was active.
        if not getattr(self, "filter_preview_label", None):
            self._stop_live_camera()
            return
        try:
            if not self.filter_preview_label.winfo_exists():
                self._stop_live_camera()
                return
        except Exception:
            self._stop_live_camera()
            return
        ok, frame = self.live_camera_cap.read()
        if ok and frame is not None and frame.size > 0:
            os.makedirs(self.camera_capture_dir, exist_ok=True)
            now = time.monotonic()
            preview = downscale_for_ocr(
                frame,
                max_side=int(getattr(self, "live_preview_max_side", 860) or 860),
            )
            out_path = os.path.join(self.camera_capture_dir, "live_current.jpg")
            ocr_frozen = bool(getattr(self, "_live_overlay_freeze_after_mismatch", False))
            if not ocr_frozen:
                if now - float(getattr(self, "live_frame_last_save_ts", 0.0) or 0.0) >= float(
                    getattr(self, "live_frame_save_interval_sec", 0.45) or 0.45
                ):
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                    self.live_frame_last_save_ts = now
            if os.path.isfile(out_path) and (not ocr_frozen):
                self.camera_filter_path = out_path
                t0 = time.monotonic()
                overlay_interval = float(getattr(self, "live_overlay_interval_sec", 0.35) or 0.35)
                if (
                    now - float(getattr(self, "live_overlay_last_ts", 0.0) or 0.0) >= overlay_interval
                    or self.live_overlay_cache is None
                ):
                    vis, overlay_ref = self._draw_ocr_overlay(preview, getattr(self, "live_ocr_fps", 0.0))
                    self.live_overlay_cache = vis
                    self.live_overlay_last_ts = now
                else:
                    overlay_ref = ""
                    vis = self._draw_fps_overlay_only(preview, getattr(self, "live_ocr_fps", 0.0))
                dt = max(1e-3, time.monotonic() - t0)
                inst_fps = 1.0 / dt
                self.live_ocr_fps = inst_fps if getattr(self, "live_ocr_fps", 0.0) <= 0 else (0.8 * self.live_ocr_fps + 0.2 * inst_fps)
                try:
                    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    lw = max(320, int(self.filter_preview_label.winfo_width() or 540))
                    lh = max(220, int(self.filter_preview_label.winfo_height() or 360))
                    pil_img.thumbnail((lw, lh), Image.Resampling.BILINEAR)
                    self.filter_photo = ImageTk.PhotoImage(pil_img)
                    self.filter_preview_label.config(image=self.filter_photo, text="")
                except Exception:
                    self.show_image(out_path, self.filter_preview_label, "filter")
                # Green rectangle means known SA/SN detected -> save capture and match immediately.
                if overlay_ref:
                    if overlay_ref == self.live_ref_streak_ref:
                        self.live_ref_streak_count += 1
                    else:
                        self.live_ref_streak_ref = overlay_ref
                        self.live_ref_streak_count = 1
                else:
                    self.live_ref_streak_ref = ""
                    self.live_ref_streak_count = 0
                if (
                    overlay_ref
                    and self.live_ref_streak_count >= int(max(1, getattr(self, "live_overlay_confirm_needed", 1) or 1))
                    and (not self.is_busy)
                    and (not getattr(self, "_live_hold_mismatch_until_camera_restart", False))
                ):
                    snap_name = "live_detected_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    snap_path = os.path.join(self.camera_capture_dir, snap_name)
                    cv2.imwrite(snap_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    self.camera_filter_path = snap_path
                    # Immediate confirmation path for green-box known references.
                    self.live_ref_streak_ref = overlay_ref
                    self.live_ref_streak_count = int(getattr(self, "live_streak_required", 6) or 6)
                    self.live_forced_ref = overlay_ref
                    self.live_forced_ref_ts = now
                    self.live_last_match_ts = now
                    selected_ref = normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or "")
                    if selected_ref:
                        zero_nine_now = self._is_zero_nine_confusion_pair(overlay_ref, selected_ref)
                        exact_now = bool((overlay_ref == selected_ref) or zero_nine_now)
                        compare_ref_now = selected_ref if zero_nine_now else overlay_ref
                        fast_score = eticket_compatibility_vs_filter(compare_ref_now, selected_ref)
                        fast_msg = (
                            f"🟢 Reference confirmed: {overlay_ref}\n"
                            f"Selected etiquette ref: {selected_ref}\n"
                            f"Fast compare: {'MATCH' if exact_now else 'MISMATCH'} ({fast_score}%)\n"
                            "Live camera stopped.\n"
                            "Final result is being prepared..."
                        )
                        if zero_nine_now:
                            fast_msg += "\n0/9 ambiguity auto-corrected with selected eticket reference."
                        self.set_status(fast_msg, tone=("match" if exact_now else "mismatch"))
                        _matcher_notify_conveyor_uart_impl("match" if exact_now else "mismatch")
                    else:
                        self.set_status(
                            f"🟢 Reference confirmed: {overlay_ref}\n"
                            "Live camera stopped.\n"
                            "Matching eticket now...",
                            tone="neutral",
                        )
                    # Freeze once green-confirmed so result is stable and readable.
                    self._stop_live_camera()
                    self.find_match(strict_live=True, forced_ref=overlay_ref)
                    return
            interval = float(getattr(self, "live_strict_ocr_interval_sec", 3.5) or 3.5)
            ready_ts = float(getattr(self, "live_strict_ready_ts", 0.0) or 0.0)
            need = int(getattr(self, "live_streak_required", 3) or 3)
            # Run strict OCR only after settle time, then every `interval` seconds; require repeated same ref.
            if (
                (not self.is_busy)
                and now >= ready_ts
                and (now - self.live_last_match_ts >= interval)
                and (not bool(getattr(self, "live_strict_busy", False)))
                and (not getattr(self, "_live_hold_mismatch_until_camera_restart", False))
            ):
                self.live_last_match_ts = now
                cap = self.live_camera_cap
                if cap is not None:
                    for _ in range(4):
                        try:
                            cap.grab()
                        except Exception:
                            break
                    ok2, latest = cap.read()
                    if ok2 and latest is not None and latest.size > 0:
                        frame = latest
                self._start_live_strict_worker(frame.copy(), need)
        next_ms = int(getattr(self, "live_ui_interval_ms", 33) or 33)
        if next_ms < 15:
            next_ms = 15
        self.live_camera_job = self.root.after(next_ms, self._live_camera_tick)

    def _start_live_strict_worker(self, frame_bgr, need):
        if getattr(self, "_live_hold_mismatch_until_camera_restart", False):
            return
        if self.live_strict_busy:
            return
        self.live_strict_busy = True
        self.set_status("Live camera running.\nReading reference (background OCR)...")

        def _worker():
            strict_ref, why = self._live_get_strict_ref(frame_bgr)
            snap_path = ""
            if strict_ref and frame_bgr is not None and frame_bgr.size > 0:
                try:
                    os.makedirs(self.camera_capture_dir, exist_ok=True)
                    snap_path = os.path.join(
                        self.camera_capture_dir,
                        "live_strict_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg",
                    )
                    cv2.imwrite(snap_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                except Exception:
                    snap_path = ""
            try:
                self.root.after(
                    0,
                    lambda sr=strict_ref, w=why, n=int(need), sp=snap_path: self._apply_live_strict_result(sr, w, n, sp),
                )
            except Exception:
                self.live_strict_busy = False
        self.live_strict_worker = threading.Thread(target=_worker, daemon=True)
        self.live_strict_worker.start()

    def _apply_live_strict_result(self, strict_ref, why, need, strict_snap_path=""):
        self.live_strict_busy = False
        if not self.live_camera_mode:
            return
        if getattr(self, "_live_hold_mismatch_until_camera_restart", False):
            return
        now = time.monotonic()
        forced_ref = normalize_reference(getattr(self, "live_forced_ref", "") or "")
        forced_ts = float(getattr(self, "live_forced_ref_ts", 0.0) or 0.0)
        # Ignore delayed strict-worker result if a green-box forced confirm happened recently.
        if (
            forced_ref
            and (now - forced_ts) < 5.0
            and strict_ref
            and normalize_reference(strict_ref) != forced_ref
        ):
            return
        if strict_ref:
            sn = normalize_reference(strict_ref)
            if getattr(self, "_live_overlay_freeze_after_mismatch", False):
                bad = normalize_reference(getattr(self, "_live_last_mismatch_blocked_ref", "") or "")
                last_ts = float(getattr(self, "_live_last_mismatch_mono_ts", 0.0) or 0.0)
                if bad and sn == bad and (time.monotonic() - last_ts) < 90.0:
                    return
            if strict_snap_path and os.path.isfile(strict_snap_path):
                self.camera_filter_path = strict_snap_path
            self.live_ref_streak_ref = strict_ref
            self.live_ref_streak_count = int(max(1, need))
            self.set_status(
                f"🟢 Reference confirmed: {strict_ref}\n"
                "Matching now..."
            )
            self.find_match(strict_live=True, forced_ref=strict_ref)
        else:
            self.live_ref_streak_ref = ""
            self.live_ref_streak_count = 0
            self.set_status(
                "Live camera running.\n"
                f"Waiting for stable readable reference... ({why})"
            )

    def _live_get_strict_ref(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return "", "empty frame"
        quick_ref, quick_why = _live_quick_ref_pipeline(frame_bgr)
        # Quick focused pass for live mode: center band + strict whitelist before heavy OCR.
        try:
            h, w = frame_bgr.shape[:2]
            y0 = max(0, int(0.24 * h))
            y1 = min(h, int(0.72 * h))
            x0 = max(0, int(0.06 * w))
            x1 = min(w, int(0.94 * w))
            roi = frame_bgr[y0:y1, x0:x1]
            if roi is not None and roi.size > 0:
                up = cv2.resize(roi, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
                g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
                _ret, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=SNAX0123456789"
                quick_hits = []
                for src in (clahe, otsu, cv2.bitwise_not(otsu)):
                    try:
                        txt = pytesseract.image_to_string(src, config=cfg) or ""
                    except Exception:
                        txt = ""
                    cand = _live_ref_from_text_fast(txt)
                    if cand:
                        quick_hits.append(cand)
                if quick_hits:
                    votes = {}
                    for c in quick_hits:
                        votes[c] = votes.get(c, 0) + 1
                    quick_best = max(votes.items(), key=lambda kv: kv[1])[0]
                    if is_valid_hifi_reference(quick_best):
                        return quick_best, "quick_live_pass"
        except Exception:
            pass
        raw_blobs = []
        candidates = []
        try:
            ref, raw = ocr_filter_reference(frame_bgr, deadline=time.monotonic() + 10.0)
        except Exception:
            return "", "ocr error"
        raw_blobs.append(raw or "")
        ocr_ref = normalize_reference(ref)
        if ocr_ref:
            candidates.append(ocr_ref)
        for c in _collect_filter_ref_candidates(raw or ""):
            if c not in candidates:
                candidates.append(c)
        # Second pass on center band helps when full-frame OCR is noisy.
        try:
            h, _w = frame_bgr.shape[:2]
            y0 = max(0, int(0.20 * h))
            y1 = min(h, int(0.78 * h))
            if y1 - y0 > 24:
                band = frame_bgr[y0:y1, :]
                ref2, raw2 = ocr_filter_reference(band, deadline=time.monotonic() + 5.0)
                raw_blobs.append(raw2 or "")
                ref2 = normalize_reference(ref2)
                if ref2 and ref2 not in candidates:
                    candidates.append(ref2)
                for c in _collect_filter_ref_candidates(raw2 or ""):
                    if c not in candidates:
                        candidates.append(c)
        except Exception:
            pass
        raw_all = "\n".join(x for x in raw_blobs if x)
        if quick_ref and quick_ref not in candidates:
            candidates.append(normalize_reference(quick_ref))
        # If OCR already contains an explicit strict token (SA##### / SN######),
        # prioritize it before fuzzy catalog voting to avoid wrong nearest-neighbor picks.
        explicit_refs = []
        for src in raw_blobs + [raw_all]:
            if not src:
                continue
            kk = _best_known_ref_from_text_confusion(src, min_score=86)
            if kk:
                explicit_refs.append(kk)
            for m in re.finditer(r"(SA|SN)\s*([0-9OILBQGZ]{5,6})", src, flags=re.IGNORECASE):
                cand = correct_reference_parts(
                    (m.group(1) or "").upper(),
                    (m.group(2) or ""),
                    allowed_prefixes=set(FILTER_PREFIX_LENGTHS.keys()),
                )
                cand = normalize_reference(cand)
                if cand and is_valid_hifi_reference(cand):
                    explicit_refs.append(cand)
        if explicit_refs:
            exp_votes = {}
            for er in explicit_refs:
                exp_votes[er] = exp_votes.get(er, 0) + 1
            exp_best = max(exp_votes.items(), key=lambda kv: kv[1])[0]
            if is_valid_hifi_reference(exp_best):
                return normalize_reference(exp_best), "explicit_strict_token_raw"
        snap_from_trace = trace_snap_to_known_reference(raw_all, min_fuzz=84)
        if snap_from_trace and snap_from_trace not in candidates:
            candidates.append(normalize_reference(snap_from_trace))
        if not candidates:
            return "", "no valid SA/SN"
        # Consolidate against known catalog using OCR digit stream voting.
        scored = []
        for cand in candidates:
            voted = normalize_reference(resolve_filter_ref_with_digit_catalog_vote(raw_all or "", cand))
            if voted and is_valid_hifi_reference(voted):
                scored.append(voted)
            if cand and is_valid_hifi_reference(cand):
                scored.append(cand)
        if not scored:
            return "", "invalid SA/SN format"
        votes = {}
        compact_raw = re.sub(r"[^A-Z0-9]", "", (raw_all or "").upper())
        for s in scored:
            weight = 1
            sc = re.sub(r"[^A-Z0-9]", "", s.upper())
            if sc and sc in compact_raw:
                weight += 2
            if s in KNOWN_REFERENCE_SET:
                weight += 1
            votes[s] = votes.get(s, 0) + weight
        best_ref, best_score = max(votes.items(), key=lambda kv: kv[1])
        if not is_valid_hifi_reference(best_ref):
            return "", "invalid SA/SN format"
        compact_ref = re.sub(r"[^A-Z0-9]", "", best_ref)
        digit_blob = re.sub(r"[^0-9]", "", (raw_all or "").upper())
        _p, d = split_reference(best_ref)
        if compact_ref and (compact_raw.count(compact_ref) < 1):
            # Curved/fragmented text fallback: accept if digits strongly supported.
            if not d or _best_digit_window_fuzz(digit_blob, d) < 92:
                return "", "weak text evidence"
        if best_score < 3 and compact_ref and compact_ref not in compact_raw:
            return "", "low vote confidence"
        # Slow/strict profile: disagreement blocks only when heavy evidence is weak.
        if quick_ref and normalize_reference(quick_ref) != normalize_reference(best_ref):
            has_explicit = bool(compact_ref and compact_ref in compact_raw)
            digit_strong = bool(d and _best_digit_window_fuzz(digit_blob, d) >= 95)
            if not (best_score >= 5 or has_explicit or digit_strong):
                return "", f"candidate disagreement ({quick_why})"
        return best_ref, "ok"

    def show_image(self, path, target_label, which):
        try:
            if target_label is None:
                return
            try:
                if not target_label.winfo_exists():
                    return
            except Exception:
                return
            if not path:
                raise ValueError("No file path provided")
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            pil_img = Image.open(path).convert("RGB")
            pil_img.thumbnail((520, 360))
            tk_img = ImageTk.PhotoImage(pil_img)
            target_label.config(image=tk_img, text="")
            if which == "filter":
                self.filter_photo = tk_img
            elif which == "eticket":
                self.eticket_photo = tk_img
            elif which == "client_filter":
                self.client_filter_photo = tk_img
            elif which == "client_eticket":
                self.client_eticket_photo = tk_img
        except Exception as exc:  # noqa: BLE001
            target_label.config(
                text=f"Could not display image.\nPath:\n{path}\n{type(exc).__name__}: {exc}",
                image="",
            )

    def find_match(self, strict_live=False, forced_ref=""):
        if self.is_busy:
            return
        self._ensure_selected_eticket_reference()
        if not self.eticket_index:
            messagebox.showwarning("Missing Data", "Select an Etickets folder and index images first.")
            return
        active_client_row = self._get_active_client_row()

        selected_filter_path = ""
        if self.camera_filter_path and os.path.isfile(self.camera_filter_path):
            selected_filter_path = self.camera_filter_path
        else:
            if not self.filter_images:
                messagebox.showwarning(
                    "Missing Data",
                    "Select a Filters folder with images, or use Capture Camera.",
                )
                return
            filter_idx = self.filter_selector.current()
            if filter_idx < 0:
                messagebox.showwarning(
                    "No Filter Selected",
                    "Choose a filter image from the dropdown, or use Capture Camera.",
                )
                return
            selected_filter_path = self.filter_images[filter_idx]
        forced_ref_n = normalize_reference(forced_ref or "")
        manual_ref = forced_ref_n if forced_ref_n else ("" if strict_live else normalize_reference(self.manual_ref_var.get()))
        # Live path calls _stop_live_camera before find_match, so use strict_live as well as current mode.
        self._was_live_when_match_started = bool(self.live_camera_mode or strict_live)
        if strict_live and forced_ref_n:
            self.set_busy(True, "Matching in progress. Please wait...", global_status="Using live confirmed reference...")
        else:
            self.set_busy(True, "Matching in progress. Please wait...", global_status="Reading filter reference...")

        def visual_fallback(selected_filter, raw_hint=""):
            trace_snap = trace_snap_to_known_reference(raw_hint or "", min_fuzz=78)

            def quick_visual_score(path_a, path_b):
                ia = load_image_bgr(path_a)
                ib = load_image_bgr(path_b)
                if ia is None or ib is None:
                    return 0
                ia = cv2.resize(ia, (320, 320), interpolation=cv2.INTER_AREA)
                ib = cv2.resize(ib, (320, 320), interpolation=cv2.INTER_AREA)
                # Focus central ring area.
                ia = ia[48:272, 48:272]
                ib = ib[48:272, 48:272]
                ga = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY)
                gb = cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
                ha = cv2.calcHist([ga], [0], None, [64], [0, 256])
                hb = cv2.calcHist([gb], [0], None, [64], [0, 256])
                cv2.normalize(ha, ha)
                cv2.normalize(hb, hb)
                hist_corr = max(0.0, float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
                ea = cv2.Canny(ga, 80, 180)
                eb = cv2.Canny(gb, 80, 180)
                edge_diff = float(np.mean(cv2.absdiff(ea, eb))) / 255.0
                edge_score = max(0.0, 1.0 - edge_diff)
                return int(min(100, round((0.75 * hist_corr * 100) + (0.25 * edge_score * 100))))

            hint_digits = re.findall(r"[0-9]{4,6}", (raw_hint or ""))
            compact_hint = re.sub(r"[^A-Z0-9]", "", (raw_hint or "").upper())
            # Cheap text-only pre-rank so we do not load hundreds of HEIC/JPEGs.
            cheap_ranked = []
            for item in self.eticket_index:
                p = item.get("path", "")
                if not p or not os.path.isfile(p):
                    continue
                ref = normalize_reference(item.get("reference", ""))
                cheap = 0
                if hint_digits and ref:
                    _pfx, rd = split_reference(ref)
                    for hd in hint_digits:
                        if rd:
                            cheap = max(cheap, fuzz.ratio(hd, rd))
                if ref and compact_hint:
                    cheap = max(cheap, fuzz.ratio(ref, compact_hint[: max(12, len(ref))]))
                if trace_snap and ref == trace_snap:
                    cheap = max(cheap, 99)
                cheap_ranked.append((cheap, item))
            cheap_ranked.sort(key=lambda x: x[0], reverse=True)
            max_loads = 36
            pre = []
            for _c, item in cheap_ranked[:max_loads]:
                p = item.get("path", "")
                q = quick_visual_score(selected_filter, p)
                ref = normalize_reference(item.get("reference", ""))
                if hint_digits and ref:
                    _pfx, rd = split_reference(ref)
                    for hd in hint_digits:
                        if rd and fuzz.ratio(hd, rd) >= 70:
                            q += 8
                            break
                if trace_snap and ref == trace_snap:
                    q += 22
                pre.append((q, item))
            pre.sort(key=lambda x: x[0], reverse=True)
            # Run expensive ORB only on top quick candidates.
            shortlist = pre[: min(6, len(pre))]
            ranked = []
            for q, item in shortlist:
                p = item.get("path", "")
                vs = self._compute_orb_visual_similarity_cached_score(selected_filter, p)
                final_score = int(round((0.65 * vs) + (0.35 * q)))
                ranked.append(
                    {
                        "path": p,
                        "reference": item.get("reference", ""),
                        "score": final_score,
                    }
                )
            ranked.sort(key=lambda x: x["score"], reverse=True)
            if trace_snap and ranked:
                ni = next(
                    (
                        i
                        for i, x in enumerate(ranked)
                        if normalize_reference(x.get("reference", "") or "") == trace_snap
                    ),
                    None,
                )
                if ni is not None and ni > 0:
                    ranked.insert(0, ranked.pop(ni))
            return (ranked[0] if ranked else None), ranked[:3]

        def task():
            deadline = time.monotonic() + 15.0
            used_manual = False
            cache_key = ""
            authoritative_filename = False
            selected_ref_live = normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or "")
            # Fast path for live camera + selected etiquette compare:
            # do not run heavy OCR/search; publish direct match/mismatch immediately.
            if strict_live and forced_ref_n and selected_ref_live:
                eff_forced = forced_ref_n
                img_snap = load_image_bgr(selected_filter_path)
                if img_snap is not None:
                    qref, _ = _live_quick_ref_pipeline(img_snap)
                    qn = normalize_reference(qref or "")
                    if qn and is_valid_hifi_reference(qn):
                        pq, dq = split_reference(qn)
                        pf, fd = split_reference(eff_forced)
                        if qn != eff_forced and pq == pf and dq and fd and len(dq) == len(fd):
                            eff_forced = qn
                zero_nine = self._is_zero_nine_confusion_pair(eff_forced, selected_ref_live)
                compare_ref = selected_ref_live if zero_nine else eff_forced
                fast_score = eticket_compatibility_vs_filter(compare_ref, selected_ref_live)
                client_pipeline, client_pipeline_tone = self._build_client_auto_pipeline(
                    compare_ref,
                    selected_ref_live,
                    active_client_row,
                    live_filter_path=selected_filter_path,
                    live_presence_profile=getattr(self, "live_last_presence", None),
                )
                return {
                    "type": "selected_compare_fast",
                    "selected_filter_path": selected_filter_path,
                    "filter_ref": compare_ref,
                    "raw_detected_ref": forced_ref_n,
                    "selected_ref": selected_ref_live,
                    "selected_exact": bool(compare_ref == selected_ref_live),
                    "selected_compare_score": int(fast_score),
                    "zero_nine_hint": bool(zero_nine),
                    "client_pipeline": client_pipeline,
                    "client_pipeline_tone": client_pipeline_tone,
                    "strict_live": strict_live,
                    "raw_text": (
                        f"Live strict reference: {forced_ref_n}"
                        + (
                            f"\nSnap/quick reconcile: {eff_forced}"
                            if eff_forced != forced_ref_n
                            else ""
                        )
                    ),
                }
            if manual_ref:
                self.root.after(0, lambda: self.set_global_status("Searching etickets..."))
                filter_ref = manual_ref
                raw_text = f"Manual input: {manual_ref}"
                used_manual = True
            else:
                self.root.after(0, lambda: self.set_global_status("Reading filter reference..."))
                image = load_image_bgr(selected_filter_path)
                if image is None:
                    raise ValueError("Image load failed.")
                mtime = int(os.path.getmtime(selected_filter_path))
                cache_key = make_filter_cache_key(selected_filter_path, mtime)
                auth_fn = authoritative_filename_reference(selected_filter_path)

                cached = (not strict_live) and (
                    cache_key in self.filter_cache and self.filter_cache[cache_key].get("reference")
                )
                if cached:
                    cr = self.filter_cache[cache_key].get("reference", "")
                    if auth_fn and normalize_reference(cr) != auth_fn:
                        del self.filter_cache[cache_key]
                        self.save_filter_cache()
                        cached = False

                if auth_fn:
                    authoritative_filename = True
                    filter_ref = auth_fn
                    raw_text = f"Authoritative filename reference: {auth_fn}\n"
                    ocr_dbg, ocr_raw = ocr_filter_reference(image, deadline=deadline)
                    raw_text += "=== OCR trace (audit only; matching uses filename) ===\n" + (ocr_raw or "")
                    if ocr_dbg and normalize_reference(ocr_dbg) != auth_fn:
                        raw_text += (
                            f"\nOCR disagrees ({normalize_reference(ocr_dbg)}); matching uses filename."
                        )
                    if not strict_live:
                        self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                        self.save_filter_cache()
                elif cached:
                    self.root.after(0, lambda: self.set_global_status("Searching etickets..."))
                    filter_ref = self.filter_cache[cache_key].get("reference", "")
                    raw_text = self.filter_cache[cache_key].get("raw_text", "")
                else:
                    filter_ref, raw_text = ocr_filter_reference(image, deadline=deadline)
                    if not strict_live:
                        self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                        self.save_filter_cache()

            if (not strict_live) and (not used_manual) and (not authoritative_filename):
                new_ref, new_raw = reconcile_ocr_ref_with_known_filename(
                    selected_filter_path, filter_ref, raw_text
                )
                if new_ref != filter_ref or new_raw != raw_text:
                    filter_ref, raw_text = new_ref, new_raw
                    if cache_key:
                        self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                        self.save_filter_cache()

            if (not strict_live) and (not used_manual) and (not authoritative_filename):
                trace_snap = trace_snap_to_known_reference(raw_text or "", min_fuzz=80)
                if trace_snap:
                    fn = normalize_reference(filter_ref or "")
                    if not fn or not is_valid_hifi_reference(fn):
                        filter_ref = trace_snap
                        raw_text = (raw_text + "\n" if raw_text else "") + f"Trace catalog snap: {trace_snap}"
                        if cache_key:
                            self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                            self.save_filter_cache()
                    elif is_valid_hifi_reference(fn) and fn not in KNOWN_REFERENCE_SET and trace_snap in KNOWN_REFERENCE_SET:
                        filter_ref = trace_snap
                        raw_text = (raw_text + "\n" if raw_text else "") + f"Trace override (catalog): {trace_snap}"
                        if cache_key:
                            self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                            self.save_filter_cache()
                    elif (
                        is_valid_hifi_reference(fn)
                        and fn in KNOWN_REFERENCE_SET
                        and trace_snap in KNOWN_REFERENCE_SET
                        and fn != trace_snap
                    ):
                        d_blob = re.sub(r"[^0-9]", "", (raw_text or "").upper())

                        def _best_digit_win(blob, kd):
                            L = len(kd)
                            if L < 4 or len(blob) < L:
                                return 0
                            return max(fuzz.ratio(blob[i : i + L], kd) for i in range(len(blob) - L + 1))

                        _pfx_f, fd = split_reference(fn)
                        _pfx_t, td = split_reference(trace_snap)
                        if len(d_blob) >= 5 and _best_digit_win(d_blob, td) > _best_digit_win(d_blob, fd) + 6:
                            filter_ref = trace_snap
                            raw_text = (
                                (raw_text + "\n" if raw_text else "")
                                + f"Trace disambiguation: {trace_snap} (digit evidence vs {fn})"
                            )
                            if cache_key:
                                self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                                self.save_filter_cache()

            if (
                (not strict_live)
                and
                filter_ref
                and (not used_manual)
                and (not authoritative_filename)
                and KNOWN_REFERENCE_SET
                and normalize_reference(filter_ref) not in KNOWN_REFERENCE_SET
            ):
                best_v_ref, _top3_v_ref = visual_fallback(selected_filter_path, raw_hint=raw_text)
                vis_ref = normalize_reference(best_v_ref.get("reference", "")) if best_v_ref else ""
                vis_sc = int(best_v_ref.get("score", 0)) if best_v_ref else 0
                if vis_ref in KNOWN_REFERENCE_SET and vis_sc >= 68:
                    prev = normalize_reference(filter_ref)
                    filter_ref = vis_ref
                    raw_text = (
                        (raw_text + "\n" if raw_text else "")
                        + f"Visual reference override: {prev} -> {vis_ref} ({vis_sc}% visual)"
                    )
                    if cache_key:
                        self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                        self.save_filter_cache()

            if strict_live and (not used_manual) and (not authoritative_filename):
                filter_ref = normalize_reference(filter_ref)
                # In live mode, keep any detected SA/SN and report mismatch if needed.
                # Only fail when OCR extracted nothing usable.
                if not filter_ref:
                    return {
                        "type": "ocr_fail",
                        "raw_text": (raw_text or "") + "\nStrict live mode: no reference extracted.",
                        "selected_filter_path": selected_filter_path,
                        "strict_live": strict_live,
                    }

            if not filter_ref:
                inferred = infer_reference_from_filename(selected_filter_path)
                if inferred:
                    filter_ref = inferred
                    raw_text = (raw_text + "\n" if raw_text else "") + f"Filename fallback: {inferred}"
                    if cache_key:
                        self.filter_cache[cache_key] = {"reference": filter_ref, "raw_text": raw_text}
                        self.save_filter_cache()
                else:
                    best_v, top3_v = visual_fallback(selected_filter_path, raw_hint=raw_text)
                    if best_v is not None:
                        return {
                            "type": "visual_fallback",
                            "selected_filter_path": selected_filter_path,
                            "raw_text": raw_text,
                            "best": best_v,
                            "top3": top3_v,
                            "strict_live": strict_live,
                        }
                    return {
                        "type": "ocr_fail",
                        "raw_text": raw_text,
                        "selected_filter_path": selected_filter_path,
                        "strict_live": strict_live,
                    }

            self.root.after(0, lambda: self.set_global_status("Searching etickets..."))
            # Strict live path must stay responsive; skip deep image recheck loop.
            if strict_live:
                best, top3, match_type = self.get_matches(filter_ref)
            else:
                best, top3, match_type = self.get_matches_with_recheck(filter_ref)
            if not used_manual and raw_text and "Filename fallback:" in raw_text:
                match_type = "Filename fallback"
            strong_filename_hint = bool(
                not used_manual
                and raw_text
                and (
                    "Strong filename hint:" in raw_text
                    or "Authoritative filename reference:" in raw_text
                )
            )
            return {
                "type": "ok",
                "selected_filter_path": selected_filter_path,
                "filter_ref": filter_ref,
                "used_manual": used_manual,
                "best": best,
                "top3": top3,
                "match_type": match_type,
                "strong_filename_hint": strong_filename_hint,
                "authoritative_filename": authoritative_filename,
                "raw_text": raw_text,
                "strict_live": strict_live,
            }

        def on_success(result):
            self.set_busy(False)
            self.set_global_status("Done âœ…")

            def _status_with_compare(msg, tone="neutral"):
                fr = result.get("filter_ref") or ""
                full = msg + self._matcher_eticket_compare_footer(fr)
                self.set_status(full, tone=tone)
                _matcher_notify_conveyor_uart_impl(tone)
                was_live = getattr(self, "_was_live_when_match_started", False)
                if not (was_live and tone == "mismatch"):
                    self._schedule_live_camera_auto_restart(full, tone)
                if was_live:
                    if tone == "mismatch":
                        self._live_overlay_freeze_after_mismatch = True
                        self._live_hold_mismatch_until_camera_restart = True
                        cand_bad = (
                            result.get("raw_detected_ref")
                            or result.get("filter_ref")
                            or ""
                        )
                        self._live_last_mismatch_blocked_ref = normalize_reference(cand_bad)
                        self._live_last_mismatch_mono_ts = time.monotonic()
                    else:
                        self._live_overlay_freeze_after_mismatch = False
                        self._live_hold_mismatch_until_camera_restart = False
                        self.live_overlay_cache = None
                        self._live_last_mismatch_blocked_ref = ""
                        self._live_last_mismatch_mono_ts = 0.0

            if result.get("type") == "selected_compare_fast":
                self.show_image(result["selected_filter_path"], self.filter_preview_label, "filter")
                if self._is_eticket_preview_locked():
                    self._redraw_matcher_selected_eticket_overlay()
                msg = (
                    f"Filter reference (live): {result['filter_ref']}\n"
                    f"Selected etiquette reference: {result['selected_ref']}\n"
                    f"Result: {'✅ MATCHES SELECTED ETICKET' if result['selected_exact'] else '❌ DOES NOT MATCH SELECTED ETICKET'}\n"
                    f"Selected compare score: {result['selected_compare_score']}%\n"
                )
                if result.get("zero_nine_hint"):
                    msg += (
                        f"Detected OCR reference: {result.get('raw_detected_ref', result['filter_ref'])}\n"
                        "0/9 ambiguity auto-corrected with selected eticket reference.\n"
                    )
                cp = (result.get("client_pipeline") or "").strip()
                if cp:
                    msg += "\n" + cp + "\n"
                tone = "match" if result["selected_exact"] else "mismatch"
                cpt = result.get("client_pipeline_tone")
                if cpt == "mismatch":
                    tone = "mismatch"
                elif cpt == "match" and tone != "mismatch":
                    tone = "match"
                _status_with_compare(msg, tone=tone)
                live_prof = getattr(self, "live_last_presence", {}) or {}
                eac_ok = bool(live_prof.get("eac", True))
                hifi_ok = bool(live_prof.get("hifi", True))
                reasons = []
                if not bool(result.get("selected_exact")):
                    reasons.append("Reference mismatch")
                if not eac_ok:
                    reasons.append("Live filter: EAC missing")
                if not hifi_ok:
                    reasons.append("Live filter: HIFI missing")
                for note in self._presence_miss_notes_from_pipeline(result.get("client_pipeline", "")):
                    if note not in reasons:
                        reasons.append(note)
                self._record_dashboard_exception(
                    station=(active_client_row[1] if active_client_row else "HIFI-MATCHER"),
                    target_ref=result.get("selected_ref", ""),
                    detected_ref=result.get("filter_ref", ""),
                    eac_ok=eac_ok,
                    hifi_ok=hifi_ok,
                    reasons=reasons,
                    result_label=("FAIL" if reasons else "PASS"),
                )
                return

            if result.get("strict_live"):
                if result["type"] == "ocr_fail":
                    _status_with_compare(
                        "Live camera: waiting for a clear SA/SN on the filter.\n"
                        "No match is shown until the reference is confirmed."
                    , tone="mismatch")
                    return
                if result["type"] == "visual_fallback":
                    _status_with_compare(
                        "Live camera: reference not confirmed — hold steady; no guess match is shown."
                    , tone="mismatch")
                    return
                if result["type"] == "ocr_timeout":
                    _status_with_compare(
                        "Live camera: OCR is slow — keep the filter steady; no result until reference is read."
                    , tone="mismatch")
                    return
            if result["type"] == "visual_fallback":
                self.show_image(result["selected_filter_path"], self.filter_preview_label, "filter")
                best_v = result["best"]
                resolved_path, cached_raw, resolve_note = self._resolve_matcher_eticket_path(best_v)
                if self._is_eticket_preview_locked():
                    self._redraw_matcher_selected_eticket_overlay()
                elif resolved_path:
                    self.show_image(resolved_path, self.eticket_preview_label, "eticket")
                else:
                    self.eticket_preview_label.config(image="", text="Matched eticket file not found")
                top3_lines = []
                for item in result["top3"]:
                    rp3, raw3, _n3 = self._resolve_matcher_eticket_path(item)
                    top3_lines.append(
                        f"- {os.path.basename(item['path'])}: {item.get('reference','NO_REF') or 'NO_REF'} ({item['score']}% visual)\n"
                        f"  Debug path (cache): {raw3 or '(none)'}\n"
                        f"  Debug path (resolved): {rp3 or '(none)'}"
                    )
                _status_with_compare(
                    "OCR could not read a valid SA/SN reference.\n"
                    "Used visual fallback (ORB) to suggest closest eticket.\n"
                    f"Best visual candidate: {os.path.basename(best_v['path'])} ({best_v['score']}%)\n"
                    f"Debug â€” cached path: {cached_raw or '(none)'}\n"
                    f"Debug â€” resolved path: {resolved_path or '(missing)'}"
                    + (f"\nDebug â€” resolve note: {resolve_note}" if resolve_note else "")
                    + "\n\nTop 3 visual candidates:\n"
                    + "\n".join(top3_lines)
                    + "\n\nRaw OCR text:\n"
                    + (result.get("raw_text", "") or "(empty)")
                , tone="mismatch")
                return
            if result["type"] == "ocr_timeout":
                messagebox.showwarning(
                    "OCR timeout",
                    "OCR timeout â€” please enter reference manually",
                )
                self.manual_ref_entry.focus_set()
                _status_with_compare(
                    f"OCR timed out (10s) for filter image: {os.path.basename(result['selected_filter_path'])}\n"
                    f"Raw OCR text:\n{result.get('raw_text', '')}"
                , tone="mismatch")
                return
            if result["type"] == "ocr_fail":
                raw_dbg = result.get("raw_text", "")
                print("[OCR DEBUG] Filter OCR raw output before dialog:\n" + (raw_dbg if raw_dbg else "(empty)"))
                messagebox.showwarning(
                    "OCR Could Not Extract Reference",
                    "Could not extract a valid filter reference.\n"
                    "Raw OCR output was printed in the console/log.\n"
                    "Type it manually and click Find Match again.",
                )
                self.manual_ref_entry.focus_set()
                _status_with_compare(
                    f"OCR failed for filter image: {os.path.basename(result['selected_filter_path'])}\n"
                    f"Raw OCR text:\n{raw_dbg}"
                , tone="mismatch")
                return

            best = result["best"]
            top3 = result.get("top3") or []

            def _top3_debug_lines(items):
                lines = []
                for item in items:
                    shown_ref = item["reference"] if item["reference"] else "NO_REF"
                    rp3, raw3, _n3 = self._resolve_matcher_eticket_path(item)
                    lines.append(
                        f"- {os.path.basename(item['path'])}: {shown_ref} ({item['score']}%)\n"
                        f"  Debug path (cache): {raw3 or '(none)'}\n"
                        f"  Debug path (resolved): {rp3 or '(none)'}"
                    )
                return lines

            if best is None:
                self.show_image(result["selected_filter_path"], self.filter_preview_label, "filter")
                top3_lines = _top3_debug_lines(top3)
                if result.get("strict_live"):
                    if self._is_eticket_preview_locked():
                        self._redraw_matcher_selected_eticket_overlay()
                    else:
                        self.eticket_preview_label.config(
                            image="",
                            text="Match appears here after SA/SN is detected and confirmed.",
                        )
                    _status_with_compare(
                        f"Live: reference {result['filter_ref']} — no eticket match in the index.\n"
                        "No closest-guess preview is shown in live mode."
                    , tone="mismatch")
                    return
                rp_closest, _, _ = self._resolve_matcher_eticket_path(top3[0]) if top3 else (None, "", "")
                if self._is_eticket_preview_locked():
                    self._redraw_matcher_selected_eticket_overlay()
                elif rp_closest:
                    self.show_image(rp_closest, self.eticket_preview_label, "eticket")
                elif top3:
                    self.eticket_preview_label.config(
                        image="",
                        text="Closest eticket preview â€” file not found on disk",
                    )
                else:
                    self.eticket_preview_label.config(image="", text="Matched eticket preview")
                _status_with_compare(
                    f"Filter reference: {result['filter_ref']}\n"
                    "âŒ NO MATCH FOUND (red)\n"
                    "No match at or above 80% confidence. Showing closest candidate preview if available.\n"
                    + ("Top 3 closest matches:\n" + "\n".join(top3_lines) if top3_lines else "")
                , tone="mismatch")
                return

            self.show_image(result["selected_filter_path"], self.filter_preview_label, "filter")

            resolved_path, cached_raw, resolve_note = self._resolve_matcher_eticket_path(best)

            top3_lines = _top3_debug_lines(top3)

            manual_note = " (manual input used)" if result["used_manual"] else ""
            best_ref = best["reference"] if best["reference"] else "NO_REF"
            score = int(best["score"])
            match_type = result["match_type"]
            selected_ref = normalize_reference(getattr(self, "matcher_selected_eticket_ref", "") or "")
            selected_exact = bool(selected_ref and normalize_reference(result["filter_ref"]) == selected_ref)
            selected_compare_score = (
                eticket_compatibility_vs_filter(result["filter_ref"], selected_ref) if selected_ref else 0
            )
            strong_fn = bool(
                result.get("strong_filename_hint") or result.get("authoritative_filename")
            )
            raw_for_conf = result.get("raw_text", "") or ""
            ref_compact = re.sub(r"[^A-Z0-9]", "", normalize_reference(result["filter_ref"]))
            raw_compact = re.sub(r"[^A-Z0-9]", "", raw_for_conf.upper())
            raw_hits = raw_compact.count(ref_compact) if ref_compact else 0
            weak_ocr_exact = bool(
                match_type == "âœ… MATCH" and (not result["used_manual"]) and (not strong_fn) and raw_hits < 2
            )
            # "MATCH" from get_matches only means filter_ref string == indexed eticket ref — not proof the photo contains it.
            if weak_ocr_exact:
                display_match_type = (
                    "UNVERIFIED TEXT MATCH (same ref as an eticket; OCR trace did not repeat the part number)"
                )
            else:
                display_match_type = match_type
            if selected_ref:
                if selected_exact:
                    display_match_type = "✅ MATCHES SELECTED ETICKET"
                else:
                    display_match_type = "❌ DOES NOT MATCH SELECTED ETICKET"
            if strong_fn and match_type == "âœ… MATCH":
                score = min(score, 92)
            if weak_ocr_exact:
                score = min(score, 89)
            diff_count = reference_difference_count(
                result["filter_ref"],
                selected_ref if selected_ref else best_ref,
            )
            severe_warning = diff_count > 2
            if weak_ocr_exact:
                severe_warning = True
            confirmed_match = match_type in ("âœ… MATCH", "âš ï¸ CLOSE MATCH") and (not weak_ocr_exact)
            if selected_ref:
                confirmed_match = selected_exact
            if self._is_eticket_preview_locked():
                self._redraw_matcher_selected_eticket_overlay()
            elif match_type == "âœ… MATCH" and not resolved_path:
                self.eticket_preview_label.config(
                    image="",
                    text=(
                        f"âœ… Reference matched: {normalize_reference(best_ref)} â€” but eticket image file "
                        "not found. Please refresh cache."
                    ),
                )
            elif resolved_path:
                # Show preview for exact/close/fuzzy/no-exact â€” user can compare; status still warns if not confirmed.
                self.show_image(resolved_path, self.eticket_preview_label, "eticket")
            else:
                self.eticket_preview_label.config(
                    image="",
                    text="Matched eticket file not found â€” check path or Refresh Cache",
                )
            if match_type == "âœ… MATCH":
                if result.get("authoritative_filename"):
                    guidance = (
                        "Reference is fixed from the image filename (e.g. 17088.jpg \u2192 SA17088). "
                        "Curved-cap OCR is attached for audit only and does not override the filename. "
                        "If the file was misnamed, rename it or use Manual Filter Reference."
                    )
                elif strong_fn:
                    guidance = (
                        "Eticket matches the reference from the image filename (OCR on the cap was not trusted alone). "
                        "Compare the filter photo to the label. If the file was renamed wrong, this match is wrong."
                    )
                else:
                    if weak_ocr_exact:
                        guidance = (
                            "The app picked a reference that matches an indexed eticket label textually, but the "
                            "OCR trace barely contains that part number (or it came from noise/consensus). "
                            "That is NOT proof the filter photo shows this SA/SN — compare images manually."
                        )
                    else:
                        guidance = (
                            "Indexed eticket reference text matches the extracted filter reference; OCR trace "
                            "repeated the part number enough to treat as reliable."
                        )
            elif match_type == "âš ï¸ CLOSE MATCH":
                guidance = "Reference is similar but not exact â€” please verify manually."
            elif match_type == "âŒ NO EXACT MATCH":
                guidance = (
                    f"No exact match found. Closest match is {best_ref} with only {score}% similarity â€” this may be incorrect."
                )
            else:
                guidance = "No matching eticket found in the folder"
            if selected_ref:
                if selected_exact:
                    guidance = (
                        f"Live/filter reference matches the selected eticket reference ({selected_ref}). "
                        f"Selected compare score: {selected_compare_score}%."
                    )
                else:
                    guidance = (
                        f"Live/filter reference does not match selected eticket reference ({selected_ref}). "
                        f"Selected compare score: {selected_compare_score}%."
                    )
            if (not selected_ref) and match_type == "âœ… MATCH" and not resolved_path:
                guidance = (
                    f"âœ… Reference matched: {normalize_reference(best_ref)} â€” but eticket image file not found. "
                    "Please refresh cache."
                )
            debug_paths = (
                f"\nDebug â€” cached path: {cached_raw or '(none)'}\n"
                f"Debug â€” resolved path: {resolved_path or '(missing)'}"
                + (f"\nDebug â€” resolve note: {resolve_note}" if resolve_note else "")
            )
            status = (
                f"Filter reference: {result['filter_ref']}{manual_note}\n"
                f"Result: {display_match_type}\n"
                f"{'Best matched eticket' if confirmed_match else 'Closest eticket'}: {os.path.basename(best['path'])}\n"
                f"Eticket reference: {best_ref}\n"
                f"Confidence: {score}%\n"
                f"{guidance}\n"
                + debug_paths
                + (
                    f"\nâš ï¸ WARNING: References differ significantly â€” {result['filter_ref']} vs {best_ref} â€” DO NOT confirm this match\n"
                    if severe_warning
                    else ""
                )
                + "\n"
                "Top 3 closest matches:\n" + "\n".join(top3_lines)
            )
            _status_with_compare(status, tone=("match" if confirmed_match else "mismatch"))
            live_prof = getattr(self, "live_last_presence", {}) or {}
            eac_ok = bool(live_prof.get("eac", True))
            hifi_ok = bool(live_prof.get("hifi", True))
            reasons = []
            if not confirmed_match:
                reasons.append("Reference mismatch")
            if not eac_ok:
                reasons.append("Live filter: EAC missing")
            if not hifi_ok:
                reasons.append("Live filter: HIFI missing")
            self._record_dashboard_exception(
                station=(active_client_row[1] if active_client_row else "HIFI-MATCHER"),
                target_ref=(selected_ref if selected_ref else best_ref),
                detected_ref=result.get("filter_ref", ""),
                eac_ok=eac_ok,
                hifi_ok=hifi_ok,
                reasons=reasons,
                result_label=("FAIL" if reasons else "PASS"),
            )

        def on_error(exc):
            self.set_busy(False)
            self.set_global_status("Done âœ…")
            messagebox.showerror("OCR Error", f"Could not process filter image:\n{exc}")

        self.run_in_background(task, on_success, on_error)

    def _deep_find_exact_eticket(self, filter_ref):
        """
        Slow fallback: rescan eticket images to recover missed references,
        then return exact match entry if found.
        Capped so Verify / other callers cannot stall on huge folders.
        """
        target = normalize_reference(filter_ref)
        if not target or not self.eticket_index:
            return None
        for item in self.eticket_index:
            current_ref = normalize_reference(item.get("reference", ""))
            if current_ref == target:
                out = dict(item)
                out["score"] = 100
                return out
        deadline = time.monotonic() + 8.0
        ocr_runs = 0
        max_ocr = 48
        ordered = sorted(
            range(len(self.eticket_index)),
            key=lambda i: eticket_compatibility_vs_filter(
                target,
                self.eticket_index[i].get("reference", ""),
            ),
            reverse=True,
        )
        for idx in ordered:
            if time.monotonic() > deadline or ocr_runs >= max_ocr:
                break
            item = self.eticket_index[idx]
            path = item.get("path")
            if not path or not os.path.isfile(path):
                continue
            image = load_image_bgr(path)
            if image is None:
                continue
            ocr_runs += 1
            new_ref = ""
            try:
                new_ref = extract_reference_maquette_eticket_tesseract(image)
            except Exception:  # noqa: BLE001
                new_ref = ""
            if not new_ref:
                try:
                    new_ref, _raw = ocr_eticket_reference(image)
                except Exception:  # noqa: BLE001
                    new_ref = ""
            if not new_ref:
                new_ref = infer_reference_from_filename(path)
            if not new_ref:
                continue
            self.eticket_index[idx]["reference"] = new_ref
            if normalize_reference(new_ref) == target:
                out = dict(self.eticket_index[idx])
                out["score"] = 100
                return out
        return None

    def get_matches_with_recheck(self, filter_ref):
        best, top3, match_type = self.get_matches(filter_ref)
        if match_type == "âœ… MATCH":
            return best, top3, match_type
        exact = self._deep_find_exact_eticket(filter_ref)
        if exact is not None:
            best2, top32, match_type2 = self.get_matches(filter_ref)
            if match_type2 == "âœ… MATCH":
                return best2, top32, match_type2
            return exact, ([exact] + [x for x in top3 if x.get("path") != exact.get("path")])[:3], "âœ… MATCH"
        return best, top3, match_type

    def _resolve_matcher_eticket_path(self, item):
        """
        Return (existing_absolute_path_or_None, cached_path_string, resolve_note).
        Tries cached path, then etickets_folder + basename, then index entries with same reference.
        """
        if not item:
            return None, "", ""
        raw = (item.get("path") or "").strip()
        ref = normalize_reference(item.get("reference", ""))
        folder = (self.etickets_folder or "").strip()
        folder_abs = os.path.abspath(folder) if folder else ""

        def try_existing(p):
            if not p:
                return None
            ap = os.path.abspath(os.path.normpath(p))
            return ap if os.path.isfile(ap) else None

        if raw:
            hit = try_existing(raw)
            if hit:
                return hit, raw, ""
            if folder_abs:
                hit = try_existing(os.path.join(folder_abs, os.path.basename(raw)))
                if hit:
                    return hit, raw, "resolved_via_etickets_folder_basename"

        if folder_abs and ref:
            want_bn = os.path.basename(raw) if raw else ""
            if want_bn:
                hit = try_existing(os.path.join(folder_abs, want_bn))
                if hit:
                    return hit, raw, "resolved_via_filename_in_etickets_folder"
            for it in self.eticket_index:
                if normalize_reference(it.get("reference", "")) != ref:
                    continue
                p = (it.get("path") or "").strip()
                for cand in (p, os.path.join(folder_abs, os.path.basename(p)) if p else ""):
                    hit = try_existing(cand)
                    if hit:
                        return hit, raw, "resolved_via_index_reference"
        return None, raw, ""

    def get_matches(self, filter_ref):
        fn = normalize_reference(filter_ref)
        candidates = [x for x in self.eticket_index if x["reference"]]
        candidates = [x for x in candidates if is_valid_hifi_reference(x["reference"])]
        if not candidates:
            return None, [], "âŒ NO MATCH FOUND"

        _, f_digits = split_reference(fn)

        exact = [x for x in candidates if normalize_reference(x["reference"]) == fn]
        if exact:
            best = dict(exact[0])
            best["score"] = 100
            ranked = sorted(
                (
                    {
                        **x,
                        "score": eticket_compatibility_vs_filter(fn, x["reference"]),
                    }
                    for x in candidates
                ),
                key=lambda item: item["score"],
                reverse=True,
            )
            return best, ranked[:3], "âœ… MATCH"

        ranked = sorted(
            (
                {
                    **x,
                    "score": eticket_compatibility_vs_filter(fn, x["reference"]),
                }
                for x in candidates
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        top3 = ranked[:3]
        best = top3[0] if top3 else None
        if not best:
            return None, top3, "âŒ NO MATCH FOUND"
        bref = normalize_reference(best.get("reference", ""))
        _, bd = split_reference(bref) if bref else ("", "")
        dr = fuzz.ratio(f_digits, bd) if f_digits and bd else 0
        # "Close" only when same prefix and digit string is truly similar (avoids SA17088 vs SA14074).
        if best["score"] >= 97 and dr >= 93:
            return best, top3, "âš ï¸ CLOSE MATCH"
        if 80 <= best["score"] < 97 or (best["score"] >= 97 and dr < 93):
            return best, top3, "âŒ NO EXACT MATCH"
        return None, top3, "âŒ NO MATCH FOUND"

    def reset(self):
        self._cancel_live_camera_auto_restart()
        self._live_last_mismatch_blocked_ref = ""
        self._live_last_mismatch_mono_ts = 0.0
        self.filters_folder = ""
        self.etickets_folder = ""
        self.filter_images = []
        self.eticket_index = []
        self.presence_profile_cache = {}
        self.step4_serial_cache = {}
        self.visual_score_cache = {}
        self.filter_photo = None
        self.eticket_photo = None
        self.manual_ref_var.set("")
        self.camera_filter_path = ""
        self._live_hold_mismatch_until_camera_restart = False
        self._live_overlay_freeze_after_mismatch = False
        self._stop_live_camera()
        stop_live_ocr_preview()
        self.filter_selector.set("")
        self.filter_selector["values"] = []
        self.filters_label.config(text="No filters folder selected")
        self.etickets_label.config(text="No etickets folder selected")
        self.filter_preview_label.config(image="", text="Filter preview")
        self.eticket_preview_label.config(image="", text="Matched eticket preview")
        self.matcher_selected_eticket_path = ""
        self.matcher_selected_eticket_ref = ""
        self.matcher_selected_eticket_serial_path = ""
        if getattr(self, "matcher_serial_photo_var", None) is not None:
            self.matcher_serial_photo_var.set("")
        if getattr(self, "eticket_selector", None) is not None:
            self.eticket_selector.set("")
            self.eticket_selector["values"] = []
        self.set_status("Reset complete. Select both folders to start again.")
        self.set_global_status(
            "OCR Ready âœ…"
        )


def main():
    root = tk.Tk()
    app = HifiMatcherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


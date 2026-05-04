import importlib.util
import os
import re
import time


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(_SCRIPT_DIR, "rayen mansali bellehy emchy.py")
FILTERS_DIR = r"C:\Users\espace info\Pictures\filtres"
ETICKETS_DIR = r"C:\Users\espace info\Pictures\etiquette"


def load_module(path):
    spec = importlib.util.spec_from_file_location("hifi_matcher_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def list_images(folder):
    out = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}:
            out.append(p)
    return out


def expected_from_filename(mod, path):
    base = os.path.splitext(os.path.basename(path))[0].upper()
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    explicit = re.search(r"\b(SA|SN)\s*([0-9]{5,6})\b", base)
    if explicit:
        return mod.normalize_reference(explicit.group(1) + explicit.group(2))
    compact = re.sub(r"[^A-Z0-9]", "", base)
    m = re.search(r"(SA|SN)([0-9]{5,6})", compact)
    if m:
        return mod.normalize_reference(m.group(1) + m.group(2))
    if re.fullmatch(r"[0-9]{5}", compact):
        return "SA" + compact
    if re.fullmatch(r"[0-9]{6}", compact):
        return "SN" + compact
    return ""


def main():
    t0 = time.time()
    mod = load_module(SCRIPT_PATH)

    filters = list_images(FILTERS_DIR)
    etickets = list_images(ETICKETS_DIR)
    print(f"filters={len(filters)} etickets={len(etickets)}")

    # Build known references from eticket OCR first.
    et_refs = []
    et_by_ref = {}
    for p in etickets:
        img = mod.load_image_bgr(p)
        if img is None:
            continue
        ref, _trace = mod.ocr_eticket_reference_fast(img, timeout_sec=2.0)
        ref = mod.normalize_reference(ref)
        if not ref:
            ref = expected_from_filename(mod, p)
        if ref and mod.is_valid_hifi_reference(ref):
            et_refs.append(ref)
            et_by_ref.setdefault(ref, []).append(p)

    mod.KNOWN_REFERENCE_SET = set(et_refs)
    print(f"known_refs_from_etickets={len(mod.KNOWN_REFERENCE_SET)}")

    strong_total = 0
    strong_ok = 0
    missing_in_eticket = 0
    bad_reads = []
    unresolved = []

    for p in filters:
        img = mod.load_image_bgr(p)
        if img is None:
            unresolved.append((os.path.basename(p), "IMAGE_LOAD_FAIL"))
            continue
        ref, trace = mod.ocr_filter_reference(img, deadline=time.monotonic() + 18.0)
        ref = mod.normalize_reference(ref)
        expected = expected_from_filename(mod, p)
        if expected:
            strong_total += 1
            if ref == expected:
                strong_ok += 1
            else:
                bad_reads.append((os.path.basename(p), expected, ref, (trace or "")[:160].replace("\n", " | ")))
        if ref:
            if ref not in et_by_ref:
                missing_in_eticket += 1
        else:
            unresolved.append((os.path.basename(p), "NO_REF"))

    elapsed = time.time() - t0
    print(f"elapsed_sec={elapsed:.1f}")
    print(f"strong_filename_cases={strong_total} strong_ok={strong_ok}")
    if strong_total:
        print(f"strong_accuracy={(100.0 * strong_ok / strong_total):.1f}%")
    print(f"filter_refs_missing_in_etickets={missing_in_eticket}")
    print(f"unresolved_filters={len(unresolved)}")
    print(f"bad_reads={len(bad_reads)}")

    if bad_reads:
        print("\nBAD_READS_SAMPLE:")
        for row in bad_reads[:20]:
            print(f"- file={row[0]} expected={row[1]} got={row[2]} trace={row[3]}")
    if unresolved:
        print("\nUNRESOLVED_SAMPLE:")
        for row in unresolved[:20]:
            print(f"- file={row[0]} reason={row[1]}")


if __name__ == "__main__":
    main()

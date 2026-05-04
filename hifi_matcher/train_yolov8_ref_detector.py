"""
Train a YOLOv8 detector for SA/SN reference region.

Dataset format (YOLO):
- images/train/*.jpg
- images/val/*.jpg
- labels/train/*.txt
- labels/val/*.txt

One class:
- class 0: reference

Usage:
python train_yolov8_ref_detector.py --data "C:/path/to/ref_dataset/data.yaml"
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--project", default="runs_ref_detector")
    parser.add_argument("--name", default="hifi_ref")
    parser.add_argument("--device", default="0", help="GPU id (e.g. 0) or cpu")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Install ultralytics first: pip install ultralytics\n"
            f"Import error: {exc}"
        )

    if not os.path.isfile(args.data):
        raise SystemExit(f"data.yaml not found: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        project=args.project,
        name=args.name,
        device=args.device,
        amp=True,
        patience=25,
        optimizer="AdamW",
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=2.0,
        translate=0.08,
        scale=0.35,
        perspective=0.0005,
        fliplr=0.0,
        flipud=0.0,
    )

    print("\nTraining complete.")
    print("Use best.pt as HIFI_YOLO_REF_MODEL environment variable.")


if __name__ == "__main__":
    main()

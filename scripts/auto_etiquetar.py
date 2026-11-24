from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml

from src.data_utils import (
    ANNOTATION_DIRS,
    CLASS_NAMES,
    DATA_DIR,
    RAW_SOURCE_DIRS,
    RESULTS_DIR,
    Box,
    boxes_to_yolo_text,
    list_images,
    read_image,
    scene_id_from_filename,
    write_image,
)


WHITE_THRESHOLD = 235
MIN_GRAIN_AREA = 450
PREVIEW_DIR = RESULTS_DIR / "auto_label_preview"
TEMP_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "auto_label_yolo"
TEMP_RESULTS_DIR = RESULTS_DIR / "auto_label_temp"
SEED = 42


def _class_from_hsv(hsv_pixels: np.ndarray) -> int:
    median = np.median(hsv_pixels, axis=0)
    hue, saturation, value = (float(item) for item in median)
    if value < 82 or (value < 108 and saturation < 90):
        return 3  # negro
    if 32 <= hue <= 88 and saturation >= 45:
        return 2  # verde
    if 11 <= hue < 32 and saturation >= 50:
        return 1  # pinton
    if hue < 11 or hue >= 165:
        return 0  # maduro
    # Tonos intermedios o de baja saturación no cubiertos por las reglas dadas.
    # Se asignan a pinton para no descartar detecciones; deben revisarse en preview.
    return 1


def _split_touching_grains(image: np.ndarray, binary_mask: np.ndarray) -> list[np.ndarray]:
    """Use distance-transform markers to separate touching foreground components."""
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    if distance.max() <= 0:
        return []
    _, sure_foreground = cv2.threshold(distance, max(2.0, 0.38 * float(distance.max())), 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)
    marker_count, markers = cv2.connectedComponents(sure_foreground)
    if marker_count <= 1:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [_contour_mask(binary_mask.shape, contour) for contour in contours]

    markers = markers + 1
    unknown = cv2.subtract(binary_mask, sure_foreground)
    markers[unknown > 0] = 0
    cv2.watershed(image.copy(), markers)
    regions: list[np.ndarray] = []
    for marker_id in range(2, int(markers.max()) + 1):
        region = np.zeros(binary_mask.shape, dtype=np.uint8)
        region[(markers == marker_id) & (binary_mask > 0)] = 255
        if cv2.countNonZero(region) >= MIN_GRAIN_AREA:
            regions.append(region)
    return regions


def _contour_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def segment_white_image(image: np.ndarray) -> tuple[list[Box], list[int]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_mask = np.where(gray < WHITE_THRESHOLD, 255, 0).astype(np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    boxes: list[Box] = []
    classes: list[int] = []
    for grain_mask in _split_touching_grains(image, mask):
        if cv2.countNonZero(grain_mask) < MIN_GRAIN_AREA:
            continue
        interior = cv2.erode(
            grain_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        if cv2.countNonZero(interior) < 10:
            continue
        class_id = _class_from_hsv(hsv[interior > 0])
        ys, xs = np.where(grain_mask > 0)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if (x2 - x1) * (y2 - y1) < MIN_GRAIN_AREA:
            continue
        boxes.append(Box(class_id, ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                         (x2 - x1) / width, (y2 - y1) / height))
        classes.append(class_id)
    return boxes, classes


def _write_annotation(image: np.ndarray, destination: Path, boxes: list[Box]) -> None:
    if not write_image(destination, image):
        raise OSError(f"No se pudo guardar la imagen anotada: {destination}")
    label_path = destination.with_suffix(".txt")
    label_path.write_text(boxes_to_yolo_text(boxes), encoding="utf-8")


def _draw_preview(image: np.ndarray, boxes: list[Box], title: str) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    palette = ((0, 0, 255), (0, 165, 255), (0, 255, 0), (50, 50, 50))
    for box in boxes:
        x1, y1, x2, y2 = box.pixel_bounds(width, height)
        color = palette[box.class_id]
        label = CLASS_NAMES[box.class_id]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, max(2, width // 500))
        cv2.putText(output, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, width / 1600), color, 2, cv2.LINE_AA)
    cv2.putText(output, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    return output


def _safe_prepare_outputs(overwrite: bool) -> None:
    for origin in ("blanco", "rama"):
        target = ANNOTATION_DIRS[origin]
        if target.exists() and any(target.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{target} ya contiene archivos. Revísalos o ejecuta auto-label --overwrite para reemplazarlos."
                )
            for child in target.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
        target.mkdir(parents=True, exist_ok=True)
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def _auto_label_white() -> tuple[Counter[int], dict[str, tuple[Path, np.ndarray, list[Box]]]]:
    image_paths = list_images(RAW_SOURCE_DIRS["blanco"])
    if not image_paths:
        raise FileNotFoundError(f"No hay fotos en {RAW_SOURCE_DIRS['blanco']}")
    counts: Counter[int] = Counter()
    previews: dict[str, tuple[Path, np.ndarray, list[Box]]] = {}
    for index, source in enumerate(image_paths, start=1):
        image = read_image(source, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Aviso: imagen ilegible, se omite: {source.name}")
            continue
        boxes, classes = segment_white_image(image)
        destination = ANNOTATION_DIRS["blanco"] / source.name
        _write_annotation(image, destination, boxes)
        counts.update(classes)
        if len(previews) < 6:
            previews[source.name] = (destination, image, boxes)
        if index % 25 == 0 or index == len(image_paths):
            print(f"Fase A: {index}/{len(image_paths)} fotos blancas | granos={sum(counts.values())}")
    return counts, previews


def _prepare_temp_yolo_dataset() -> Path:
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
    for split in ("train", "val"):
        (TEMP_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TEMP_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path, str]] = []
    for image_path in list_images(ANNOTATION_DIRS["blanco"]):
        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            pairs.append((image_path, label_path, scene_id_from_filename(image_path.name)))
    if len(pairs) < 5:
        raise RuntimeError("Se necesitan al menos 5 fotos auto-etiquetadas para entrenar el detector temporal.")

    unique_scenes = sorted({scene for _, _, scene in pairs})
    rng = random.Random(SEED)
    rng.shuffle(unique_scenes)
    val_scenes = set(unique_scenes[:max(1, round(len(unique_scenes) * 0.2))])
    for image_path, label_path, scene in pairs:
        split = "val" if scene in val_scenes else "train"
        shutil.copy2(image_path, TEMP_DATA_DIR / "images" / split / image_path.name)
        shutil.copy2(label_path, TEMP_DATA_DIR / "labels" / split / label_path.name)

    config = {
        "path": str(TEMP_DATA_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    data_yaml = TEMP_DATA_DIR / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Detector temporal: {len(pairs)} fotos, {len(unique_scenes) - len(val_scenes)} escenas train, "
          f"{len(val_scenes)} val; YAML={data_yaml}")
    return data_yaml


def _train_temporary_detector(data_yaml: Path, weights: str, batch_size: int, workers: int) -> Path:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("auto-label requiere CUDA; PyTorch no detecta GPU en este WSL.")
    print(f"Entrenamiento temporal con {torch.cuda.get_device_name(0)}; CUDA={torch.version.cuda}")
    TEMP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    result = model.train(
        data=str(data_yaml), epochs=20, imgsz=640, batch=batch_size, device=0,
        workers=workers, project=str(TEMP_RESULTS_DIR), name="white_seed",
        exist_ok=True, cache=False, amp=True, deterministic=True, patience=5,
        plots=False, save=True, verbose=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO no generó el detector temporal esperado: {best}")
    del model
    torch.cuda.empty_cache()
    return best


def _auto_label_branches(model_path: Path, conf: float, iou: float,
                         previews: dict[str, tuple[Path, np.ndarray, list[Box]]]) -> Counter[int]:
    from ultralytics import YOLO

    image_paths = list_images(RAW_SOURCE_DIRS["rama"])
    if not image_paths:
        raise FileNotFoundError(f"No hay fotos en {RAW_SOURCE_DIRS['rama']}")
    model = YOLO(str(model_path))
    counts: Counter[int] = Counter()
    for index, source in enumerate(image_paths, start=1):
        image = read_image(source, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Aviso: imagen ilegible, se omite: {source.name}")
            continue
        prediction = model.predict(source=image, conf=conf, iou=iou, imgsz=640, device=0, verbose=False)[0]
        boxes: list[Box] = []
        if prediction.boxes is not None and len(prediction.boxes):
            height, width = image.shape[:2]
            xyxy = prediction.boxes.xyxy.detach().cpu().numpy()
            class_ids = prediction.boxes.cls.detach().cpu().numpy().astype(int)
            scores = prediction.boxes.conf.detach().cpu().numpy()
            for coords, class_id, score in zip(xyxy, class_ids, scores):
                if not 0 <= class_id < len(CLASS_NAMES) or not math.isfinite(float(score)):
                    continue
                x1, y1, x2, y2 = (int(round(value)) for value in coords)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(Box(class_id, ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
                                 (x2 - x1) / width, (y2 - y1) / height))
        destination = ANNOTATION_DIRS["rama"] / source.name
        _write_annotation(image, destination, boxes)
        counts.update(box.class_id for box in boxes)
        if len(previews) < 12:
            previews[source.name] = (destination, image, boxes)
        if index % 25 == 0 or index == len(image_paths):
            print(f"Fase B: {index}/{len(image_paths)} fotos de rama | detecciones={sum(counts.values())}")
    del model
    return counts


def _save_previews(previews: dict[str, tuple[Path, np.ndarray, list[Box]]]) -> None:
    selected = list(previews.items())[:12]
    if len(selected) < 12:
        raise RuntimeError(f"Solo se pudieron preparar {len(selected)} previews; se esperaban 12.")
    for index, (name, (_path, image, boxes)) in enumerate(selected):
        origin = "blanco" if index < 6 else "rama"
        preview = _draw_preview(image, boxes, f"{origin}: {name}")
        if not write_image(PREVIEW_DIR / f"{origin}_{index % 6 + 1:02d}.jpg", preview):
            raise OSError(f"No se pudo guardar preview para {name}")


def run_auto_label(*, overwrite: bool = False, weights: str = "yolo26n.pt",
                   batch_size: int = 8, workers: int = 4, conf: float = 0.35,
                   iou: float = 0.5) -> None:
    if os.name != "nt":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Ejecuta auto-label desde WSL con CUDA activo; no se usará CPU silenciosamente.")
    for origin, source_dir in RAW_SOURCE_DIRS.items():
        if not source_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta permitida de {origin}: {source_dir}")
    _safe_prepare_outputs(overwrite)
    white_counts, previews = _auto_label_white()
    data_yaml = _prepare_temp_yolo_dataset()
    model_path = _train_temporary_detector(data_yaml, weights, batch_size, workers)
    branch_counts = _auto_label_branches(model_path, conf, iou, previews)
    _save_previews(previews)

    report = {
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "sources": {"blanco": str(RAW_SOURCE_DIRS["blanco"]), "rama": str(RAW_SOURCE_DIRS["rama"])},
        "white_segmented_grains": {CLASS_NAMES[index]: int(white_counts[index]) for index in range(len(CLASS_NAMES))},
        "branch_pseudo_detections": {CLASS_NAMES[index]: int(branch_counts[index]) for index in range(len(CLASS_NAMES))},
        "total_white_grains": int(sum(white_counts.values())),
        "total_branch_detections": int(sum(branch_counts.values())),
        "temporary_detector": str(model_path),
        "confidence": conf,
        "iou": iou,
        "note": "Pseudo-etiquetas automáticas: revisar previews y anotaciones antes de interpretar métricas como exactitud real.",
    }
    report_path = PREVIEW_DIR / "auto_label_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "kind": "automatic_pseudo_labels",
        "source": "scripts/auto_etiquetar.py",
        "class_names": list(CLASS_NAMES),
        "report": report_path.relative_to(DATA_DIR.parent).as_posix(),
        "note": "Pseudo-etiquetas automáticas; no son verdad terreno revisada.",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "annotation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nConteo de granos segmentados en fondo blanco:")
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {index}: {name}: {white_counts[index]}")
    print("Conteo de pseudo-detecciones en ramas:")
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {index}: {name}: {branch_counts[index]}")
    print(f"Previews y reporte: {PREVIEW_DIR}")
    print("ADVERTENCIA: son pseudo-etiquetas automáticas, no verdad terreno revisada.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmenta y pre-etiqueta automáticamente las fotos permitidas.")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza anotaciones automáticas existentes.")
    parser.add_argument("--weights", default="yolo26n.pt", help="Pesos iniciales del detector temporal.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    run_auto_label(overwrite=args.overwrite, weights=args.weights, batch_size=args.batch_size,
                   workers=args.workers, conf=args.conf, iou=args.iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

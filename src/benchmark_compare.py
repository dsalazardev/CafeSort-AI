from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_utils import CLASS_NAMES, DATA_DIR, RESULTS_DIR, read_image

IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.70
YOLO_CONFIDENCE = 0.25
IMAGE_SIZE = 640
MOBILENET_SIZE = 224


def _scores(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for class_name in CLASS_NAMES:
        values = counts[class_name]
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[class_name] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    tp = sum(int(counts[name]["tp"]) for name in CLASS_NAMES)
    fp = sum(int(counts[name]["fp"]) for name in CLASS_NAMES)
    fn = sum(int(counts[name]["fn"]) for name in CLASS_NAMES)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result["micro_average"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    return result


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.maximum(0.0, right_bottom - left_top)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        overlaps = _iou_xyxy(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _weighted_mean(values: list[tuple[float | None, int]]) -> float | None:
    pairs = [(value, weight) for value, weight in values if value is not None and weight > 0]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    return sum(float(value) * weight for value, weight in pairs) / total_weight


def _load_session(path: Path) -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else available[:1]
    return ort.InferenceSession(str(path), providers=providers)


def _input_metadata(session: ort.InferenceSession, default_hw: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    metadata = session.get_inputs()[0]
    shape = metadata.shape
    if (
        len(shape) == 4
        and isinstance(shape[2], int)
        and isinstance(shape[3], int)
        and shape[2] > 0
        and shape[3] > 0
    ):
        return metadata.name, (int(shape[2]), int(shape[3]))
    return metadata.name, default_hw


def _letterbox(image: np.ndarray, new_hw: tuple[int, int]) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    new_height, new_width = new_hw
    ratio = min(new_height / height, new_width / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_x = (new_width - resized_width) / 2.0
    pad_y = (new_height - resized_height) / 2.0
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, pad_x, pad_y


def _yolo_tensor(image: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float, float, float]:
    padded, ratio, pad_x, pad_y = _letterbox(image, input_hw)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
    return tensor, ratio, pad_x, pad_y


def _parse_yolo_output(
    output,
    ratio: float,
    pad_x: float,
    pad_y: float,
    original_hw: tuple[int, int],
    conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(output, (list, tuple)):
        output = output[0]
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Salida YOLO no soportada: forma {array.shape}.")

    num_classes = len(CLASS_NAMES)
    detection_columns = (4 + num_classes, 5 + num_classes)
    if array.shape[0] in detection_columns and array.shape[1] not in detection_columns:
        array = array.T
    elif array.shape[0] == 6 and array.shape[1] not in detection_columns and array.shape[1] != 6:
        array = array.T

    if array.shape[1] == 6 and array.shape[0] not in detection_columns:
        boxes = array[:, :4].astype(np.float32)
        scores = array[:, 4].astype(np.float32)
        classes = array[:, 5].astype(np.int64)
    else:
        if array.shape[1] == 4 + num_classes:
            boxes_xywh = array[:, :4].astype(np.float32)
            class_scores = array[:, 4:4 + num_classes].astype(np.float32)
        elif array.shape[1] == 5 + num_classes:
            boxes_xywh = array[:, :4].astype(np.float32)
            objectness = array[:, 4:5].astype(np.float32)
            class_scores = array[:, 5:5 + num_classes].astype(np.float32) * objectness
        else:
            raise ValueError(f"Salida YOLO no soportada: forma {array.shape}.")
        scores = class_scores.max(axis=1)
        classes = class_scores.argmax(axis=1).astype(np.int64)
        center_x, center_y, width, height = boxes_xywh.T
        boxes = np.stack(
            (center_x - width / 2.0, center_y - height / 2.0, center_x + width / 2.0, center_y + height / 2.0),
            axis=1,
        ).astype(np.float32)

    valid = (
        np.isfinite(scores)
        & (scores >= conf)
        & (classes >= 0)
        & (classes < num_classes)
    )
    boxes = boxes[valid]
    scores = scores[valid]
    classes = classes[valid]
    if boxes.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    image_height, image_width = original_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, max(image_width - 1, 0))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, max(image_height - 1, 0))
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    classes = classes[valid_boxes]
    if boxes.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    kept_indices: list[int] = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(classes == class_id)
        if indices.size:
            keep = _nms(boxes[indices], scores[indices], NMS_IOU_THRESHOLD)
            kept_indices.extend(indices[keep].tolist())
    if not kept_indices:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    selected = np.asarray(kept_indices, dtype=np.int64)
    order = np.argsort(scores[selected])[::-1]
    selected = selected[order]
    return boxes[selected], scores[selected], classes[selected]


def _detection_counts_for_origin(
    records: list[dict],
    session: ort.InferenceSession,
    input_name: str,
    input_hw: tuple[int, int],
    conf: float,
):
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    frame_exact: list[bool] = []
    latencies: list[float] = []

    if records:
        warmup_image = read_image(DATA_DIR.parent / records[0]["yolo_image"], cv2.IMREAD_COLOR)
        if warmup_image is None:
            raise ValueError(f"No se pudo abrir una imagen del split test: {records[0]['yolo_image']}")
        warmup_tensor, _, _, _ = _yolo_tensor(warmup_image, input_hw)
        session.run(None, {input_name: warmup_tensor})

    for record in records:
        image_path = DATA_DIR.parent / record["yolo_image"]
        image = read_image(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo abrir una imagen del split test: {image_path}")
        height, width = image.shape[:2]
        gt_by_class: dict[int, list[list[float]]] = defaultdict(list)
        for box in record["boxes"]:
            cx, cy, box_width, box_height = box["bbox_yolo"]
            gt_by_class[int(box["class_id"])].append(
                [
                    (cx - box_width / 2.0) * width,
                    (cy - box_height / 2.0) * height,
                    (cx + box_width / 2.0) * width,
                    (cy + box_height / 2.0) * height,
                ]
            )

        tensor, ratio, pad_x, pad_y = _yolo_tensor(image, input_hw)
        start = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        latencies.append((time.perf_counter() - start) * 1000.0)
        pred_boxes, pred_conf, pred_cls = _parse_yolo_output(
            outputs,
            ratio,
            pad_x,
            pad_y,
            (height, width),
            conf,
        )
        frame_correct = True

        for class_id, class_name in enumerate(CLASS_NAMES):
            gt = np.asarray(gt_by_class[class_id], dtype=np.float32).reshape((-1, 4))
            pred_indices = np.flatnonzero(pred_cls == class_id)
            pred_indices = pred_indices[np.argsort(pred_conf[pred_indices])[::-1]]
            matched: set[int] = set()
            for pred_index in pred_indices:
                overlaps = _iou_xyxy(pred_boxes[pred_index], gt)
                if overlaps.size:
                    for gt_index in np.argsort(overlaps)[::-1]:
                        if overlaps[gt_index] >= IOU_THRESHOLD and int(gt_index) not in matched:
                            matched.add(int(gt_index))
                            counts[class_name]["tp"] += 1
                            break
                    else:
                        counts[class_name]["fp"] += 1
                        frame_correct = False
                else:
                    counts[class_name]["fp"] += 1
                    frame_correct = False
            missed = len(gt) - len(matched)
            counts[class_name]["fn"] += missed
            if missed:
                frame_correct = False
        frame_exact.append(frame_correct)

    return {
        "images": len(records),
        "object_detection": _scores(counts),
        "exact_frame_accuracy": float(np.mean(frame_exact)) if frame_exact else 0.0,
        "mean_latency_ms_per_image": _safe_mean(latencies),
        "latency_samples": len(latencies),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, indices: np.ndarray) -> dict:
    if not len(indices):
        return {
            "samples": 0,
            "accuracy": None,
            "per_class": {
                name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for name in CLASS_NAMES
            },
        }
    truth = y_true[indices]
    predicted = y_pred[indices]
    report = classification_report(
        truth,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": len(indices),
        "accuracy": float(accuracy_score(truth, predicted)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
    }


def _mobilenet_onnx_predictions(
    session: ort.InferenceSession,
    records: list[dict],
    batch_size: int,
) -> dict:
    input_name, input_hw = _input_metadata(session, (MOBILENET_SIZE, MOBILENET_SIZE))
    if input_hw != (MOBILENET_SIZE, MOBILENET_SIZE):
        raise RuntimeError(
            f"El ONNX de MobileNetV2 espera {input_hw}; se esperaba ({MOBILENET_SIZE}, {MOBILENET_SIZE})."
        )
    metadata = session.get_inputs()[0]
    onnx_batch = metadata.shape[0] if len(metadata.shape) == 4 else batch_size
    if not isinstance(onnx_batch, int) or onnx_batch <= 0:
        onnx_batch = batch_size
    effective_batch_size = max(1, min(batch_size, int(onnx_batch)))

    records_with_crops: list[tuple[dict, list[tuple[Path, int]]]] = []
    for record in records:
        frame_crops = [
            (DATA_DIR.parent / box["mobilenet_crop"], int(box["class_id"]))
            for box in record["boxes"]
        ]
        records_with_crops.append((record, frame_crops))
    if not any(frame_crops for _, frame_crops in records_with_crops):
        raise RuntimeError("No hay cajas de cerezas en test para evaluar MobileNetV2.")

    dummy = np.zeros((1, 3, MOBILENET_SIZE, MOBILENET_SIZE), dtype=np.float32)
    session.run(None, {input_name: dummy})

    y_true_values: list[int] = []
    y_pred_values: list[int] = []
    origin_values: list[str] = []
    frame_latencies: list[tuple[str, float]] = []
    elapsed_ms = 0.0

    for record, frame_crops in records_with_crops:
        frame_elapsed_ms = 0.0
        for start_index in range(0, len(frame_crops), effective_batch_size):
            batch = frame_crops[start_index:start_index + effective_batch_size]
            images: list[np.ndarray] = []
            for crop_path, _ in batch:
                image = read_image(crop_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"No se pudo abrir un recorte MobileNet del split test: {crop_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[:2] != (MOBILENET_SIZE, MOBILENET_SIZE):
                    image = cv2.resize(image, (MOBILENET_SIZE, MOBILENET_SIZE), interpolation=cv2.INTER_AREA)
                # ONNX embeds Keras preprocess_input, so the external input stays RGB [0, 255].
                image = image.astype(np.float32)
                images.append(image.transpose(2, 0, 1))
            inputs = np.ascontiguousarray(np.stack(images).astype(np.float32))
            start = time.perf_counter()
            outputs = session.run(None, {input_name: inputs})[0]
            batch_elapsed_ms = (time.perf_counter() - start) * 1000.0
            if outputs.ndim != 2 or outputs.shape[1] != len(CLASS_NAMES):
                raise RuntimeError(
                    f"Salida ONNX de MobileNetV2 inesperada: {outputs.shape}; "
                    f"se esperaba (N, {len(CLASS_NAMES)})."
                )
            elapsed_ms += batch_elapsed_ms
            frame_elapsed_ms += batch_elapsed_ms
            y_true_values.extend(class_id for _, class_id in batch)
            y_pred_values.extend(np.argmax(outputs, axis=1).tolist())
            origin_values.extend(record["origin"] for _ in batch)
        if frame_crops:
            frame_latencies.append((record["origin"], frame_elapsed_ms))

    y_true = np.asarray(y_true_values, dtype=np.int64)
    y_pred = np.asarray(y_pred_values, dtype=np.int64)
    origin_array = np.asarray(origin_values)
    origins = sorted({record["origin"] for record in records})
    by_origin = {
        origin: {
            **_classification_metrics(y_true, y_pred, np.flatnonzero(origin_array == origin)),
            "mean_latency_ms_per_frame": _safe_mean(
                [latency for frame_origin, latency in frame_latencies if frame_origin == origin]
            ),
            "frames_with_crops": sum(1 for frame_origin, _ in frame_latencies if frame_origin == origin),
        }
        for origin in origins
    }
    return {
        "samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": _classification_metrics(y_true, y_pred, np.arange(len(y_true)))["per_class"],
        "by_origin": by_origin,
        "mean_latency_ms_per_frame": _safe_mean([latency for _, latency in frame_latencies]),
        "mean_latency_ms_per_crop": elapsed_ms / len(y_true),
        "latency_batch_size": effective_batch_size,
        "frames_with_crops": len(frame_latencies),
        "latency_note": (
            "Inferencia ONNX Runtime CPU de los recortes anotados, con batch "
            f"{effective_batch_size}; no incluye lectura de archivo ni detección/localización."
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist(),
    }


def _model_size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _format_number(value: float | int | None) -> str:
    return "N/D" if value is None else f"{float(value):.4f}"


def _write_reports(report: dict, yolo_counts: dict[str, dict[str, int]]) -> tuple[Path, Path]:
    csv_path = RESULTS_DIR / "benchmark_summary.csv"
    rows: list[dict[str, str | int | float | None]] = []
    for origin in ("total", *sorted(report["yolo"]["by_origin"])):
        scores = _scores(yolo_counts) if origin == "total" else report["yolo"]["by_origin"][origin]["object_detection"]
        latency = (
            report["yolo"]["overall_mean_latency_ms_per_image"]
            if origin == "total"
            else report["yolo"]["by_origin"][origin]["mean_latency_ms_per_image"]
        )
        for class_name in (*CLASS_NAMES, "micro_average"):
            metric = scores[class_name]
            rows.append(
                {
                    "model": "YOLO",
                    "origin": origin,
                    "class": class_name,
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "accuracy": report["yolo"]["exact_frame_accuracy"]
                    if origin == "total"
                    else report["yolo"]["by_origin"][origin]["exact_frame_accuracy"],
                    "mean_latency_ms": latency,
                    "onnx_size_mb": report["yolo"]["onnx_size_mb"],
                }
            )

    mobilenet = report["mobilenetv2"]["classification"]
    for origin in ("total", *sorted(mobilenet["by_origin"])):
        metrics = mobilenet if origin == "total" else mobilenet["by_origin"][origin]
        for class_name in CLASS_NAMES:
            metric = metrics["per_class"][class_name]
            rows.append(
                {
                    "model": "MobileNetV2",
                    "origin": origin,
                    "class": class_name,
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "accuracy": metrics["accuracy"],
                    "mean_latency_ms": metrics["mean_latency_ms_per_frame"],
                    "onnx_size_mb": report["mobilenetv2"]["onnx_size_mb"],
                }
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    yolo_overall = report["yolo"]["overall_object_detection"]
    md = [
        "# CafeSort-AI: comparativa de modelos",
        "",
        f"Imágenes de test: {report['test_images']}",
        "",
        "## Resumen de ejecución",
        "",
        "| Modelo | ONNX (MB) | Latencia ONNX Runtime CPU | Exactitud global | F1 micro de detección |",
        "|---|---:|---:|---:|---:|",
        f"| YOLO | {_format_number(report['yolo']['onnx_size_mb'])} | "
        f"{_format_number(report['yolo']['overall_mean_latency_ms_per_image'])} ms/fotograma | "
        f"{_format_number(report['yolo']['exact_frame_accuracy'])} por fotograma | "
        f"{_format_number(yolo_overall['micro_average']['f1'])} |",
        f"| MobileNetV2 | {_format_number(report['mobilenetv2']['onnx_size_mb'])} | "
        f"{_format_number(mobilenet['mean_latency_ms_per_frame'])} ms/fotograma | "
        f"{_format_number(mobilenet['accuracy'])} por recorte | N/A |",
        "",
        "La latencia se mide con `onnxruntime.InferenceSession` en CPU; excluye preprocesamiento y postprocesamiento.",
        f"El batch de MobileNetV2 es {mobilenet['latency_batch_size']}.",
        "",
        "## YOLO: detección por clase",
        "",
        "| Clase | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name in CLASS_NAMES:
        metric = yolo_overall[class_name]
        md.append(
            f"| {class_name} | {_format_number(metric['precision'])} | {_format_number(metric['recall'])} | "
            f"{_format_number(metric['f1'])} | {metric['tp']} | {metric['fp']} | {metric['fn']} |"
        )
    md.extend(
        [
            "",
            "## YOLO: resultado por origen",
            "",
            "| Origen | Imágenes | Exactitud de fotograma | Precision micro | Recall micro | F1 micro | Latencia media (ms/fotograma) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for origin, result in report["yolo"]["by_origin"].items():
        micro = result["object_detection"]["micro_average"]
        md.append(
            f"| {origin} | {result['images']} | {_format_number(result['exact_frame_accuracy'])} | "
            f"{_format_number(micro['precision'])} | {_format_number(micro['recall'])} | "
            f"{_format_number(micro['f1'])} | {_format_number(result['mean_latency_ms_per_image'])} |"
        )
    for origin, result in report["yolo"]["by_origin"].items():
        md.extend(["", f"### YOLO: {origin}", "", "| Clase | Precision | Recall | F1 |", "|---|---:|---:|---:|"])
        for class_name in CLASS_NAMES:
            metric = result["object_detection"][class_name]
            md.append(
                f"| {class_name} | {_format_number(metric['precision'])} | {_format_number(metric['recall'])} | "
                f"{_format_number(metric['f1'])} |"
            )
    md.extend(
        [
            "",
            "## MobileNetV2: clasificación de recortes",
            "",
            f"Exactitud global: **{_format_number(mobilenet['accuracy'])}** sobre {mobilenet['samples']} granos.",
            "",
            "| Clase | Precision | Recall | F1 | Soporte |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name in CLASS_NAMES:
        metric = mobilenet["per_class"][class_name]
        md.append(
            f"| {class_name} | {_format_number(metric['precision'])} | {_format_number(metric['recall'])} | "
            f"{_format_number(metric['f1'])} | {metric['support']} |"
        )
    md.extend(
        [
            "",
            "| Origen | Recortes | Exactitud | Latencia media (ms/fotograma) |",
            "|---|---:|---:|---:|",
        ]
    )
    for origin, result in mobilenet["by_origin"].items():
        md.append(
            f"| {origin} | {result['samples']} | {_format_number(result['accuracy'])} | "
            f"{_format_number(result['mean_latency_ms_per_frame'])} |"
        )
    for origin, result in mobilenet["by_origin"].items():
        md.extend(["", f"### MobileNetV2: {origin}", "", "| Clase | Precision | Recall | F1 | Soporte |", "|---|---:|---:|---:|---:|"])
        for class_name in CLASS_NAMES:
            metric = result["per_class"][class_name]
            md.append(
                f"| {class_name} | {_format_number(metric['precision'])} | {_format_number(metric['recall'])} | "
                f"{_format_number(metric['f1'])} | {metric['support']} |"
            )
    md.extend(
        [
            "",
            "## Alcance de la comparación",
            "",
            report["comparison_note"],
            "",
            "La latencia YOLO se mide por imagen con ONNX Runtime. La latencia MobileNetV2 se mide con ONNX Runtime sobre los recortes anotados; no incluye localizar ni recortar los granos.",
            "",
            "Los resultados dependen de las anotaciones y del split `test` generado en `data/splits_manifest.json`.",
            report.get("annotation_quality_note", ""),
            "",
        ]
    )
    md_path = RESULTS_DIR / "benchmark_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return csv_path, md_path


def benchmark(conf: float = YOLO_CONFIDENCE, image_size: int = IMAGE_SIZE, batch_size: int = 16) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero.")
    manifest_path = DATA_DIR / "splits_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Falta data/splits_manifest.json. Ejecuta `python main.py build-splits` primero.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    auto_label_report_path = RESULTS_DIR / "auto_label_preview" / "auto_label_report.json"
    has_auto_labels = auto_label_report_path.is_file()
    annotation_provenance = manifest.get("annotation_provenance", {})
    provenance_kind = annotation_provenance.get("kind", "unknown")
    if provenance_kind == "provided_segmented_dataset":
        annotation_quality = "provided_segmented_annotations"
        annotation_quality_note = (
            "Las etiquetas proceden de dataset/Segmentado; sus polígonos se convirtieron a cajas YOLO y se mapearon "
            "a las cuatro clases del proyecto. Los splits de este proyecto se reconstruyeron por escena y no se "
            "reutilizaron las particiones fuente. Las métricas reflejan la calidad de esas anotaciones y del mapeo."
        )
    elif provenance_kind == "automatic_pseudo_labels" or (provenance_kind == "unknown" and has_auto_labels):
        annotation_quality = "automatic_pseudo_labels"
        annotation_quality_note = (
            "ADVERTENCIA: el test usa pseudo-etiquetas automáticas generadas por segmentación HSV y YOLO temporal; "
            "estas métricas miden consistencia con esas etiquetas, no exactitud frente a verdad terreno humana. "
            "Verifica results/auto_label_preview antes de presentar resultados como validados."
        )
    else:
        annotation_quality = provenance_kind
        annotation_quality_note = str(annotation_provenance.get("note", "Comprueba la procedencia y revisión de las anotaciones."))
    test_records = [record for record in manifest["records"] if record["split"] == "test"]
    if not test_records:
        raise RuntimeError("El manifiesto no contiene imágenes de test.")

    yolo_onnx = RESULTS_DIR / "yolo" / "yolo_cafe.onnx"
    mobilenet_onnx = RESULTS_DIR / "mobilenet" / "mobilenetv2_cafe.onnx"
    for exported_path in (yolo_onnx, mobilenet_onnx):
        if not exported_path.is_file():
            raise FileNotFoundError(f"Falta el modelo ONNX exportado: {exported_path}")

    yolo_session = _load_session(yolo_onnx)
    mobilenet_session = _load_session(mobilenet_onnx)
    yolo_input_name, yolo_input_hw = _input_metadata(yolo_session, (image_size, image_size))
    if yolo_input_hw != (image_size, image_size):
        print(
            f"El ONNX de YOLO usa entrada {yolo_input_hw[1]}x{yolo_input_hw[0]}; "
            f"se usa esa forma en lugar de --imgsz={image_size}."
        )

    by_origin: dict[str, list[dict]] = defaultdict(list)
    for record in test_records:
        by_origin[record["origin"]].append(record)

    print(f"Benchmark YOLO en ONNX Runtime CPU: {yolo_onnx}")
    yolo_results = {
        origin: _detection_counts_for_origin(
            records,
            yolo_session,
            yolo_input_name,
            yolo_input_hw,
            conf,
        )
        for origin, records in sorted(by_origin.items())
    }

    print(f"Benchmark MobileNetV2 en ONNX Runtime CPU: {mobilenet_onnx}")
    mobilenet_results = _mobilenet_onnx_predictions(mobilenet_session, test_records, batch_size)

    yolo_counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    for result in yolo_results.values():
        for class_name in CLASS_NAMES:
            for key in ("tp", "fp", "fn"):
                yolo_counts[class_name][key] += int(result["object_detection"][class_name][key])

    report = {
        "test_images": len(test_records),
        "annotation_quality": annotation_quality,
        "annotation_quality_note": annotation_quality_note,
        "runtime": {
            "provider": "CPUExecutionProvider",
            "yolo_onnx": yolo_onnx.relative_to(RESULTS_DIR.parent).as_posix(),
            "mobilenet_onnx": mobilenet_onnx.relative_to(RESULTS_DIR.parent).as_posix(),
        },
        "yolo": {
            "onnx_size_bytes": _model_size(yolo_onnx),
            "onnx_size_mb": _model_size(yolo_onnx) / (1024 * 1024) if yolo_onnx.is_file() else None,
            "iou_threshold": IOU_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "confidence_threshold": conf,
            "overall_object_detection": _scores(yolo_counts),
            "exact_frame_accuracy": float(
                np.average(
                    [value["exact_frame_accuracy"] for value in yolo_results.values()],
                    weights=[value["images"] for value in yolo_results.values()],
                )
            ),
            "by_origin": yolo_results,
            "overall_mean_latency_ms_per_image": _weighted_mean(
                [
                    (value["mean_latency_ms_per_image"], value["images"])
                    for value in yolo_results.values()
                ]
            ),
            "latency_note": (
                "Inferencia ONNX Runtime CPU por imagen; el tiempo mide session.run "
                "y no incluye lectura, letterbox ni postprocesamiento."
            ),
        },
        "mobilenetv2": {
            "onnx_size_bytes": _model_size(mobilenet_onnx),
            "onnx_size_mb": _model_size(mobilenet_onnx) / (1024 * 1024) if mobilenet_onnx.is_file() else None,
            "classification": mobilenet_results,
        },
        "comparison_note": (
            "Las métricas de YOLO y MobileNetV2 no son directamente equivalentes: YOLO detecta y clasifica en la imagen completa; "
            "MobileNetV2 clasifica recortes construidos desde las cajas anotadas. Ambos se ejecutan con ONNX Runtime en CPU."
        ),
    }
    output_dir = RESULTS_DIR / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparacion.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path, markdown_path = _write_reports(report, yolo_counts)
    print(f"Comparación JSON: {output_path}")
    print(f"Resumen CSV: {csv_path}")
    print(f"Informe Markdown: {markdown_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara YOLO y MobileNetV2 sobre el mismo split de test con ONNX Runtime.")
    parser.add_argument("--conf", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    benchmark(conf=args.conf, image_size=args.imgsz, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

import cv2
import numpy as np
import yaml
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (  # noqa: E402
    ANNOTATION_DIRS,
    CLASS_NAMES,
    DATA_DIR,
    RAW_SOURCE_DIRS,
    ROOT_DIR,
    Box,
    boxes_to_yolo_text,
    crop_to_square_224,
    list_images,
    parse_yolo_labels,
    read_image,
    scene_id_from_filename,
    write_image,
)

SEED = 42
SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class ImageRecord:
    origin: str
    image_path: Path
    label_path: Path
    source_path: Path
    boxes: list[Box]
    scene_id: str
    split: str = ""


def collect_annotated_images() -> tuple[list[ImageRecord], dict[str, int]]:
    records: list[ImageRecord] = []
    unreviewed: dict[str, int] = {}
    for origin, annotation_dir in ANNOTATION_DIRS.items():
        images = list_images(annotation_dir)
        image_stems = {image.stem.casefold() for image in images}
        labels = list(annotation_dir.glob("*.txt")) if annotation_dir.exists() else []
        orphan_labels = [label.name for label in labels if label.stem.casefold() not in image_stems]
        if orphan_labels:
            raise ValueError(f"Etiquetas sin imagen en {annotation_dir}: {orphan_labels[:5]}")
        unreviewed[origin] = 0
        for image_path in images:
            label_path = annotation_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                unreviewed[origin] += 1
                continue
            source_path = RAW_SOURCE_DIRS[origin] / image_path.name
            if not source_path.is_file():
                raise FileNotFoundError(f"No encuentro la foto cruda original: {source_path}")
            records.append(
                ImageRecord(
                    origin=origin,
                    image_path=image_path,
                    label_path=label_path,
                    source_path=source_path,
                    boxes=parse_yolo_labels(label_path),
                    scene_id=scene_id_from_filename(image_path.name),
                )
            )
    if not records:
        raw_counts = {
            origin: len(list_images(source_dir))
            for origin, source_dir in RAW_SOURCE_DIRS.items()
        }
        raise RuntimeError(
            "Hay imágenes crudas, pero todavía no hay anotaciones YOLO guardadas en "
            f"data/annotations_raw (fondo blanco: {raw_counts['blanco']}, rama: {raw_counts['rama']}). "
            "Los nombres de archivo no contienen las cajas ni las clases de cada cereza. "
            "Primero ejecuta `python main.py label` y guarda las etiquetas manuales; después "
            "`python main.py build-splits` creará el manifiesto, los recortes y data/yolo_dataset/data.yaml."
        )
    return records, unreviewed


def image_strata(records: list[ImageRecord]) -> list[int]:
    presence = Counter(
        class_id
        for record in records
        for class_id in {box.class_id for box in record.boxes}
    )
    strata: list[int] = []
    for record in records:
        present = {box.class_id for box in record.boxes}
        if not present:
            strata.append(len(CLASS_NAMES))
        else:
            strata.append(min(present, key=lambda class_id: (presence[class_id], class_id)))
    return strata


N_SPLITS = 5


def _group_folds(origin_records: list[ImageRecord], origin: str) -> tuple[np.ndarray, str]:
    """Create five group-aware folds, preferring stratification by image stratum."""
    scene_ids = [record.scene_id for record in origin_records]
    unique_scenes = len(set(scene_ids))
    if unique_scenes < N_SPLITS:
        raise ValueError(
            f"{origin}: se requieren al menos {N_SPLITS} escenas únicas para crear train/val/test; "
            f"hay {unique_scenes}."
        )
    strata = image_strata(origin_records)
    try:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(splitter.split(origin_records, strata, scene_ids))
        method = "StratifiedGroupKFold"
    except ValueError as exc:
        print(
            f"{origin}: StratifiedGroupKFold no pudo estratificar ({exc}); "
            "se usa GroupKFold y se optimiza la distribución de clases al elegir los folds."
        )
        splitter = GroupKFold(n_splits=N_SPLITS)
        folds = list(splitter.split(origin_records, strata, scene_ids))
        method = "GroupKFold"

    fold_of = np.full(len(origin_records), -1, dtype=np.int64)
    for fold_index, (_, test_indices) in enumerate(folds):
        if np.any(fold_of[test_indices] != -1):
            raise RuntimeError(f"{origin}: los folds se solapan en algún índice.")
        fold_of[test_indices] = fold_index
    if np.any(fold_of < 0):
        raise RuntimeError(f"{origin}: hay imágenes sin fold asignado.")

    scene_folds: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(origin_records):
        scene_folds[record.scene_id].add(int(fold_of[index]))
    conflicts = {scene: sorted(folds) for scene, folds in scene_folds.items() if len(folds) > 1}
    if conflicts:
        raise RuntimeError(f"{origin}: una escena quedó repartida entre folds: {conflicts}")
    return fold_of, method


def _select_fold_splits(
    fold_of: np.ndarray,
    origin_records: list[ImageRecord],
) -> tuple[dict[int, str], float]:
    """Choose 3 folds for train, 1 for val and 1 for test (60/20/20 of the folds)."""
    total_images = len(origin_records)
    best: tuple[float, float, int, int, dict[int, str]] | None = None
    for val_fold in range(N_SPLITS):
        for test_fold in range(N_SPLITS):
            if val_fold == test_fold:
                continue
            split_of_fold = {
                fold: "train"
                for fold in range(N_SPLITS)
                if fold not in (val_fold, test_fold)
            }
            split_of_fold[val_fold] = "val"
            split_of_fold[test_fold] = "test"

            image_counts: Counter[str] = Counter()
            box_counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_ORDER}
            for index, record in enumerate(origin_records):
                split = split_of_fold[int(fold_of[index])]
                image_counts[split] += 1
                for box in record.boxes:
                    box_counts[split][box.class_id] += 1
            if any(image_counts[split] == 0 for split in SPLIT_ORDER):
                continue

            image_deviation = sum(
                abs(image_counts[split] / total_images - SPLIT_RATIOS[split])
                for split in SPLIT_ORDER
            )
            class_deviation = 0.0
            missing_train = 0
            missing_eval = 0
            for class_id in range(len(CLASS_NAMES)):
                total_boxes = sum(box_counts[split][class_id] for split in SPLIT_ORDER)
                if total_boxes == 0:
                    continue
                class_deviation += sum(
                    abs(box_counts[split][class_id] / total_boxes - SPLIT_RATIOS[split])
                    for split in SPLIT_ORDER
                )
                if box_counts["train"][class_id] == 0:
                    missing_train += 1
                if box_counts["val"][class_id] == 0:
                    missing_eval += 1
                if box_counts["test"][class_id] == 0:
                    missing_eval += 1

            # Prioriza la proporción 60/20/20 y, entre empates, la cobertura de clases.
            # Un train sin clases presentes se penaliza fuerte para no bloquear el entrenamiento.
            score = image_deviation + 10.0 * missing_train
            class_score = class_deviation + 0.25 * missing_eval
            candidate = (score, class_score, val_fold, test_fold, split_of_fold)
            if best is None or candidate[:4] < best[:4]:
                best = candidate

    if best is None:
        raise RuntimeError("No se encontró una combinación de folds que produzca train/val/test no vacíos.")
    return best[4], best[0]


def split_one_origin(origin_records: list[ImageRecord], origin: str) -> None:
    if len(origin_records) < 5:
        raise ValueError(f"{origin}: se requieren al menos 5 fotos etiquetadas para crear train/val/test.")
    fold_of, method = _group_folds(origin_records, origin)
    split_of_fold, score = _select_fold_splits(fold_of, origin_records)

    for index, record in enumerate(origin_records):
        record.split = split_of_fold[int(fold_of[index])]

    scene_splits: dict[str, set[str]] = defaultdict(set)
    for record in origin_records:
        scene_splits[record.scene_id].add(record.split)
    conflicts = {scene: sorted(splits) for scene, splits in scene_splits.items() if len(splits) > 1}
    if conflicts:
        raise RuntimeError(f"{origin}: una escena quedó repartida entre splits: {conflicts}")

    counts = Counter(record.split for record in origin_records)
    print(
        f"{origin}: {len(origin_records)} fotos, {len(scene_splits)} escenas, {method}; "
        f"train={counts['train']} val={counts['val']} test={counts['test']} (score={score:.4f})"
    )


def prepare_records(records: list[ImageRecord]) -> list[ImageRecord]:
    by_origin: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_origin[record.origin].append(record)
    for origin in ANNOTATION_DIRS:
        current = sorted(by_origin.get(origin, []), key=lambda item: item.image_path.name.casefold())
        if not current:
            raise RuntimeError(
                f"No hay anotaciones válidas para el origen '{origin}'. "
                "Etiqueta fotos de ambos orígenes para crear splits independientes 60/20/20."
            )
        split_one_origin(current, origin)
    train_classes = {
        box.class_id
        for record in records
        if record.split == "train"
        for box in record.boxes
    }
    missing_train_classes = [
        CLASS_NAMES[class_id]
        for class_id in range(len(CLASS_NAMES))
        if class_id not in train_classes
    ]
    if missing_train_classes:
        raise ValueError(
            f"Faltan clases en train tras el split: {missing_train_classes}. "
            "Etiqueta más granos de esas clases y vuelve a ejecutar."
        )
    return sorted(records, key=lambda item: (item.split, item.origin, item.image_path.name.casefold()))


def _class_counts(boxes: list[Box]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for box in boxes:
        counts[CLASS_NAMES[box.class_id]] += 1
    return counts


def _ensure_outputs_empty_or_overwrite(overwrite: bool) -> None:
    targets = [DATA_DIR / "yolo_dataset", DATA_DIR / "mobilenet_dataset", DATA_DIR / "splits_manifest.json"]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(DATA_DIR.parent)).replace("\\", "/") for path in existing)
        raise FileExistsError(f"Ya existen salidas de partición: {names}. Usa --overwrite para regenerarlas.")


def build_datasets(records: list[ImageRecord], overwrite: bool = False) -> dict:
    _ensure_outputs_empty_or_overwrite(overwrite)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".build_splits_", dir=DATA_DIR))
    staged_yolo = staging_root / "yolo_dataset"
    staged_mobilenet = staging_root / "mobilenet_dataset"
    for split in SPLIT_ORDER:
        (staged_yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (staged_yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
        for class_name in CLASS_NAMES:
            (staged_mobilenet / split / class_name).mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict] = []
    grain_counts: dict[str, dict[str, Counter]] = {
        origin: {split: Counter() for split in SPLIT_ORDER} for origin in ANNOTATION_DIRS
    }
    image_counts: dict[str, Counter] = {origin: Counter() for origin in ANNOTATION_DIRS}
    try:
        for record in records:
            if record.split not in SPLIT_ORDER:
                raise ValueError(f"Registro sin partición: {record.image_path}")
            image = read_image(record.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo abrir la foto anotada: {record.image_path}")
            filename = f"{record.origin}__{record.image_path.name}"
            yolo_image_path = staged_yolo / "images" / record.split / filename
            yolo_label_path = staged_yolo / "labels" / record.split / f"{Path(filename).stem}.txt"
            shutil.copy2(record.image_path, yolo_image_path)
            yolo_label_path.write_text(boxes_to_yolo_text(record.boxes), encoding="utf-8")

            box_entries: list[dict] = []
            for object_index, box in enumerate(record.boxes, start=1):
                crop_name = f"{record.origin}__{record.image_path.stem}__{object_index:04d}.jpg"
                crop_path = staged_mobilenet / record.split / CLASS_NAMES[box.class_id] / crop_name
                seed_text = f"{record.origin}/{record.image_path.name}/{object_index}"
                crop = crop_to_square_224(image, box, record.origin, zlib.crc32(seed_text.encode("utf-8")))
                if not write_image(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"No se pudo guardar el recorte: {crop_path}")
                box_entries.append(
                    {
                        "object_index": object_index,
                        "class_id": box.class_id,
                        "class_name": CLASS_NAMES[box.class_id],
                        "bbox_yolo": [box.cx, box.cy, box.width, box.height],
                        "mobilenet_crop": (
                            DATA_DIR / "mobilenet_dataset" / record.split / CLASS_NAMES[box.class_id] / crop_name
                        ).relative_to(ROOT_DIR).as_posix(),
                    }
                )
                grain_counts[record.origin][record.split][CLASS_NAMES[box.class_id]] += 1
            image_counts[record.origin][record.split] += 1

            manifest_records.append(
                {
                    "file": record.source_path.relative_to(ROOT_DIR).as_posix(),
                    "origin": record.origin,
                    "scene_id": record.scene_id,
                    "split": record.split,
                    "annotation_file": record.label_path.relative_to(ROOT_DIR).as_posix(),
                    "yolo_image": (
                        DATA_DIR / "yolo_dataset" / "images" / record.split / filename
                    ).relative_to(ROOT_DIR).as_posix(),
                    "yolo_label": (
                        DATA_DIR / "yolo_dataset" / "labels" / record.split / f"{Path(filename).stem}.txt"
                    ).relative_to(ROOT_DIR).as_posix(),
                    "boxes_per_class": _class_counts(record.boxes),
                    "boxes": box_entries,
                }
            )

        data_yaml = {
            "path": str(DATA_DIR / "yolo_dataset"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(CLASS_NAMES),
            "names": list(CLASS_NAMES),
        }
        (staged_yolo / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        provenance_path = DATA_DIR / "annotation_provenance.json"
        annotation_provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.is_file()
            else {"kind": "unknown"}
        )
        manifest = {
            "schema_version": 1,
            "seed": SEED,
            "split_ratios": SPLIT_RATIOS,
            "classes": list(CLASS_NAMES),
            "annotation_provenance": annotation_provenance,
            "records": manifest_records,
        }
        (staging_root / "splits_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        target_yolo = DATA_DIR / "yolo_dataset"
        target_mobilenet = DATA_DIR / "mobilenet_dataset"
        target_manifest = DATA_DIR / "splits_manifest.json"
        if overwrite:
            for target in (target_yolo, target_mobilenet):
                if target.exists():
                    shutil.rmtree(target)
            if target_manifest.exists():
                target_manifest.unlink()
        os.replace(staged_yolo, target_yolo)
        os.replace(staged_mobilenet, target_mobilenet)
        os.replace(staging_root / "splits_manifest.json", target_manifest)
        print_summary(records, image_counts, grain_counts)
        return manifest
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def print_summary(
    records: list[ImageRecord],
    image_counts: dict[str, Counter],
    grain_counts: dict[str, dict[str, Counter]],
) -> None:
    origins = tuple(ANNOTATION_DIRS)
    all_image_counts = Counter()
    all_grain_counts: dict[str, Counter] = {split: Counter() for split in SPLIT_ORDER}
    for origin in origins:
        total = sum(image_counts[origin].values())
        print(f"\n{origin}: {total} fotos etiquetadas")
        print("split  fotos  %origen  " + " ".join(f"{name:>6}" for name in CLASS_NAMES) + "  granos")
        for split in SPLIT_ORDER:
            count = image_counts[origin][split]
            all_image_counts[split] += count
            grain = grain_counts[origin][split]
            for name in CLASS_NAMES:
                all_grain_counts[split][name] += grain[name]
            percent = 100.0 * count / total if total else 0.0
            total_grains = sum(grain[name] for name in CLASS_NAMES)
            class_counts = " ".join(f"{grain[name]:>6}" for name in CLASS_NAMES)
            print(f"{split:<5} {count:>5} {percent:>7.2f}%  {class_counts} {total_grains:>7}")

    total_images = len(records)
    print(f"\ntotal: {total_images} fotos etiquetadas")
    print("split  fotos  %total   " + " ".join(f"{name:>6}" for name in CLASS_NAMES) + "  granos")
    for split in SPLIT_ORDER:
        count = all_image_counts[split]
        percent = 100.0 * count / total_images if total_images else 0.0
        grain = all_grain_counts[split]
        total_grains = sum(grain[name] for name in CLASS_NAMES)
        class_counts = " ".join(f"{grain[name]:>6}" for name in CLASS_NAMES)
        print(f"{split:<5} {count:>5} {percent:>7.2f}%  {class_counts} {total_grains:>7}")


def run_build_splits(overwrite: bool = False) -> dict:
    records, unreviewed = collect_annotated_images()
    if any(unreviewed.values()):
        print(f"Fotos crudas aún sin etiqueta (no se particionan): {unreviewed}")
    records = prepare_records(records)
    return build_datasets(records, overwrite=overwrite)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crea el manifiesto y datasets compartidos desde anotaciones propias.")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza solo las salidas generadas por este script.")
    args = parser.parse_args(argv)
    run_build_splits(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

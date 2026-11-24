from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

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
    parse_yolo_labels,
    read_image,
    scene_id_from_filename,
    write_image,
)


SEGMENTED_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "Segmentado"
CLASS_MAPS = {
    "blanco": {
        "maduros": 0,
        "pintones": 1,
        "secos": 3,
        "sobremaduros": 3,
        "verdes": 2,
    },
    "rama": {
        "overripe": 3,
        "ripe": 0,
        "semi_ripe": 1,
        "unripe": 2,
    },
}
SEGMENTED_DIRS = {"blanco": "Fondo_blanco", "rama": "Cafe_arbol"}


def _polygon_or_box_to_bbox(class_id: int, values: list[float], source: Path, line_number: int) -> Box:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{source}:{line_number}: coordenadas no finitas.")
    if len(values) == 4:
        cx, cy, width, height = values
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and width > 0 and height > 0):
            raise ValueError(f"{source}:{line_number}: bbox YOLO inválida.")
        return Box(class_id, cx, cy, width, height)
    if len(values) < 6 or len(values) % 2:
        raise ValueError(f"{source}:{line_number}: se esperaba bbox o polígono YOLO con >=3 puntos.")

    # Roboflow puede dejar puntos unos pocos píxeles fuera del marco. Se recortan
    # al área normalizada de la imagen antes de formar una caja de detección.
    xs = [min(1.0, max(0.0, values[index])) for index in range(0, len(values), 2)]
    ys = [min(1.0, max(0.0, values[index])) for index in range(1, len(values), 2)]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{source}:{line_number}: polígono degenerado al limitarlo a la imagen.")
    return Box(class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def _read_source_labels(origin: str) -> tuple[dict[str, tuple[Path, list[Box]]], Counter[int], Counter[str]]:
    dataset_name = SEGMENTED_DIRS[origin]
    dataset_dir = SEGMENTED_ROOT / dataset_name
    config_path = dataset_dir / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No encuentro el YAML de clases del dataset: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_names = config.get("names")
    if not isinstance(source_names, list):
        raise ValueError(f"{config_path}: `names` debe ser una lista de clases.")
    class_map = CLASS_MAPS[origin]
    unknown = [name for name in source_names if str(name).casefold() not in class_map]
    if unknown:
        raise ValueError(f"Clases no mapeadas en {config_path}: {unknown}")

    records: dict[str, tuple[Path, list[Box]]] = {}
    class_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Faltan carpetas images/labels para {dataset_name}/{split}.")
        labels = {path.stem.casefold(): path for path in label_dir.glob("*.txt")}
        images = [path for path in image_dir.iterdir() if path.is_file()]
        if len({path.stem.casefold() for path in images}) != len(images):
            raise ValueError(f"Nombres de imagen duplicados en {image_dir}.")
        image_stems = {path.stem.casefold() for path in images}
        if image_stems != set(labels):
            missing = sorted(image_stems - set(labels))[:5]
            orphan = sorted(set(labels) - image_stems)[:5]
            raise ValueError(f"Pares imagen/etiqueta incompletos en {dataset_name}/{split}; sin etiqueta={missing}, huérfanas={orphan}.")

        for image_path in images:
            key = image_path.stem.casefold()
            if key in records:
                raise ValueError(f"La misma imagen aparece en más de un split de {dataset_name}: {image_path.name}")
            label_path = labels[key]
            boxes: list[Box] = []
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    source_class = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Etiqueta inválida en {label_path}:{line_number}") from exc
                if not 0 <= source_class < len(source_names):
                    raise ValueError(f"ID de clase fuera de rango en {label_path}:{line_number}")
                target_class = class_map[str(source_names[source_class]).casefold()]
                boxes.append(_polygon_or_box_to_bbox(target_class, values, label_path, line_number))
                class_counts[target_class] += 1
            records[key] = (image_path, boxes)
            split_counts[split] += 1

    print(
        f"{dataset_name}: {len(records)} imágenes; clases destino="
        f"{ {CLASS_NAMES[class_id]: class_counts[class_id] for class_id in range(len(CLASS_NAMES))} }; "
        f"splits originales ignorados={dict(split_counts)}"
    )
    return records, class_counts, split_counts


def import_segmentado(*, overwrite: bool = False) -> None:
    source_records = {origin: _read_source_labels(origin) for origin in ("blanco", "rama")}
    prepared: dict[str, list[tuple[Path, Path, list[Box], str]]] = {}
    for origin, (records, _counts, _splits) in source_records.items():
        raw_images = list_images(RAW_SOURCE_DIRS[origin])
        raw_by_stem = {path.stem.casefold(): path for path in raw_images}
        if len(raw_by_stem) != len(raw_images):
            raise ValueError(f"Hay nombres crudos duplicados en {RAW_SOURCE_DIRS[origin]}.")
        if set(records) != set(raw_by_stem):
            missing = sorted(set(raw_by_stem) - set(records))[:8]
            extra = sorted(set(records) - set(raw_by_stem))[:8]
            raise ValueError(
                f"Las imágenes Segmentado no coinciden exactamente con el origen permitido {origin}: "
                f"sin etiquetas={missing}, extras={extra}. No se importó nada."
            )
        prepared[origin] = [
            (raw_by_stem[stem], source_image, boxes, scene_id_from_filename(source_image.name))
            for stem, (source_image, boxes) in records.items()
        ]

    # Validate all original images before backing up or replacing any annotation.
    for origin, records in prepared.items():
        for raw_image, _segmented_image, _boxes, _scene in records:
            if read_image(raw_image) is None:
                raise ValueError(f"Imagen cruda ilegible en {origin}; no se modificaron anotaciones: {raw_image}")

    existing = [path for path in ANNOTATION_DIRS.values() if path.exists() and any(path.iterdir())]
    if existing and not overwrite:
        raise FileExistsError("Ya existen anotaciones. Usa `import-segmentado --overwrite` para respaldarlas y reemplazarlas.")

    backup_dir: Path | None = None
    if existing:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = RESULTS_DIR / "auto_label_archive" / stamp
        for origin, annotation_dir in ANNOTATION_DIRS.items():
            if annotation_dir.exists():
                target = backup_dir / origin
                target.mkdir(parents=True, exist_ok=True)
                for label_path in annotation_dir.glob("*.txt"):
                    shutil.copy2(label_path, target / label_path.name)
                for image_path in annotation_dir.iterdir():
                    if image_path.is_file() and image_path.suffix.lower() != ".txt":
                        (target / image_path.name).write_bytes(image_path.read_bytes())
                for child in annotation_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        shutil.rmtree(child)

    imported_counts: dict[str, dict[str, int]] = {}
    for origin, records in prepared.items():
        annotation_dir = ANNOTATION_DIRS[origin]
        annotation_dir.mkdir(parents=True, exist_ok=True)
        counts: Counter[int] = Counter()
        for raw_image, _segmented_image, boxes, _scene in records:
            image = read_image(raw_image)
            if image is None:
                raise ValueError(f"Imagen cruda ilegible: {raw_image}")
            destination = annotation_dir / raw_image.name
            if not write_image(destination, image):
                raise OSError(f"No se pudo guardar la imagen cruda anotada: {destination}")
            label_path = destination.with_suffix(".txt")
            label_path.write_text(boxes_to_yolo_text(boxes), encoding="utf-8")
            checked = parse_yolo_labels(label_path)
            counts.update(box.class_id for box in checked)
        imported_counts[origin] = {CLASS_NAMES[index]: int(counts[index]) for index in range(len(CLASS_NAMES))}
        print(f"Importado {origin}: {len(records)} imágenes, {sum(counts.values())} cajas.")

    provenance = {
        "kind": "provided_segmented_dataset",
        "source": "dataset/Segmentado",
        "format_source": "YOLO segmentation polygons; converted to detection bounding boxes",
        "class_names": list(CLASS_NAMES),
        "class_maps": {
            "Fondo_blanco": {"Maduros": "maduro", "Pintones": "pinton", "Secos": "negro", "Sobremaduros": "negro", "Verdes": "verde"},
            "Cafe_arbol": {"overripe": "negro", "ripe": "maduro", "semi_ripe": "pinton", "unripe": "verde"},
        },
        "class_counts": imported_counts,
        "original_splits_reused": False,
        "scene_split_policy": "project grouped splits are rebuilt by scene id",
        "source_split_counts": {
            origin: {split: int(count) for split, count in source_records[origin][2].items()}
            for origin in source_records
        },
        "backup_annotations": str(backup_dir) if backup_dir else None,
        "note": "Provided polygons and class mapping; inspect held-out scene metrics and samples.",
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "annotation_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Procedencia: {DATA_DIR / 'annotation_provenance.json'}")
    if backup_dir:
        print(f"Respaldo de anotaciones anteriores: {backup_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa las etiquetas de dataset/Segmentado para las fotos crudas permitidas.")
    parser.add_argument("--overwrite", action="store_true", help="Respalda y reemplaza anotaciones actuales.")
    args = parser.parse_args()
    import_segmentado(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

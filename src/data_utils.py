from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

CLASS_NAMES = ("maduro", "pinton", "verde", "negro")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_SUFFIX_RE = re.compile(r"_(?:jpg|jpeg|png|bmp|webp)\.rf\.[^.]+$", re.IGNORECASE)
SCENE_VARIANT_PREFIX_RE = re.compile(r"^[1-9]_")
SCENE_DATE_PREFIX_RE = re.compile(r"^\d{8}_\d{6}")
SCENE_IMAGE_PREFIX_RE = re.compile(r"^IMG_", re.IGNORECASE)

RAW_SOURCE_DIRS = {
    "blanco": ROOT_DIR / "dataset" / "Fotos" / "Fotos_fondo_blanco",
    "rama": ROOT_DIR / "dataset" / "Fotos" / "Fotos_palo_cafe_granos",
}
ANNOTATION_DIRS = {
    "blanco": DATA_DIR / "annotations_raw" / "blanco",
    "rama": DATA_DIR / "annotations_raw" / "rama",
}


def read_image(path: Path, flags: int = -1):
    """Decode through Python file I/O so Windows paths with accents work."""
    import cv2
    import numpy as np

    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path: Path, image, params: list[int] | None = None) -> bool:
    """Encode with OpenCV and write using Unicode-safe pathlib I/O."""
    import cv2

    success, encoded = cv2.imencode(path.suffix, image, params or [])
    if not success:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return True


@contextmanager
def ascii_project_root():
    """Expose the project through a temporary ASCII drive for Ultralytics/OpenCV on Windows."""
    if os.name != "nt" or all(ord(character) < 128 for character in str(ROOT_DIR)):
        yield ROOT_DIR
        return

    for letter in "ZYXWVUTSRQPONMLKJIHGFEDCBA":
        drive = f"{letter}:"
        if os.path.exists(f"{drive}\\"):
            continue
        subprocess.run(["subst", drive, str(ROOT_DIR)], check=True, capture_output=True)
        try:
            yield Path(f"{drive}\\")
        finally:
            subprocess.run(["subst", drive, "/D"], check=True, capture_output=True)
        return
    raise RuntimeError("No hay una letra de unidad libre para una ruta ASCII temporal de YOLO.")


@dataclass(frozen=True)
class Box:
    """One YOLO detection box using normalized center-x, center-y, width, height."""

    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    def pixel_bounds(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        left = max(0, int(math.floor((self.cx - self.width / 2) * image_width)))
        top = max(0, int(math.floor((self.cy - self.height / 2) * image_height)))
        right = min(image_width, int(math.ceil((self.cx + self.width / 2) * image_width)))
        bottom = min(image_height, int(math.ceil((self.cy + self.height / 2) * image_height)))
        if right <= left or bottom <= top:
            raise ValueError(f"La caja no ocupa píxeles válidos: {self}")
        return left, top, right, bottom


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def scene_id_from_filename(filename: str) -> str:
    """Return the original scene identifier shared by Roboflow variants and duplicate exports.

    Examples:
        canva_100_jpg.rf.abc123.jpg -> canva_100
        1_20211204_100335_jpg.rf.abc123.jpg -> 20211204_100335
        2_20211204_100335_jpg.rf.def456.jpg -> 20211204_100335
        20211204_100056_jpg.rf.abc123.jpg -> 20211204_100056
    """
    stem = Path(filename).stem
    stem = ROBOFLOW_SUFFIX_RE.sub("", stem)
    if SCENE_VARIANT_PREFIX_RE.match(stem):
        remainder = stem[2:]
        if SCENE_DATE_PREFIX_RE.match(remainder) or SCENE_IMAGE_PREFIX_RE.match(remainder):
            stem = remainder
    return stem


def parse_yolo_labels(path: Path) -> list[Box]:
    """Read five-column YOLO detection labels; an empty file is a valid negative frame."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de etiquetas: {path}")

    boxes: list[Box] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: se esperaban 5 columnas YOLO, llegaron {len(parts)}.")
        try:
            class_id = int(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: etiqueta YOLO inválida.") from exc
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path}:{line_number}: class_id fuera de 0..{len(CLASS_NAMES) - 1}.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: coordenada no finita.")
        cx, cy, width, height = values
        if width <= 0 or height <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
            raise ValueError(f"{path}:{line_number}: centro o dimensiones fuera de rango.")
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1.000001 or cy + height / 2 > 1.000001:
            raise ValueError(f"{path}:{line_number}: la caja sale de los límites normalizados de la imagen.")
        boxes.append(Box(class_id, cx, cy, width, height))
    return boxes


def boxes_to_yolo_text(boxes: Iterable[Box]) -> str:
    lines = [
        f"{box.class_id} {box.cx:.8f} {box.cy:.8f} {box.width:.8f} {box.height:.8f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def box_from_pixels(class_id: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Box:
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("La caja debe tener ancho y alto positivos.")
    return Box(
        class_id=class_id,
        cx=((x1 + x2) / 2) / width,
        cy=((y1 + y2) / 2) / height,
        width=(x2 - x1) / width,
        height=(y2 - y1) / height,
    )


def crop_to_square_224(image, box: Box, origin: str, seed: int):
    """Crop one box, retain its aspect ratio, darken connected white background, and resize."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.pixel_bounds(width, height)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError(f"Recorte vacío para la caja {box}.")

    if origin == "blanco":
        near_white = np.all(crop > 240, axis=2).astype(np.uint8)
        component_count, component_labels = cv2.connectedComponents(near_white, connectivity=8)
        edge_components = np.unique(
            np.concatenate((component_labels[0, :], component_labels[-1, :], component_labels[:, 0], component_labels[:, -1]))
        )
        background_ids = edge_components[edge_components != 0]
        if background_ids.size:
            background_mask = np.isin(component_labels, background_ids)
            rng = np.random.default_rng(seed)
            texture = rng.normal(loc=30.0, scale=4.0, size=crop.shape).clip(15, 55).astype(np.uint8)
            crop[background_mask] = texture[background_mask]

    crop_height, crop_width = crop.shape[:2]
    side = max(crop_height, crop_width)
    if origin == "blanco":
        padding_color = (30, 30, 30)
    else:
        border_pixels = np.concatenate((crop[0, :, :], crop[-1, :, :], crop[:, 0, :], crop[:, -1, :]), axis=0)
        padding_color = tuple(int(value) for value in np.median(border_pixels, axis=0))
    square = np.empty((side, side, 3), dtype=np.uint8)
    square[:, :] = padding_color
    offset_y = (side - crop_height) // 2
    offset_x = (side - crop_width) // 2
    square[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width] = crop
    return cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)


def require_torch_cuda():
    """Return PyTorch and require the CUDA device used internally by Ultralytics."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Ultralytics necesita PyTorch para entrenar YOLO.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch no detecta CUDA; YOLO no se ejecutará en CPU.")
    return torch, torch.cuda.get_device_name(0)

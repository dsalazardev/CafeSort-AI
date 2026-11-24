from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (  # noqa: E402
    ANNOTATION_DIRS,
    CLASS_NAMES,
    RAW_SOURCE_DIRS,
    Box,
    box_from_pixels,
    boxes_to_yolo_text,
    list_images,
    parse_yolo_labels,
    read_image,
)

WINDOW_NAME = "CafeSort-AI | Etiquetado YOLO"
IMAGE_AREA = (1040, 760)
SIDEBAR_WIDTH = 420
WINDOW_HEIGHT = 800
CLASS_COLORS = ((40, 40, 230), (0, 165, 255), (60, 190, 40), (100, 100, 100))


@dataclass
class EditableBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int | None = None

    def contains(self, x: int, y: int, margin: int = 0) -> bool:
        return self.x1 - margin <= x <= self.x2 + margin and self.y1 - margin <= y <= self.y2 + margin


class Annotator:
    def __init__(self, include_saved: bool = False) -> None:
        self.include_saved = include_saved
        self.image = None
        self.origin = ""
        self.image_path: Path | None = None
        self.display_scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.boxes: list[EditableBox] = []
        self.selected_index: int | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.status = "Selecciona una caja y pulsa 1-4 para asignar clase."
        self._display_image = None
        self.current_index = 0
        self.total_images = 0

    @staticmethod
    def automatic_white_candidates(image: np.ndarray) -> list[EditableBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 239, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = image.shape[0] * image.shape[1]
        min_area = max(150.0, image_area * 0.00025)
        max_area = image_area * 0.12
        boxes: list[EditableBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not min_area <= area <= max_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 4 or height < 4:
                continue
            boxes.append(EditableBox(x, y, x + width, y + height))
        boxes.sort(key=lambda box: (box.y1, box.x1))
        return boxes

    @staticmethod
    def _boxes_from_saved(image: np.ndarray, label_path: Path) -> list[EditableBox]:
        if not label_path.exists():
            return []
        height, width = image.shape[:2]
        boxes: list[EditableBox] = []
        for box in parse_yolo_labels(label_path):
            x1, y1, x2, y2 = box.pixel_bounds(width, height)
            boxes.append(EditableBox(x1, y1, x2, y2, box.class_id))
        return boxes

    def _annotation_paths(self) -> tuple[Path, Path]:
        assert self.image_path is not None
        destination = ANNOTATION_DIRS[self.origin]
        return destination / self.image_path.name, destination / f"{self.image_path.stem}.txt"

    def _load(self, origin: str, image_path: Path) -> None:
        self.origin = origin
        self.image_path = image_path
        image = read_image(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo abrir la imagen: {image_path}")
        self.image = image
        saved_image, saved_label = self._annotation_paths()
        if saved_label.exists() and not self.include_saved:
            self.boxes = []
            self.status = "Ya etiquetada; se omite."
        elif saved_label.exists():
            source_image = saved_image if saved_image.exists() else image_path
            self.image = read_image(source_image, cv2.IMREAD_COLOR)
            self.boxes = self._boxes_from_saved(self.image, saved_label)
            self.status = "Revisando anotacion existente."
        elif origin == "blanco":
            self.boxes = self.automatic_white_candidates(self.image)
            self.status = f"Cajas candidatas: {len(self.boxes)}. Revisa cada una y asigna su clase."
        else:
            self.boxes = []
            self.status = "Rama: dibuja una caja sobre cada cereza visible."
        self.selected_index = 0 if self.boxes else None
        self.drag_start = None
        self.drag_end = None

    def _image_point(self, x: int, y: int) -> tuple[int, int] | None:
        if x < 0 or y < 0 or x >= self.display_width or y >= self.display_height:
            return None
        height, width = self.image.shape[:2]
        return (
            max(0, min(width - 1, int(x / self.display_scale))),
            max(0, min(height - 1, int(y / self.display_scale))),
        )

    def _nearest_box(self, x: int, y: int) -> int | None:
        for index in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[index].contains(x, y, margin=4):
                return index
        return None

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _userdata) -> None:
        point = self._image_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            self.drag_start = point
            self.drag_end = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            end = point or self.drag_start
            start = self.drag_start
            self.drag_start = None
            self.drag_end = None
            if abs(end[0] - start[0]) >= 6 and abs(end[1] - start[1]) >= 6:
                height, width = self.image.shape[:2]
                x1, x2 = sorted((start[0], end[0]))
                y1, y2 = sorted((start[1], end[1]))
                box = EditableBox(max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2))
                if box.x2 > box.x1 and box.y2 > box.y1:
                    self.boxes.append(box)
                    self.selected_index = len(self.boxes) - 1
                    self.status = "Caja nueva; pulsa 1-4 para asignar clase."
            else:
                self.selected_index = self._nearest_box(*start)
                if self.selected_index is None:
                    self.status = "Clic en una caja para seleccionarla; arrastra para dibujar."
                else:
                    self.status = "Caja seleccionada; pulsa 1-4 para asignar o corregir clase."
        elif event == cv2.EVENT_RBUTTONDOWN:
            if point is None:
                return
            index = self._nearest_box(*point)
            if index is not None and 0 <= index < len(self.boxes):
                del self.boxes[index]
                if not self.boxes:
                    self.selected_index = None
                else:
                    self.selected_index = min(index, len(self.boxes) - 1)
                self.status = "Caja eliminada."

    def _render(self) -> np.ndarray:
        image_height, image_width = self.image.shape[:2]
        self.display_scale = min(IMAGE_AREA[0] / image_width, IMAGE_AREA[1] / image_height)
        self.display_width = max(1, int(round(image_width * self.display_scale)))
        self.display_height = max(1, int(round(image_height * self.display_scale)))
        preview = cv2.resize(self.image, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)

        for index, box in enumerate(self.boxes):
            color = CLASS_COLORS[box.class_id] if box.class_id is not None else (0, 220, 255)
            thickness = 3 if index == self.selected_index else 2
            cv2.rectangle(
                preview,
                (int(box.x1 * self.display_scale), int(box.y1 * self.display_scale)),
                (int(box.x2 * self.display_scale), int(box.y2 * self.display_scale)),
                color,
                thickness,
            )
            label = CLASS_NAMES[box.class_id] if box.class_id is not None else "PROPUESTA - SIN CLASE"
            cv2.putText(preview, f"{index + 1}: {label}", (int(box.x1 * self.display_scale), max(18, int(box.y1 * self.display_scale) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        if self.drag_start is not None and self.drag_end is not None:
            p1 = (int(self.drag_start[0] * self.display_scale), int(self.drag_start[1] * self.display_scale))
            p2 = (int(self.drag_end[0] * self.display_scale), int(self.drag_end[1] * self.display_scale))
            cv2.rectangle(preview, p1, p2, (255, 255, 0), 2)

        canvas = np.zeros((WINDOW_HEIGHT, self.display_width + SIDEBAR_WIDTH, 3), dtype=np.uint8)
        canvas[:, :] = (28, 31, 35)
        canvas[:self.display_height, :self.display_width] = preview
        panel = canvas[:, self.display_width:]
        lines = [
            "ETIQUETADO MANUAL - NO PREDICE",
            f"Foto {self.current_index}/{self.total_images} | origen: {self.origin}",
            f"Foto: {self.image_path.name[:36]}",
            f"Cajas: {len(self.boxes)} | Sin clase: {sum(box.class_id is None for box in self.boxes)}",
            "",
            "Tu tarea: marcar la clase real de cada cereza.",
            "Aún no hay predicción: falta etiquetar y entrenar.",
            "Selecciona una caja; luego pulsa su tecla:",
            "1 maduro (ID 0)",
            "2 pinton (ID 1)",
            "3 verde (ID 2)",
            "4 negro (ID 3)",
            "",
            "FONDO BLANCO: amarillo = propuesta",
            "Revisa las propuestas; pueden ser falsas",
            "o unir varias cerezas en una sola caja.",
            "RAMA: dibuja una caja por cada cereza.",
            "",
            "Clic izq: seleccionar | arrastrar: crear",
            "Clic der sobre caja: eliminar",
            "",
            "N / ENTER: guardar y siguiente",
            "S: saltar sin guardar",
            "R: rehacer propuestas de fondo blanco",
            "ESC: guardar y salir (sin clase no guarda)",
            "",
            f"Estado: {self.status[:48]}",
        ]
        for index, text in enumerate(lines):
            y = 28 + index * 28
            if y >= WINDOW_HEIGHT - 8:
                break
            color = (120, 220, 255) if text.startswith(("1 ", "2 ", "3 ", "4 ")) else (235, 235, 235)
            scale = 0.43 if len(text) > 43 else 0.48
            thickness = 2 if index in (0, 5) else 1
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return canvas

    def _save(self) -> bool:
        if any(box.class_id is None for box in self.boxes):
            self.status = "Faltan clases: selecciona cada caja y pulsa 1-4, o elimina la falsa."
            return False
        assert self.image_path is not None
        image_path, label_path = self._annotation_paths()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists() or image_path.resolve() != self.image_path.resolve():
            import shutil
            shutil.copy2(self.image_path, image_path)
        height, width = self.image.shape[:2]
        normalized = [
            box_from_pixels(box.class_id, box.x1, box.y1, box.x2, box.y2, width, height)
            for box in self.boxes
        ]
        label_path.write_text(boxes_to_yolo_text(normalized), encoding="utf-8")
        self.status = f"Guardado: {label_path.name}"
        return True

    def run(self) -> None:
        items = [(origin, image) for origin, folder in RAW_SOURCE_DIRS.items() for image in list_images(folder)]
        if not items:
            raise FileNotFoundError("No se encontraron fotos en las dos carpetas crudas permitidas.")
        stems: set[tuple[str, str]] = set()
        for origin, image in items:
            key = (origin, image.stem.casefold())
            if key in stems:
                raise ValueError(f"Hay dos imágenes con el mismo nombre base en {origin}: {image.stem}")
            stems.add(key)

        self.total_images = len(items)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        position = 0
        try:
            while position < len(items):
                self.current_index = position + 1
                origin, image_path = items[position]
                saved_image = ANNOTATION_DIRS[origin] / image_path.name
                saved_label = ANNOTATION_DIRS[origin] / f"{image_path.stem}.txt"
                if saved_label.exists() and not self.include_saved:
                    position += 1
                    continue

                self._load(origin, image_path)
                should_advance = False
                while not should_advance:
                    cv2.imshow(WINDOW_NAME, self._render())
                    key = cv2.waitKey(25) & 0xFF
                    if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                        class_id = key - ord("1")
                        if self.selected_index is not None:
                            self.boxes[self.selected_index].class_id = class_id
                            self.status = f"Caja {self.selected_index + 1}: {CLASS_NAMES[class_id]}"
                        else:
                            self.status = "Primero selecciona o dibuja una caja."
                    elif key in (ord("n"), ord("N"), 10, 13):
                        if self._save():
                            should_advance = True
                    elif key in (ord("s"), ord("S")):
                        should_advance = True
                    elif key in (ord("r"), ord("R")) and origin == "blanco":
                        self.boxes = self.automatic_white_candidates(self.image)
                        self.selected_index = 0 if self.boxes else None
                        self.status = "Candidatas regeneradas; revisa y asigna clase."
                    elif key == 27:
                        if self._save():
                            return
                position += 1
        finally:
            cv2.destroyAllWindows()
        print(f"Etiquetado terminado. Fotos revisadas: {len(items)}")


def run_annotator(include_saved: bool = False) -> None:
    Annotator(include_saved=include_saved).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Etiqueta cerezas de las dos carpetas crudas de CafeSort-AI.")
    parser.add_argument("--review", action="store_true", help="Abrir también las fotos ya etiquetadas para corregirlas.")
    args = parser.parse_args(argv)
    run_annotator(include_saved=args.review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

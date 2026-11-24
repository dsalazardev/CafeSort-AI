from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "luckfox_rv1106"
CLASSES = ["maduro", "pinton", "verde", "negro"]


def _copy_model_bundle(model_output: Path) -> None:
    model_dirs = [p for p in model_output.rglob("*.rknn")]
    if not model_dirs:
        raise FileNotFoundError(f"No se encontró ningún .rknn en {model_output}")
    for source in model_dirs:
        destination = OUT / "yolo26n-cafe-rv1106.rknn"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        print(f"Modelo copiado: {destination}")
    for source in model_output.rglob("tail.onnx"):
        shutil.copy2(source, OUT / "yolo26n-rv1106-tail.onnx")
    for source in model_output.rglob("metadata.yaml"):
        shutil.copy2(source, OUT / "yolo26n-rv1106-metadata.yaml")


def export_yolo() -> None:
    from ultralytics import YOLO

    weights = ROOT / "results" / "yolo" / "train-3" / "weights" / "best.pt"
    dataset = ROOT / "data" / "yolo_dataset"
    yaml_path = Path("/tmp/cafesort_rv1106_yolo_calibration.yaml")
    yaml_path.write_text(
        "path: " + json.dumps(dataset.as_posix()) + "\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "nc: 4\n"
        "names: [maduro, pinton, verde, negro]\n",
        encoding="utf-8",
    )
    model = YOLO(str(weights))
    print(f"Exportando YOLO26n INT8 para rv1106 con calibración de train: {yaml_path}")
    expected_export = weights.parent / f"{weights.stem}_rknn_model"
    if not expected_export.exists() or not any(expected_export.glob("*.rknn")):
        exported = model.export(
            format="rknn",
            quantize=8,
            data=str(yaml_path),
            fraction=1.0,
            imgsz=640,
            batch=1,
            device="cpu",
            project=str(OUT),
            name="rv1106",
            exist_ok=True,
        )
        expected_export = Path(exported)
        print(f"Ultralytics exportó: {expected_export}")
    else:
        print(f"Se reutiliza la conversión ya creada: {expected_export}")
    _copy_model_bundle(expected_export)
    if not (OUT / "yolo26n-rv1106-tail.onnx").is_file():
        raise FileNotFoundError("YOLO26 INT8 requiere su tail.onnx para decodificar las detecciones")


def _mobile_calibration_list() -> Path:
    randomizer = random.Random(1106)
    crops_root = ROOT / "data" / "mobilenet_dataset" / "train"
    selected: list[Path] = []
    for class_name in CLASSES:
        candidates = sorted((crops_root / class_name).glob("*"))
        candidates = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(candidates) < 125:
            raise RuntimeError(f"Se esperaban 125 imágenes de calibración para {class_name}, hay {len(candidates)}")
        selected.extend(randomizer.sample(candidates, 125))
    path = Path("/tmp/cafesort_rv1106_mobilenet_calibration.txt")
    path.write_text("\n".join(p.as_posix() for p in selected) + "\n", encoding="utf-8")
    return path


def export_mobilenet() -> None:
    from rknn.api import RKNN

    source = ROOT / "results" / "mobilenet" / "mobilenetv2_cafe.onnx"
    calibration = _mobile_calibration_list()
    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            target_platform="rv1106",
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            quantized_algorithm="normal",
        )
        result = rknn.load_onnx(model=str(source))
        if result != 0:
            raise RuntimeError(f"RKNN no pudo cargar MobileNet ONNX (código {result})")
        result = rknn.build(do_quantization=True, dataset=str(calibration))
        if result != 0:
            raise RuntimeError(f"Falló la conversión RKNN INT8 de MobileNet (código {result})")
        destination = OUT / "mobilenetv2-cafe-rv1106.rknn"
        result = rknn.export_rknn(str(destination))
        if result != 0:
            raise RuntimeError(f"Falló la exportación de MobileNet RKNN (código {result})")
    finally:
        rknn.release()
    print(f"MobileNetV2 RKNN: {destination}")


def write_bundle_docs() -> None:
    (OUT / "labels.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    metadata = {
        "target": "Rockchip RV1106 / Luckfox Pico Max",
        "precision": "INT8",
        "classes": {str(i): name for i, name in enumerate(CLASSES)},
        "models": {
            "yolo26n-cafe-rv1106.rknn": {
                "purpose": "Detección y clasificación sobre fotograma completo",
                "input_size": [640, 640],
                "input_to_rknn": "RGB NHWC uint8 0..255, letterbox 114; no dividir por 255",
                "postprocess": "Usar yolo26n-rv1106-tail.onnx y NMS. Requiere ONNX Runtime CPU en placa o implementar la decodificación en C/C++.",
            },
            "mobilenetv2-cafe-rv1106.rknn": {
                "purpose": "Clasificación de un recorte ya localizado",
                "input_size": [224, 224],
                "input_to_rknn": "RGB NHWC uint8 0..255, recorte de un grano y resize; no aplicar normalización externa",
                "output": "4 scores en el orden de labels.txt; pedir salida float al runtime o de-cuantizar con sus escalas.",
            },
        },
        "conversion_toolkit": "RKNN-Toolkit2 2.3.2",
        "quantization_data": {
            "yolo": "data/yolo_dataset/images/train (split train, sin test)",
            "mobilenet": "500 crops, 125 por clase, del split train",
        },
        "not_hardware_validated": True,
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "model_manifest.json"
        },
    }
    (OUT / "model_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README_LUCKFOX.md").write_text(
        """# CafeSort-AI para Luckfox Pico Max (RV1106)

Carpeta de modelos convertidos al formato RKNN INT8 para el NPU RV1106.

## Archivos

- `yolo26n-cafe-rv1106.rknn`: detector principal; localiza y clasifica.
- `yolo26n-rv1106-tail.onnx`: decodificación de la cabeza YOLO INT8.
- `mobilenetv2-cafe-rv1106.rknn`: clasificador opcional de recortes; MobileNet no detecta ni localiza granos.
- `labels.txt`: clases por ID: maduro, pinton, verde, negro.
- `model_manifest.json`: dimensiones, preprocesamiento y limitaciones de salida.
- Los ONNX originales sin cuantizar siguen en `results/yolo/` y `results/mobilenet/`; no se necesitan en la placa.

## Importante para el código de cámara

YOLO recibe fotograma RGB 640x640 en NHWC uint8, preparado con letterbox gris (114). Para el RKNN no dividas por 255. El `.rknn` INT8 de detección produce tensores de cabeza crudos; ejecuta el `tail.onnx` en CPU y NMS después, o implementa la misma decodificación/NMS en C/C++.

MobileNetV2 recibe un recorte de cereza RGB 224x224 en NHWC uint8 y clasifica solo ese recorte. Solicita al runtime la salida en float, o de-cuantízala con los parámetros que consulta del modelo. El pipeline completo es YOLO -> recortes de detección -> MobileNet (si se quiere probar la cascada); para una cámara, empieza con YOLO solo.

La conversión no sustituye una validación con la cámara y runtime exacto del Luckfox. Los `.rknn` se deben probar en la placa; la versión de `librknnrt`/driver del sistema debe ser compatible con RKNN-Toolkit2 2.3.2. Los modelos se calibraron con datos del split train; las fotos de cinta azul/verde no están representadas.

## Copia

Copia los `.rknn`, el tail ONNX y `labels.txt` a la placa junto con la aplicación C/C++/Python que inicializa `librknnrt` y realiza el postprocesamiento. No copies los ONNX de referencia para inferencia NPU.

## Reproducción de la exportación

Este proyecto usa Python 3.14 para entrenar. RKNN-Toolkit2 2.3.2 exige una herramienta de exportación compatible con Python 3.12 o anterior; `scripts/export_luckfox.py` se ejecuta en el entorno temporal `/tmp/cafesort-rknn-venv`, sin cambiar el entorno de entrenamiento.
""",
        encoding="utf-8",
    )
    metadata["files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(OUT.iterdir())
        if path.is_file() and path.name != "model_manifest.json"
    }
    (OUT / "model_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        export_yolo()
        export_mobilenet()
        write_bundle_docs()
    except Exception as exc:
        print(f"ERROR exportando bundle RV1106: {exc}", file=sys.stderr)
        raise
    print(f"\nBundle RV1106 listo en: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

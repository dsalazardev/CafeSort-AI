from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from src.data_utils import DATA_DIR, RESULTS_DIR, ROOT_DIR, ascii_project_root, require_torch_cuda

DEFAULT_EPOCHS = 50
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 8


def train(
    epochs: int = DEFAULT_EPOCHS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weights: str = "yolo26n.pt",
) -> Path:
    data_yaml = DATA_DIR / "yolo_dataset" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            "Falta data/yolo_dataset/data.yaml. Ese archivo se crea al ejecutar `python main.py build-splits`, "
            "después de guardar anotaciones YOLO en `data/annotations_raw/`. Las fotos crudas por sí solas "
            "no especifican la caja y clase de cada cereza."
        )

    _, gpu_name = require_torch_cuda()
    from ultralytics import YOLO

    print(f"YOLO entrenará con Ultralytics/PyTorch en CUDA: {gpu_name}")
    with ascii_project_root() as project_root:
        project_dir = project_root / "results" / "yolo"
        project_dir.mkdir(parents=True, exist_ok=True)
        runtime_yaml = project_root / "data" / "yolo_dataset" / "data_runtime.yaml"
        data_config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        data_config["path"] = str(project_root / "data" / "yolo_dataset")
        runtime_yaml.write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        try:
            model = YOLO(weights)
            model.train(
                data=str(runtime_yaml),
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                device=0,
                workers=0,
                project=str(project_dir),
                name="train",
                exist_ok=False,
                seed=42,
                deterministic=True,
                plots=True,
                verbose=True,
            )
            run_dir = Path(model.trainer.save_dir)
            best_weights = run_dir / "weights" / "best.pt"
            if not best_weights.is_file():
                raise FileNotFoundError(f"Ultralytics no produjo los pesos esperados: {best_weights}")

            best_model = YOLO(str(best_weights))
            best_model.val(
                data=str(runtime_yaml),
                split="test",
                imgsz=image_size,
                batch=batch_size,
                device=0,
                workers=0,
                project=str(project_dir),
                name=f"{run_dir.name}_test",
                exist_ok=True,
                plots=True,
                verbose=True,
            )
            exported = Path(
                best_model.export(
                    format="onnx",
                    imgsz=image_size,
                    opset=12,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    device=0,
                )
            )
            import onnx

            exported_model = onnx.load(str(exported))
            onnx.checker.check_model(exported_model)
            opsets = [entry.version for entry in exported_model.opset_import if entry.domain in ("", "ai.onnx")]
            if opsets != [12]:
                raise RuntimeError(f"El ONNX de YOLO debe usar Opset 12; el archivo exportado usa {opsets}.")
            stable_onnx = project_dir / "yolo_cafe.onnx"
            if exported.resolve() != stable_onnx.resolve():
                shutil.copy2(exported, stable_onnx)

            relative_run = run_dir.relative_to(project_dir)
            actual_weights = RESULTS_DIR / "yolo" / relative_run / "weights" / "best.pt"
            actual_onnx = RESULTS_DIR / "yolo" / "yolo_cafe.onnx"
            print(f"Pesos YOLO: {actual_weights}")
            print(f"Modelo ONNX: {actual_onnx}")
            return actual_weights
        finally:
            runtime_yaml.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena YOLO con Ultralytics y CUDA.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--weights", default="yolo26n.pt", help="Pesos iniciales Ultralytics, por defecto yolo26n.pt.")
    args = parser.parse_args(argv)
    train(epochs=args.epochs, image_size=args.imgsz, batch_size=args.batch_size, weights=args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

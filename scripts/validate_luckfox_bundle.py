from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "luckfox_rv1106"


def main() -> None:
    expected = {
        "yolo26n-cafe-rv1106.rknn",
        "mobilenetv2-cafe-rv1106.rknn",
        "yolo26n-rv1106-tail.onnx",
        "yolo26n-rv1106-metadata.yaml",
        "labels.txt",
        "model_manifest.json",
        "README_LUCKFOX.md",
    }
    missing = sorted(name for name in expected if not (OUT / name).is_file())
    if missing:
        raise FileNotFoundError(f"Faltan archivos de despliegue: {missing}")
    for name in ("yolo26n-cafe-rv1106.rknn", "mobilenetv2-cafe-rv1106.rknn"):
        if (OUT / name).stat().st_size < 100_000:
            raise RuntimeError(f"El archivo RKNN parece incompleto: {name}")
    labels = (OUT / "labels.txt").read_text(encoding="utf-8").splitlines()
    if labels != ["maduro", "pinton", "verde", "negro"]:
        raise RuntimeError(f"Orden de etiquetas incorrecto: {labels}")
    tail_path = OUT / "yolo26n-rv1106-tail.onnx"
    tail = onnx.load(str(tail_path))
    onnx.checker.check_model(tail)
    manifest_path = OUT / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["target"] != "Rockchip RV1106 / Luckfox Pico Max":
        raise RuntimeError("El manifiesto no declara el objetivo RV1106")
    bad_hashes = []
    for name, expected_hash in manifest.get("files_sha256", {}).items():
        path = OUT / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual_hash != expected_hash:
            bad_hashes.append(name)
    if bad_hashes:
        raise RuntimeError(f"Checksum incorrecto o archivo ausente: {bad_hashes}")
    print(f"Bundle comprobado: {OUT}")
    for name in sorted(expected):
        path = OUT / name
        print(f"{name}: {path.stat().st_size:,} bytes")
    print(f"Tail ONNX: {len(tail.graph.input)} entradas, {len(tail.graph.output)} salidas; ONNX checker OK")
    print("Checksum de artefactos registrados: OK")
    print("Nota: esta comprobación no sustituye ejecutar los .rknn en la placa física.")


if __name__ == "__main__":
    main()

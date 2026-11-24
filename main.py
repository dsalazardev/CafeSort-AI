from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent


def _run_isolated_module(module: str, arguments: list[str]) -> None:
    """Run one GPU workload in a fresh process so its CUDA allocator is released on exit."""
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT_DIR,
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cafesort-ai",
        description="Etiquetado, entrenamiento y comparación de clasificadores de cereza de café.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    label = commands.add_parser("label", help="Abre el anotador de cajas YOLO para las fotos permitidas.")
    label.add_argument("--review", action="store_true", help="Permite corregir fotos ya etiquetadas.")

    auto_label = commands.add_parser("auto-label", help="Segmenta el fondo blanco y pre-etiqueta ramas con YOLO temporal.")
    auto_label.add_argument("--overwrite", action="store_true", help="Reemplaza anotaciones automáticas existentes.")
    auto_label.add_argument("--weights", default="yolo26n.pt")
    auto_label.add_argument("--batch-size", type=int, default=8)
    auto_label.add_argument("--workers", type=int, default=4)
    auto_label.add_argument("--conf", type=float, default=0.35)
    auto_label.add_argument("--iou", type=float, default=0.5)

    import_segmentado = commands.add_parser("import-segmentado", help="Importa y remapea las etiquetas de dataset/Segmentado.")
    import_segmentado.add_argument("--overwrite", action="store_true", help="Respalda las etiquetas actuales y las reemplaza.")

    splits = commands.add_parser("build-splits", help="Crea splits por origen, manifiesto y datasets.")
    splits.add_argument("--overwrite", action="store_true", help="Reemplaza solo las salidas generadas.")

    mobilenet = commands.add_parser("train-mobilenet", help="Entrena MobileNetV2 con TensorFlow/Keras.")
    mobilenet.add_argument("--epochs", type=int, default=25)
    mobilenet.add_argument("--batch-size", type=int, default=16)
    mobilenet.add_argument("--scratch", action="store_true", help="No usa pesos iniciales ImageNet.")
    mobilenet.add_argument("--allow-cpu", action="store_true", help="Permite CPU si TensorFlow no detecta CUDA.")

    yolo = commands.add_parser("train-yolo", help="Entrena YOLO mediante Ultralytics/PyTorch y CUDA.")
    yolo.add_argument("--epochs", type=int, default=50)
    yolo.add_argument("--imgsz", type=int, default=640)
    yolo.add_argument("--batch-size", type=int, default=8)
    yolo.add_argument("--weights", default="yolo26n.pt")

    benchmark = commands.add_parser("benchmark", help="Compara ambos modelos con ONNX Runtime sobre el split test compartido.")
    benchmark.add_argument("--conf", type=float, default=0.25)
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--batch-size", type=int, default=16)

    all_command = commands.add_parser("all", help="Ejecuta etiquetado, preparación, entrenamiento y comparación.")
    all_command.add_argument("--review", action="store_true")
    all_command.add_argument("--overwrite", action="store_true")
    all_command.add_argument("--mobilenet-epochs", type=int, default=25)
    all_command.add_argument("--yolo-epochs", type=int, default=50)
    all_command.add_argument("--mobilenet-batch-size", type=int, default=16)
    all_command.add_argument("--yolo-batch-size", type=int, default=8)
    all_command.add_argument("--imgsz", type=int, default=640)
    all_command.add_argument("--scratch", action="store_true")
    all_command.add_argument("--allow-mobilenet-cpu", action="store_true")
    all_command.add_argument("--weights", default="yolo26n.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "label":
        from scripts.etiquetar_dataset import run_annotator

        run_annotator(include_saved=args.review)
    elif args.command == "auto-label":
        from scripts.auto_etiquetar import run_auto_label

        run_auto_label(
            overwrite=args.overwrite,
            weights=args.weights,
            batch_size=args.batch_size,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
        )
    elif args.command == "import-segmentado":
        from scripts.import_segmentado import import_segmentado as run_import_segmentado

        run_import_segmentado(overwrite=args.overwrite)
    elif args.command == "build-splits":
        from scripts.build_splits import run_build_splits

        run_build_splits(overwrite=args.overwrite)
    elif args.command == "train-mobilenet":
        from src.train_mobilenet import train

        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            pretrained=not args.scratch,
            allow_cpu=args.allow_cpu,
        )
    elif args.command == "train-yolo":
        from src.train_yolo import train

        train(epochs=args.epochs, image_size=args.imgsz, batch_size=args.batch_size, weights=args.weights)
    elif args.command == "benchmark":
        from src.benchmark_compare import benchmark

        benchmark(conf=args.conf, image_size=args.imgsz, batch_size=args.batch_size)
    elif args.command == "all":
        from scripts.etiquetar_dataset import run_annotator
        from scripts.build_splits import run_build_splits

        run_annotator(include_saved=args.review)
        run_build_splits(overwrite=args.overwrite)
        mobilenet_args = ["--epochs", str(args.mobilenet_epochs), "--batch-size", str(args.mobilenet_batch_size)]
        if args.scratch:
            mobilenet_args.append("--scratch")
        if args.allow_mobilenet_cpu:
            mobilenet_args.append("--allow-cpu")
        _run_isolated_module("src.train_mobilenet", mobilenet_args)
        _run_isolated_module(
            "src.train_yolo",
            ["--epochs", str(args.yolo_epochs), "--imgsz", str(args.imgsz), "--batch-size", str(args.yolo_batch_size), "--weights", args.weights],
        )
        _run_isolated_module(
            "src.benchmark_compare",
            ["--imgsz", str(args.imgsz), "--batch-size", str(args.mobilenet_batch_size)],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

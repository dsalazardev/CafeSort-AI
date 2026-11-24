from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tf2onnx
from sklearn.metrics import classification_report, confusion_matrix

from src.data_utils import CLASS_NAMES, DATA_DIR, RESULTS_DIR

IMG_SIZE = 224
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 16
AUTOTUNE_PARALLELISM = 2


def _configure_runtime(allow_cpu: bool) -> list[str]:
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as exc:
            raise RuntimeError(f"No se pudo limitar el crecimiento de memoria GPU ({device.name}).") from exc
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    if devices:
        print("MobileNetV2 usará TensorFlow en GPU:", ", ".join(device.name for device in devices))
    else:
        if not allow_cpu:
            raise RuntimeError(
                "MobileNetV2 requiere TensorFlow con GPU CUDA para este proyecto. "
                "Windows nativo no expone la RTX a TensorFlow; configura CUDA de TensorFlow en WSL2 "
                "o vuelve a ejecutar con --allow-cpu si aceptas entrenar en CPU."
            )
        print("AVISO: TensorFlow no detectó GPU; MobileNetV2 se entrenará en CPU por solicitud explícita.")
        print("En Windows nativo TensorFlow no ofrece CUDA; no se sustituye MobileNet por PyTorch.")
    return [device.name for device in devices]


def _dataset(directory: Path, batch_size: int, shuffle: bool):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42 if shuffle else None,
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = AUTOTUNE_PARALLELISM
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)


def _build_model(pretrained: bool = True) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_rgb")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    images = augmentation(inputs)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = True
    for layer in backbone.layers[:100]:
        layer.trainable = False
    features = backbone(images, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(features)
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_CafeSort")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def _export_onnx(model: tf.keras.Model, destination: Path) -> None:
    @tf.function(input_signature=[tf.TensorSpec((1, 3, IMG_SIZE, IMG_SIZE), tf.float32, name="input")])
    def serving(images_nchw):
        images_nhwc = tf.transpose(images_nchw, (0, 2, 3, 1))
        return {"class_probabilities": model(images_nhwc, training=False)}

    tf2onnx.convert.from_function(
        serving,
        input_signature=serving.input_signature,
        opset=12,
        output_path=str(destination),
    )
    import onnx
    import onnxruntime as ort

    exported = onnx.load(str(destination))
    onnx.checker.check_model(exported)
    default_opsets = [item.version for item in exported.opset_import if item.domain in ("", "ai.onnx")]
    if default_opsets != [12]:
        raise RuntimeError(f"MobileNetV2 debe exportarse con Opset 12; se obtuvo {default_opsets}.")
    input_shape = [item.dim_value for item in exported.graph.input[0].type.tensor_type.shape.dim]
    if input_shape != [1, 3, IMG_SIZE, IMG_SIZE]:
        raise RuntimeError(f"Entrada ONNX inesperada: {input_shape}; se esperaba [1, 3, 224, 224].")
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, IMG_SIZE, IMG_SIZE), np.float32)})[0]
    if result.shape != (1, len(CLASS_NAMES)):
        raise RuntimeError(f"Salida ONNX inesperada: {result.shape}; se esperaba (1, 4).")


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pretrained: bool = True,
    allow_cpu: bool = False,
) -> Path:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs y batch_size deben ser mayores que cero.")
    root = DATA_DIR / "mobilenet_dataset"
    split_dirs = {name: root / name for name in ("train", "val", "test")}
    for name, directory in split_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Falta {directory}. Ejecuta `python main.py label` y `python main.py build-splits`.")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    for class_name in CLASS_NAMES:
        class_dir = split_dirs["train"] / class_name
        if not class_dir.is_dir() or not any(
            path.is_file() and path.suffix.lower() in supported_extensions for path in class_dir.iterdir()
        ):
            raise RuntimeError(f"La clase `{class_name}` no tiene recortes en train. Revisa las etiquetas.")
    for name in ("val", "test"):
        if not any(
            path.is_file() and path.suffix.lower() in supported_extensions
            for class_name in CLASS_NAMES
            for path in (split_dirs[name] / class_name).glob("*")
            if (split_dirs[name] / class_name).is_dir()
        ):
            raise RuntimeError(f"El split `{name}` no contiene recortes para evaluar.")

    gpu_devices = _configure_runtime(allow_cpu=allow_cpu)
    datasets = {
        "train": _dataset(split_dirs["train"], batch_size, True),
        "val": _dataset(split_dirs["val"], batch_size, False),
        "test": _dataset(split_dirs["test"], batch_size, False),
    }
    train_counts = {
        class_name: sum(
            1 for path in (split_dirs["train"] / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in supported_extensions
        )
        for class_name in CLASS_NAMES
    }
    nonzero_counts = [count for count in train_counts.values() if count > 0]
    if len(nonzero_counts) != len(CLASS_NAMES):
        raise RuntimeError(f"No se puede balancear MobileNet: faltan clases en train: {train_counts}")
    total_train = sum(nonzero_counts)
    class_weight = {
        class_id: total_train / (len(CLASS_NAMES) * train_counts[class_name])
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    print("Muestras MobileNet por clase:", train_counts)
    print("Pesos balanceados de clase:", {CLASS_NAMES[index]: round(weight, 3) for index, weight in class_weight.items()})
    output_dir = RESULTS_DIR / "mobilenet"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mobilenetv2_cafe.keras"
    onnx_path = output_dir / "mobilenetv2_cafe.onnx"
    model = _build_model(pretrained)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "historial_entrenamiento.csv")),
    ]
    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    model = tf.keras.models.load_model(checkpoint_path)
    test_loss, test_accuracy = model.evaluate(datasets["test"], verbose=0)
    y_true = np.concatenate([labels.numpy() for _, labels in datasets["test"]]).astype(np.int64)
    y_pred = np.argmax(model.predict(datasets["test"], verbose=0), axis=1)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        digits=4,
        zero_division=0,
    )
    (output_dir / "reporte_clasificacion_MobileNetV2.txt").write_text(report, encoding="utf-8")
    (output_dir / "labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(CLASS_NAMES)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "resumen_entrenamiento.json").write_text(
        json.dumps(
            {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy), "gpu_devices": gpu_devices,
             "class_names": list(CLASS_NAMES), "image_size": IMG_SIZE},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Matriz de confusión - MobileNetV2")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    fig.tight_layout()
    fig.savefig(output_dir / "matriz_confusion_MobileNetV2.png", dpi=180)
    plt.close(fig)
    for metric, title, ylabel, filename in (
        ("loss", "Pérdida - MobileNetV2", "Cross-entropy", "perdida_MobileNetV2.png"),
        ("accuracy", "Exactitud - MobileNetV2", "Exactitud", "exactitud_MobileNetV2.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history.get(metric, []), label=f"train {metric}")
        ax.plot(history.history.get(f"val_{metric}", []), label=f"val {metric}")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
    _export_onnx(model, onnx_path)
    print(f"MobileNetV2 Keras: {checkpoint_path}")
    print(f"MobileNetV2 ONNX (NCHW, Opset 12): {onnx_path}")
    print(f"Exactitud test: {test_accuracy:.4f}\n{report}")
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena MobileNetV2 con TensorFlow/Keras.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--scratch", action="store_true", help="Inicializa MobileNetV2 sin pesos de ImageNet.")
    parser.add_argument("--allow-cpu", action="store_true", help="Permite entrenar en CPU si TensorFlow no detecta CUDA.")
    args = parser.parse_args(argv)
    train(epochs=args.epochs, batch_size=args.batch_size, pretrained=not args.scratch, allow_cpu=args.allow_cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

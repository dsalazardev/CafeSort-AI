# CafeSort-AI

Proyecto para crear anotaciones propias, entrenar MobileNetV2 y YOLO, exportarlos a ONNX y comparar sus resultados para el SoC RV1106 de Luckfox Pico Max.

## Datos permitidos

Solo se leen imágenes de `dataset/Fotos/Fotos_fondo_blanco/` y `dataset/Fotos/Fotos_palo_cafe_granos/`. El código no usa `dataset/Zip/` ni `dataset/Fotos/Fotos_general/`.

El orden de clases es fijo en anotación, entrenamiento, exportación y evaluación: ID 0 `maduro`, ID 1 `pinton`, ID 2 `verde`, ID 3 `negro`.

## Estado del proyecto

El etiquetador **no es un modelo de clasificación**: es una herramienta para que una persona dibuje/revise cajas y asigne la clase real de cada cereza. Las cajas amarillas en fotos de fondo blanco son propuestas geométricas, no predicciones; pueden ser falsas o abarcar más de una cereza. No hay clasificación entrenada hasta completar etiquetas, generar los splits y entrenar los modelos. El entrenamiento no se inicia automáticamente al abrir `label`.

## Entorno de entrenamiento con CUDA

El entrenamiento se ejecuta en Ubuntu 26.04 sobre WSL2, con Python 3.14 y un único entorno Linux `uv` que contiene TensorFlow/Keras, PyTorch/Ultralytics y CUDA. La etiqueta manual puede seguir ejecutándose desde Windows. El entorno Linux vive en `$HOME/.venvs/cafesort-ai` para no mezclar sus binarios con el `.venv` de Windows; ambos leen y escriben el mismo proyecto y sus datos en `/mnt/c`.

Abre Ubuntu 26.04 desde PowerShell:

El pre-etiquetado automático se ejecuta en WSL con CUDA mediante `uv run python main.py auto-label`. Segmenta las fotos blancas por umbral/morfología/watershed y entrena un `yolo26n.pt` temporal de 20 épocas para proponer cajas en rama. Escribe pseudo-etiquetas en `data/annotations_raw/`, seis previews de cada origen y un reporte en `results/auto_label_preview/`. Las previews y etiquetas son predicciones automáticas: deben revisarse y sus métricas no son exactitud validada por una persona.

Para generar las pseudo-etiquetas:

```bash
uv run python main.py auto-label
```

Revisa `results/auto_label_preview/` antes de seguir. `--overwrite` vuelve a generar y reemplaza los archivos de anotación existentes.

Cuando estén disponibles las etiquetas segmentadas de `dataset/Segmentado/`, puedes importarlas sobre las pseudo-etiquetas con:

```bash
uv run python main.py import-segmentado --overwrite
```

El importador convierte sus polígonos a cajas de detección, remapea `Secos` y `Sobremaduros` a `negro`, comprueba que todos los nombres coincidan con las fotos crudas permitidas y respalda las anotaciones previas en `results/auto_label_archive/`. Después regenera los splits por escena con `build-splits --overwrite`.

```powershell
wsl -d Ubuntu-26.04
```

Dentro de Ubuntu, configura el entorno una vez:

```bash
cd "/mnt/c/Users/SALAZAR/Desktop/COMPARTIDA/PROGRAMACIÓN/CafeSort-AI"
source scripts/wsl_env.sh
uv python install 3.14
uv sync
```

Comprueba que ambos frameworks vean la RTX 4060:

```bash
source scripts/wsl_env.sh
uv run python -c "import tensorflow as tf; print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))"
uv run python -c "import torch; print('YOLO CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

`uv.lock` fija las versiones resueltas. El proceso de MobileNet configura crecimiento de memoria GPU; los entrenamientos se ejecutan secuencialmente, no en paralelo, para liberar VRAM entre modelos.

## Flujo

1. Etiqueta/revisa las fotos:

   ```powershell
   uv run python main.py label
   ```

   Selecciona una caja y pulsa `1` maduro, `2` pinton, `3` verde o `4` negro. En blanco revisa y corrige las propuestas; en rama dibuja una caja por cereza. Arrastra con clic izquierdo para crear una caja, clic derecho sobre una caja la elimina, `N`/Enter guarda y avanza, `S` salta sin guardar, `R` regenera propuestas blancas y `Esc` guarda y sale. `--review` vuelve a abrir etiquetas existentes. Si una caja no tiene clase, la foto no se guarda.

   Se guarda imagen y etiqueta YOLO en `data/annotations_raw/blanco/` o `data/annotations_raw/rama/`. Una etiqueta vacía representa una foto revisada sin cerezas.

2. Crea particiones y ambos datasets desde el manifiesto común:

   ```powershell
   uv run python main.py build-splits
   ```

   Las imágenes de cada origen se separan a nivel de escena para evitar fuga de variantes duplicadas. Se generan `data/splits_manifest.json`, `data/yolo_dataset/` y recortes `224x224` con padding para `data/mobilenet_dataset/`. Las imágenes de blanco simulan el fondo de banda oscura detrás de cada recorte. El resumen muestra distribución real; con pocos datos/clases puede no resultar exactamente 60/20/20, y el comando debe detenerse si no hay suficientes escenas para evitar una partición engañosa.

3. Entrena por separado **desde la terminal Ubuntu/WSL2**, con el entorno configurado arriba:

   ```bash
   source scripts/wsl_env.sh
   uv run python main.py train-mobilenet
   uv run python main.py train-yolo
   ```

   MobileNetV2 es TensorFlow/Keras, entrada de imagen `224x224`, AdamW, ReduceLROnPlateau, aumento moderado y early stopping; exporta `.keras` y ONNX Opset 12 con entrada NCHW estática `1x3x224x224`. El ONNX recibe RGB `[0,255]` y lleva la normalización de MobileNetV2 integrada en la gráfica, para que benchmark e inferencia usen el mismo preprocesamiento.

   YOLO se ejecuta mediante `from ultralytics import YOLO`, utiliza PyTorch/CUDA a través de Ultralytics, y por defecto inicia desde `yolo26n.pt`. Se limita `workers=0`; `all` ejecuta cada entrenamiento en procesos secuenciales para liberar memoria GPU. Lotes predeterminados: 16 para MobileNet y 8 para YOLO.

4. Compara sobre las imágenes del split test compartido, también desde WSL2:

   ```bash
   source scripts/wsl_env.sh
   uv run python main.py benchmark
   ```

   MobileNet clasifica los recortes de anotación; su benchmark no mide detección. YOLO detecta y clasifica en la imagen completa. La comparación incluye métricas por clase/origen, confusión, tamaño ONNX y latencia ONNX Runtime CPU. La latencia MobileNet no incluye localizar la cereza; por tanto, las latencias no son directamente equivalentes.

   Los informes se escriben en `results/benchmark_summary.csv`, `results/benchmark_report.md` y `results/benchmark/comparacion.json`.

`uv run python main.py all` ejecuta etiqueta interactiva, partición, ambos entrenamientos y benchmark en secuencia. Para correr los modelos se requieren imágenes etiquetadas suficientes; no existen resultados de entrenamiento hasta que ese trabajo se complete.

## Artefactos

- MobileNet: `results/mobilenet/mobilenetv2_cafe.keras`, `.onnx`, matriz de confusión e informe.
- YOLO: `results/yolo/train*/weights/best.pt` y `results/yolo/yolo_cafe.onnx`.
- Benchmark: los tres informes indicados arriba.

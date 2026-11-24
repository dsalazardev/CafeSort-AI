# CafeSort-AI para Luckfox Pico Max (RV1106)

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

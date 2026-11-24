# CafeSort-AI: comparativa de modelos

Imágenes de test: 149

## Resumen de ejecución

| Modelo | ONNX (MB) | Latencia ONNX Runtime CPU | Exactitud global | F1 micro de detección |
|---|---:|---:|---:|---:|
| YOLO | 9.3984 | 20.7766 ms/fotograma | 0.3154 por fotograma | 0.7671 |
| MobileNetV2 | 8.5174 | 15.7743 ms/fotograma | 0.8342 por recorte | N/A |

La latencia se mide con `onnxruntime.InferenceSession` en CPU; excluye preprocesamiento y postprocesamiento.
El batch de MobileNetV2 es 1.

## YOLO: detección por clase

| Clase | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| maduro | 0.7809 | 0.7470 | 0.7635 | 310 | 87 | 105 |
| pinton | 0.7411 | 0.7947 | 0.7670 | 209 | 73 | 54 |
| verde | 0.7463 | 0.8658 | 0.8016 | 200 | 68 | 31 |
| negro | 0.6926 | 0.8204 | 0.7511 | 338 | 150 | 74 |

## YOLO: resultado por origen

| Origen | Imágenes | Exactitud de fotograma | Precision micro | Recall micro | F1 micro | Latencia media (ms/fotograma) |
|---|---:|---:|---:|---:|---:|---:|
| blanco | 100 | 0.4700 | 0.9048 | 0.9456 | 0.9248 | 20.6063 |
| rama | 49 | 0.0000 | 0.5317 | 0.6067 | 0.5667 | 21.1240 |

### YOLO: blanco

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| maduro | 0.8810 | 0.9331 | 0.9063 |
| pinton | 0.8033 | 0.9608 | 0.8750 |
| verde | 0.9353 | 1.0000 | 0.9665 |
| negro | 0.9612 | 0.9254 | 0.9430 |

### YOLO: rama

| Clase | Precision | Recall | F1 |
|---|---:|---:|---:|
| maduro | 0.5703 | 0.4534 | 0.5052 |
| pinton | 0.6937 | 0.6894 | 0.6916 |
| verde | 0.5426 | 0.6931 | 0.6087 |
| negro | 0.3913 | 0.6250 | 0.4813 |

## MobileNetV2: clasificación de recortes

Exactitud global: **0.8342** sobre 1321 granos.

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| maduro | 0.8292 | 0.8072 | 0.8181 | 415 |
| pinton | 0.8228 | 0.7414 | 0.7800 | 263 |
| verde | 0.9318 | 0.8874 | 0.9091 | 231 |
| negro | 0.7978 | 0.8908 | 0.8417 | 412 |

| Origen | Recortes | Exactitud | Latencia media (ms/fotograma) |
|---|---:|---:|---:|
| blanco | 754 | 0.9310 | 13.3407 |
| rama | 567 | 0.7055 | 21.0648 |

### MobileNetV2: blanco

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| maduro | 0.8843 | 0.9331 | 0.9080 | 254 |
| pinton | 0.9158 | 0.8529 | 0.8832 | 102 |
| verde | 0.9774 | 1.0000 | 0.9886 | 130 |
| negro | 0.9612 | 0.9254 | 0.9430 | 268 |

### MobileNetV2: rama

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| maduro | 0.7206 | 0.6087 | 0.6599 | 161 |
| pinton | 0.7606 | 0.6708 | 0.7129 | 161 |
| verde | 0.8621 | 0.7426 | 0.7979 | 101 |
| negro | 0.5891 | 0.8264 | 0.6879 | 144 |

## Alcance de la comparación

Las métricas de YOLO y MobileNetV2 no son directamente equivalentes: YOLO detecta y clasifica en la imagen completa; MobileNetV2 clasifica recortes construidos desde las cajas anotadas. Ambos se ejecutan con ONNX Runtime en CPU.

La latencia YOLO se mide por imagen con ONNX Runtime. La latencia MobileNetV2 se mide con ONNX Runtime sobre los recortes anotados; no incluye localizar ni recortar los granos.

Los resultados dependen de las anotaciones y del split `test` generado en `data/splits_manifest.json`.
Las etiquetas proceden de dataset/Segmentado; sus polígonos se convirtieron a cajas YOLO y se mapearon a las cuatro clases del proyecto. Los splits de este proyecto se reconstruyeron por escena y no se reutilizaron las particiones fuente. Las métricas reflejan la calidad de esas anotaciones y del mapeo.

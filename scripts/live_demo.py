from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import onnxruntime as ort

from src.benchmark_compare import (
    IMAGE_SIZE,
    MOBILENET_SIZE,
    _input_metadata,
    _load_session,
    _parse_yolo_output,
    _yolo_tensor,
)
from src.data_utils import CLASS_NAMES, DATA_DIR, RESULTS_DIR, read_image


COLORS = ((40, 190, 60), (20, 180, 240), (220, 150, 20), (40, 40, 220))


def _draw_panel(image: np.ndarray, boxes, scores, class_ids, heading: str, secondary=None) -> np.ndarray:
    canvas = image.copy()
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids, strict=True)):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = COLORS[int(class_id)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if secondary is None:
            label = f"{CLASS_NAMES[int(class_id)]} {score:.2f}"
        else:
            label = f"{CLASS_NAMES[int(class_id)]} {float(secondary[index]):.2f} | YOLO {score:.2f}"
        font_scale = max(0.45, image.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(y1, th + 8)
        cv2.rectangle(canvas, (x1, top - th - 8), (x1 + tw + 8, top), color, -1)
        cv2.putText(canvas, label, (x1 + 4, top - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (24, 30, 42), -1)
    cv2.putText(canvas, heading, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def _mobilenet_predict(session, input_name: str, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crops = []
    height, width = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width, int(x2)), min(height, int(y2))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            crop = np.zeros((MOBILENET_SIZE, MOBILENET_SIZE, 3), dtype=np.uint8)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (MOBILENET_SIZE, MOBILENET_SIZE), interpolation=cv2.INTER_AREA)
        crops.append(crop.transpose(2, 0, 1).astype(np.float32))
    if not crops:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    outputs = [session.run(None, {input_name: np.ascontiguousarray(crop[None, ...])})[0][0] for crop in crops]
    output = np.asarray(outputs)
    labels = np.argmax(output, axis=1)
    confidence = output[np.arange(len(labels)), labels]
    return labels, confidence.astype(np.float32)


def _select_records(limit: int) -> list[dict]:
    manifest_path = DATA_DIR / "splits_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "test"]
    groups = {origin: [r for r in records if r["origin"] == origin] for origin in ("blanco", "rama")}
    randomizer = random.Random(4060)
    selected: list[dict] = []
    per_origin = max(1, limit // 2)
    for origin in ("blanco", "rama"):
        pool = groups[origin]
        randomizer.shuffle(pool)
        selected.extend(pool[:per_origin])
    selected.sort(key=lambda record: (record["origin"], record["file"]))
    return selected[:limit]


def generate(limit: int = 60, conf: float = 0.25) -> Path:
    output_dir = RESULTS_DIR / "live_demo"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    yolo_path = RESULTS_DIR / "yolo" / "yolo_cafe.onnx"
    mobile_path = RESULTS_DIR / "mobilenet" / "mobilenetv2_cafe.onnx"
    yolo = _load_session(yolo_path)
    mobile = _load_session(mobile_path)
    yolo_input, yolo_hw = _input_metadata(yolo, (IMAGE_SIZE, IMAGE_SIZE))
    mobile_input, mobile_hw = _input_metadata(mobile, (MOBILENET_SIZE, MOBILENET_SIZE))
    if mobile_hw != (MOBILENET_SIZE, MOBILENET_SIZE):
        raise RuntimeError(f"MobileNet ONNX espera {mobile_hw}; se esperaba {(MOBILENET_SIZE, MOBILENET_SIZE)}.")

    records = _select_records(limit)
    frames = []
    for index, record in enumerate(records, start=1):
        image_path = DATA_DIR.parent / record["yolo_image"]
        image = read_image(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer el fotograma: {image_path}")
        start = time.perf_counter()
        tensor, ratio, pad_x, pad_y = _yolo_tensor(image, yolo_hw)
        raw = yolo.run(None, {yolo_input: tensor})
        yolo_ms = (time.perf_counter() - start) * 1000
        boxes, scores, ids = _parse_yolo_output(raw, ratio, pad_x, pad_y, image.shape[:2], conf)
        start = time.perf_counter()
        mobile_ids, mobile_scores = _mobilenet_predict(mobile, mobile_input, image, boxes)
        mobile_ms = (time.perf_counter() - start) * 1000

        left = _draw_panel(image, boxes, scores, ids, f"YOLO26n | {len(boxes)} detecciones | {yolo_ms:.0f} ms")
        right = _draw_panel(image, boxes, mobile_scores, mobile_ids,
                            f"MobileNetV2 sobre las mismas cajas | {len(boxes)} granos | {mobile_ms:.0f} ms",
                            secondary=scores)
        height = min(720, max(left.shape[0], right.shape[0]))
        scale = height / left.shape[0]
        width = max(1, int(left.shape[1] * scale))
        left = cv2.resize(left, (width, height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (width, height), interpolation=cv2.INTER_AREA)
        separator = np.full((height, 4, 3), (65, 72, 88), dtype=np.uint8)
        combined = np.hstack((left, separator, right))
        frame_name = f"{index:03d}.jpg"
        frame_path = frames_dir / frame_name
        if not cv2.imwrite(str(frame_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise RuntimeError(f"No se pudo escribir {frame_path}")
        frames.append({
            "file": f"frames/{frame_name}",
            "origin": record["origin"],
            "image": Path(record["file"]).name,
            "detections": int(len(boxes)),
            "yolo_ms": round(yolo_ms, 1),
            "mobilenet_ms": round(mobile_ms, 1),
            "reclassified": int(np.count_nonzero(ids != mobile_ids)),
        })
        print(f"[{index}/{len(records)}] {record['origin']} {Path(record['file']).name}: "
              f"{len(boxes)} cajas, YOLO {yolo_ms:.0f} ms, MobileNet {mobile_ms:.0f} ms")

    payload = {
        "frames": frames,
        "models": {
            "yolo": "YOLO26n ONNX",
            "mobilenet": "MobileNetV2 ONNX",
            "provider": "ONNX Runtime CPU",
            "confidence": conf,
        },
    }
    (output_dir / "frames.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(_html(), encoding="utf-8")
    print(f"Vista lista: {output_dir / 'index.html'}")
    return output_dir


def _html() -> str:
    return r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CafeSort-AI · Comparación en vivo</title><style>
:root{color-scheme:dark;--bg:#10141c;--panel:#1b2230;--muted:#aeb8c8;--accent:#58d6a5}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf2fa;font:15px system-ui,Segoe UI,sans-serif}.wrap{max-width:1500px;margin:auto;padding:20px}.top{display:flex;gap:20px;justify-content:space-between;align-items:center;flex-wrap:wrap}.title h1{margin:0;font-size:24px}.title p{margin:6px 0;color:var(--muted)}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat,.notice,.controls{background:var(--panel);border:1px solid #30394a;border-radius:12px;padding:12px 16px}.stat b{display:block;color:var(--accent);font-size:18px}.notice{margin:16px 0;color:#e9d8a6}.panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panels img{width:100%;height:auto;display:block;background:#080a0e;border-radius:12px;border:1px solid #343d4c}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:14px}.controls button,.controls select{background:#2a3444;color:white;border:1px solid #48566b;border-radius:8px;padding:9px 13px;font:inherit;cursor:pointer}.controls input{accent-color:var(--accent)}.meta{color:var(--muted);margin-left:auto}.foot{color:var(--muted);font-size:13px;margin-top:12px}@media(max-width:850px){.panels{grid-template-columns:1fr}.meta{margin-left:0}}
</style></head><body><main class="wrap"><header class="top"><div class="title"><h1>CafeSort-AI · Prueba de fotogramas</h1><p>Reproducción simulada con fotos reales del conjunto de prueba</p></div><div class="stats"><div class="stat">Fotograma<b id="counter">—</b></div><div class="stat">Detecciones YOLO<b id="detections">—</b></div><div class="stat">Etiquetas distintas<b id="changes">—</b></div></div></header>
<div class="notice"><b>Cómo leer la comparación:</b> YOLO detecta y clasifica. MobileNetV2 no localiza objetos; aquí clasifica los mismos recortes que detectó YOLO. El panel derecho prueba la cascada YOLO + MobileNet y puede cambiar etiquetas.</div>
<section class="panels"><img id="frame" alt="Fotograma comparativo YOLO y MobileNetV2"></section>
<section class="controls"><button id="toggle">⏸ Pausar</button><button id="prev">◀ Anterior</button><button id="next">Siguiente ▶</button><label>Velocidad <select id="speed"><option value="1500">0,7 fps</option><option value="800" selected>1,25 fps</option><option value="400">2,5 fps</option><option value="200">5 fps</option></select></label><label>Origen <select id="origin"><option value="todos">Todos</option><option value="blanco">Fondo blanco</option><option value="rama">Rama</option></select></label><input id="seek" type="range" min="0" value="0"><span id="meta" class="meta"></span></section>
<p class="foot">Inferencia local con ONNX Runtime CPU. Los fotogramas proceden del split test; no es una cámara conectada. La latencia medida incluye inferencia, pero no lectura de archivo ni postprocesamiento completo.</p></main>
<script>let data=[],view=[],i=0,timer=null;const $=s=>document.querySelector(s);async function init(){data=(await(await fetch('frames.json')).json()).frames;filter();play()}function filter(){const origin=$('#origin').value;view=data.map((f,n)=>({...f,n})).filter(f=>origin==='todos'||f.origin===origin);$('#seek').max=Math.max(0,view.length-1);i=Math.min(i,Math.max(0,view.length-1));show()}function show(){if(!view.length)return;const f=view[i];$('#frame').src=f.file;$('#counter').textContent=`${i+1} / ${view.length}`;$('#detections').textContent=f.detections;$('#changes').textContent=f.reclassified;$('#seek').value=i;$('#meta').textContent=`${f.origin==='blanco'?'Fondo blanco':'Rama'} · ${f.image} · YOLO ${f.yolo_ms} ms · MobileNet ${f.mobilenet_ms} ms`; }function step(n){i=(i+n+view.length)%view.length;show()}function play(){clearInterval(timer);timer=setInterval(()=>step(1),Number($('#speed').value));$('#toggle').textContent='⏸ Pausar'}function pause(){clearInterval(timer);timer=null;$('#toggle').textContent='▶ Reproducir'}$('#toggle').onclick=()=>timer?pause():play();$('#prev').onclick=()=>step(-1);$('#next').onclick=()=>step(1);$('#speed').onchange=()=>timer&&play();$('#origin').onchange=()=>{i=0;filter()};$('#seek').oninput=()=>{i=Number($('#seek').value);show()};init();</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara una simulación visual de YOLO26n y MobileNetV2 sobre el test.")
    parser.add_argument("--limit", type=int, default=60, help="Cantidad de fotogramas (máximo disponible: 149).")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit debe ser al menos 1.")
    generate(min(args.limit, 149), args.conf)


if __name__ == "__main__":
    main()

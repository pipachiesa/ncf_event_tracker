"""Diagnostico: por que el detector no encuentra la pelota.

Corre el modelo de pelota SOLO, sobre unos pocos frames del video, barriendo
imgsz y confianza. Aisla si el problema es el modelo/parametros o el resto del
pipeline (tracking, homografia, filtros).

Uso:
    python3 data_cleanup/debug_ball.py --video /ruta/video.mp4
    python3 data_cleanup/debug_ball.py --video v.mp4 --ball-model football --frames 12
    python3 data_cleanup/debug_ball.py --video v.mp4 --save-crops /tmp/ball_debug
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from main import (DEFAULT_BALL_MODEL, FALLBACK_BALL_MODEL, resolve_model_path,
                  resolve_ball_class_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--ball-model", default=DEFAULT_BALL_MODEL)
    p.add_argument("--frames", type=int, default=8,
                   help="Cuantos frames muestrear a lo largo del video.")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920],
                   help="Resoluciones de inferencia a probar.")
    p.add_argument("--conf", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.25],
                   help="Umbrales de confianza a probar.")
    p.add_argument("--save-crops", default=None,
                   help="Carpeta donde guardar los frames anotados.")
    args = p.parse_args()

    from ultralytics import YOLO

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"No puedo abrir el video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"VIDEO: {w}x{h} @ {fps:.3g} fps, {total} frames "
          f"({total / (fps or 25) / 60:.1f} min)")
    if w < 1280:
        print("  AVISO: video de baja resolucion. La pelota puede ocupar pocos "
              "pixeles y ser casi indetectable; subir --imgsz no inventa detalle.")

    resolved = resolve_model_path(args.ball_model, FALLBACK_BALL_MODEL)
    print(f"\nMODELO de pelota: '{args.ball_model}' -> {resolved}")
    model = YOLO(resolved)
    print(f"  clases: {model.names}")
    ball_id = resolve_ball_class_id(model)
    print(f"  class_id de pelota que usa el pipeline: {ball_id} "
          f"-> '{model.names.get(ball_id, '???')}'")
    if model.names.get(ball_id, "").lower() not in ("ball", "sports ball", "football"):
        print("  ⚠️  Ese nombre de clase NO parece una pelota. Si el class_id "
              "esta mal, el pipeline descarta todas las detecciones buenas.")

    # Frames repartidos a lo largo del partido (evitando los bordes).
    idxs = np.linspace(total * 0.1, total * 0.9, args.frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append((int(i), img))
    cap.release()
    print(f"\nMuestreados {len(frames)} frames.\n")

    lowest = min(args.conf)
    print(f"{'imgsz':>6} | " + " | ".join(f"conf>={c:<5}" for c in args.conf)
          + " |  mejor conf  | area mediana px")
    print("-" * 86)

    best_overall = (0.0, None)
    for imgsz in args.imgsz:
        counts = {c: 0 for c in args.conf}
        best = 0.0
        areas = []
        for fno, img in frames:
            r = model(img, conf=lowest, imgsz=imgsz, verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            keep = cls == ball_id
            if not keep.any():
                continue
            bc, bx = confs[keep], xyxy[keep]
            top = int(np.argmax(bc))
            best = max(best, float(bc[top]))
            areas.append(float((bx[top][2] - bx[top][0]) * (bx[top][3] - bx[top][1])))
            for c in args.conf:
                if (bc >= c).any():
                    counts[c] += 1
        med = f"{np.median(areas):.0f}" if areas else "-"
        row = " | ".join(f"{counts[c]:>2}/{len(frames):<7}" for c in args.conf)
        print(f"{imgsz:>6} | {row} |    {best:.3f}     | {med}")
        if best > best_overall[0]:
            best_overall = (best, imgsz)

    print()
    best_conf, best_imgsz = best_overall
    if best_conf == 0:
        print("DIAGNOSTICO: el modelo NO ve la pelota en ningun frame, a ninguna "
              "resolucion.\n"
              "  -> No es un problema de --ball-conf. Revisa: (a) que el class_id "
              "de arriba sea realmente la pelota; (b) que el modelo sea de futbol "
              "y no COCO; (c) si la camara es muy abierta, la pelota puede medir "
              "pocos pixeles.")
    elif best_conf < 0.1:
        print(f"DIAGNOSTICO: la ve, pero muy debil (max {best_conf:.3f} a imgsz "
              f"{best_imgsz}).\n  -> Baja --ball-conf por debajo de {best_conf:.2f} "
              f"y usa --imgsz {best_imgsz}.")
    else:
        print(f"DIAGNOSTICO: el modelo SI detecta la pelota (max {best_conf:.3f} a "
              f"imgsz {best_imgsz}).\n  -> El problema esta despues: parametros del "
              f"pipeline, class_id, o el filtrado/homografia. No es el detector.")

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
        for fno, img in frames:
            r = model(img, conf=lowest, imgsz=best_imgsz or 1280, verbose=False)[0]
            out = img.copy()
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(),
                                                  r.boxes.conf.cpu().numpy(), cls):
                    if k != ball_id:
                        continue
                    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 255), 2)
                    cv2.putText(out, f"{c:.2f}", (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.save_crops, f"frame_{fno}.jpg"), out)
        print(f"\nFrames anotados en: {args.save_crops}")


if __name__ == "__main__":
    main()

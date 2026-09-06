#!/usr/bin/env python3
"""Segunda pasada de deteccion de pelota con SAHI (slicing) SOLO en los huecos.

El detector full-frame pierde (o detecta con confianza inservible) la pelota
LEJANA porque, al reescalar el frame a imgsz, la pelota ocupa poquisimos pixeles.
SAHI parte el frame en tiles con solape y detecta en cada uno: la pelota lejana
ocupa mas pixeles y sube el recall. Medido en 24 frames GT (24-ago): las pelotas
a >60 m saltaron de conf 0,14-0,37 a 0,36-0,58.

Pero SAHI ensucia (mete falsos y duplica detecciones), asi que NO se usa como
reemplazo global: solo se corre en los frames donde el Viterbi actual NO tiene
pelota (los "huecos"). Asi:
  - no toca los frames que ya estaban bien (no hay regresiones),
  - los falsos que agrega los filtran igual: margen off-pitch + blacklist de
    celdas estaticas + continuidad del Viterbi (todo ya existe).

Proyeccion imagen->cancha: NO re-corre PnLCalib. Recupera la homografia por
frame de los JUGADORES del tracking CSV (su bottom-center en imagen y su
X_Pitch/Y_Pitch), igual que check_pitch_overlay.py. Asi los candidatos SAHI
quedan en EL MISMO sistema de coordenadas que los candidatos existentes.

Salida: un CSV de candidatos mergeado (original + los nuevos de SAHI), con el
mismo formato que main.py, listo para ball_viterbi.py.

Uso:
    python data_cleanup/sahi_huecos.py \\
        --video        ~/football_data/matches/clip-test/video.mp4 \\
        --tracking-csv ~/football_data/matches/clip-test/tracking.csv \\
        --output-candidates ~/football_data/matches/clip-test/tracking_ball_candidates_sahi.csv

Despues:
    python data_cleanup/ball_viterbi.py \\
        --tracking-csv ~/football_data/matches/clip-test/tracking.csv \\
        --candidates   ~/football_data/matches/clip-test/tracking_ball_candidates_sahi.csv \\
        --output       ~/football_data/matches/clip-test/tracking_vit_sahi.csv
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

# torch 2.6 pone weights_only=True por defecto y rompe el unpickle de los .pt de
# ultralytics; los checkpoints son de fuente confiable, restauramos el full-load.
import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ball_viterbi import (  # noqa: E402
    load_candidates, solve, read_fps, _offpitch, impossible_fraction,
    PITCH_L_M, PITCH_W_M, PITCH_L_CM, PITCH_W_CM,
)


def resolve_ball_model(spec):
    """'football' -> baja los pesos del Hub; si no, es una ruta a un .pt."""
    if spec == "football":
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="uisikdag/yolo-v8-football-players-detection",
            filename="best.pt")
    return os.path.expanduser(spec)


def ball_class_id(model):
    for cid, name in (model.names or {}).items():
        if str(name).strip().lower() in ("ball", "sports ball", "soccer ball", "football"):
            return int(cid)
    return 0


def read_meta_stride(tracking_csv):
    import json
    meta = tracking_csv.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        return int(json.load(open(meta)).get("frame_stride") or 1)
    print(f"AVISO: no encuentro {meta}; asumo frame_stride=1.")
    return 1


def homographies_from_players(tracking_csv):
    """{frame: H(imagen->cancha_cm)} recuperada del bottom-center de los jugadores."""
    by_frame = {}
    with open(tracking_csv) as fh:
        for r in csv.DictReader(fh):
            if r.get("Object") not in ("player", "goalkeeper"):
                continue
            try:
                f = int(r["Frame"])
                px, py = float(r["X_Pitch"]), float(r["Y_Pitch"])
                if px == 0 and py == 0:
                    continue
                bx = (float(r["X1"]) + float(r["X2"])) / 2.0
                by = float(r["Y2"])                       # pies
            except (KeyError, TypeError, ValueError):
                continue
            by_frame.setdefault(f, []).append((bx, by, px, py))
    H = {}
    for f, pts in by_frame.items():
        if len(pts) < 4:
            continue
        src = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        dst = np.array([[p[2], p[3]] for p in pts], dtype=np.float32)
        h, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if h is not None:
            H[f] = h
    return H


def make_slicer(model, ball_id, imgsz, conf):
    def cb(image_slice):
        r = model(image_slice, conf=conf, imgsz=imgsz, verbose=False)[0]
        d = sv.Detections.from_ultralytics(r)
        return d[d.class_id == ball_id]
    return lambda frame: sv.InferenceSlicer(
        callback=cb,
        slice_wh=(frame.shape[1] // 2 + 100, frame.shape[0] // 2 + 100),
        overlap_wh=(100, 100),
        iou_threshold=0.1,
    )(frame)


def project(h, x, y):
    p = h @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracking-csv", required=True)
    ap.add_argument("--candidates", default=None,
                    help="default: <tracking>_ball_candidates.csv")
    ap.add_argument("--output-candidates", required=True)
    ap.add_argument("--ball-model", default="football")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--ball-conf", type=float, default=0.10)
    ap.add_argument("--max-huecos", type=int, default=0,
                    help="0 = todos; >0 recorta para una prueba rapida")
    args = ap.parse_args()

    video = os.path.expanduser(args.video)
    tracking = os.path.expanduser(args.tracking_csv)
    cand_path = os.path.expanduser(
        args.candidates or tracking.rsplit(".", 1)[0] + "_ball_candidates.csv")
    out_path = os.path.expanduser(args.output_candidates)

    fps = read_fps(tracking)
    stride = read_meta_stride(tracking)

    # 1) candidatos actuales -> Viterbi -> que frames NO tienen pelota (huecos)
    by_frame = load_candidates(cand_path)
    path = solve(by_frame, fps)
    have_ball = set(path)
    frames_all = set()
    with open(tracking) as fh:
        for r in csv.DictReader(fh):
            try:
                frames_all.add(int(r["Frame"]))
            except (KeyError, ValueError):
                pass
    huecos = sorted(frames_all - have_ball)
    print(f"frames totales {len(frames_all)} | con pelota (Viterbi) {len(have_ball)} "
          f"| huecos {len(huecos)}")

    # 2) homografia por frame desde los jugadores
    H = homographies_from_players(tracking)
    print(f"homografias recuperadas: {len(H)} frames")

    # 3) SAHI en los huecos que tienen homografia
    model = YOLO(resolve_ball_model(args.ball_model))
    ball_id = ball_class_id(model)
    print(f"modelo pelota cargado, ball_class_id={ball_id}")
    slicer = make_slicer(model, ball_id, args.imgsz, args.ball_conf)

    cap = cv2.VideoCapture(video)
    huecos_con_h = [f for f in huecos if f in H]
    if args.max_huecos:
        huecos_con_h = huecos_con_h[:args.max_huecos]
    print(f"corriendo SAHI en {len(huecos_con_h)} huecos con homografia...")

    new_rows = []
    huecos_recuperados = 0
    for n, f in enumerate(huecos_con_h):
        if n % 50 == 0:
            print(f"  {n}/{len(huecos_con_h)}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f - 1) * stride)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = slicer(frame)
        if len(dets) == 0:
            continue
        h = H[f]
        added = False
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            conf = float(dets.confidence[i]) if dets.confidence is not None else 0.0
            bx, by = (x1 + x2) / 2.0, y2          # mismo anchor que main.py (pies)
            xc, yc = project(h, bx, by)           # cancha en cm
            x_m, y_m = xc / PITCH_L_CM * PITCH_L_M, yc / PITCH_W_CM * PITCH_W_M
            if _offpitch(x_m, y_m):
                continue
            new_rows.append((f, conf, x1, y1, x2, y2, xc, yc))
            added = True
        if added:
            huecos_recuperados += 1
    cap.release()

    print(f"\nhuecos con candidato SAHI en cancha: {huecos_recuperados}/{len(huecos_con_h)}")
    print(f"candidatos SAHI nuevos: {len(new_rows)}")

    # 4) merge: candidatos originales + los nuevos de SAHI
    orig = list(csv.DictReader(open(cand_path)))
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Conf", "X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"])
        for r in orig:
            w.writerow([r["Frame"], r["Conf"], r["X1"], r["Y1"], r["X2"],
                        r["Y2"], r["X_Pitch"], r["Y_Pitch"]])
        for (f, conf, x1, y1, x2, y2, xc, yc) in new_rows:
            w.writerow([f, f"{conf:.4f}", f"{x1:.1f}", f"{y1:.1f}",
                        f"{x2:.1f}", f"{y2:.1f}", f"{xc:.1f}", f"{yc:.1f}"])
    print(f"CSV mergeado -> {out_path}")

    # 5) antes/despues: cobertura del Viterbi e imposibles
    merged = load_candidates(out_path)
    path2 = solve(merged, fps)
    print("\n" + "=" * 56)
    print(f"COBERTURA del Viterbi (frames con pelota):")
    print(f"  antes  {len(have_ball):5d}")
    print(f"  despues{len(path2):5d}   (+{len(path2) - len(have_ball)})")
    print(f"movimientos imposibles (>35 m/s):")
    print(f"  antes  {impossible_fraction(path, fps):.2f}%")
    print(f"  despues{impossible_fraction(path2, fps):.2f}%")
    print("\nSiguiente: corre ball_viterbi.py con --candidates", out_path)
    print("y despues check_homography.py sobre el tracking resultante.")


if __name__ == "__main__":
    main()

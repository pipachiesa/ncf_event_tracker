#!/usr/bin/env python3
"""Re-proyecta un tracking existente con las homografias de PnLCalib.

No re-trackea: reusa las detecciones (coords de imagen X1..Y2 ya estan en el CSV)
y les aplica la homografia buena imagen->cancha(cm) por frame que produjo
`recalibrate_spain_france_colab.ipynb` (`spain-france_homographies.json`).

Re-proyecta el BOTTOM_CENTER de cada caja (los pies), igual que main.py. Solo
toca las filas dentro de los frames que tienen homografia (las 3 ventanas
etiquetadas); el resto queda igual (no lo usa el AUC).

Salida: tracking re-proyectado + candidatos de pelota re-proyectados. Despues:
  ball_viterbi.py sobre los candidatos nuevos -> escribe la pelota buena.
  train.py sobre el tracking resultante -> AUC.

Uso:
    python3 data_cleanup/reproject_from_homographies.py \\
        --tracking     events_model/dataset/spain-france/tracking.csv \\
        --candidates   events_model/dataset/spain-france/tracking_ball_candidates.csv \\
        --homographies ~/Downloads/spain-france_homographies.json \\
        --out-tracking   ~/football_data/matches/spain-france/tracking_recal.csv \\
        --out-candidates ~/football_data/matches/spain-france/tracking_recal_ball_candidates.csv
"""
import argparse
import bisect
import csv
import json
import os

import numpy as np

PITCH_L_CM, PITCH_W_CM = 10500.0, 6800.0
MAX_STALE = 6          # frames: si la H mas cercana esta a mas de esto, no re-proyectar


def load_homographies(path):
    """{frame:int -> H 3x3 imagen->cancha(cm)} (descarta los None)."""
    raw = json.load(open(os.path.expanduser(path)))
    H = {}
    for k, v in raw.items():
        if v is None:
            continue
        H[int(k)] = np.array(v, dtype=float).reshape(3, 3)
    return H


def nearest_H(H, frames_sorted, f):
    """La H del frame con homografia mas cercano a f, si esta a <= MAX_STALE."""
    i = bisect.bisect_left(frames_sorted, f)
    best, bestd = None, MAX_STALE + 1
    for j in (i - 1, i):
        if 0 <= j < len(frames_sorted):
            d = abs(frames_sorted[j] - f)
            if d < bestd:
                best, bestd = frames_sorted[j], d
    return H.get(best) if best is not None else None


def project(h, px, py):
    p = h @ np.array([px, py, 1.0])
    return p[0] / p[2], p[1] / p[2]


def reproject_csv(in_path, out_path, H, frames_sorted, is_candidates):
    """Reescribe X_Pitch/Y_Pitch de las filas que caen en un frame con H."""
    rows = list(csv.DictReader(open(os.path.expanduser(in_path))))
    hdr = rows[0].keys() if rows else []
    n_reproj = n_skip = 0
    for r in rows:
        try:
            f = int(r["Frame"])
        except (KeyError, ValueError):
            continue
        h = nearest_H(H, frames_sorted, f)
        if h is None:
            n_skip += 1
            continue
        x1, y1, x2, y2 = float(r["X1"]), float(r["Y1"]), float(r["X2"]), float(r["Y2"])
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue                       # fila "sin objeto"
        bx, by = (x1 + x2) / 2.0, y2       # pies (BOTTOM_CENTER), como main.py
        xc, yc = project(h, bx, by)
        r["X_Pitch"] = f"{xc:.1f}"
        r["Y_Pitch"] = f"{yc:.1f}"
        n_reproj += 1
    with open(os.path.expanduser(out_path), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hdr))
        w.writeheader()
        w.writerows(rows)
    kind = "candidatos" if is_candidates else "tracking"
    print(f"  {kind}: {n_reproj} filas re-proyectadas, {n_skip} sin H (fuera de ventanas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--homographies", required=True)
    ap.add_argument("--out-tracking", required=True)
    ap.add_argument("--out-candidates", required=True)
    args = ap.parse_args()

    H = load_homographies(args.homographies)
    frames_sorted = sorted(H)
    print(f"homografias: {len(H)} frames, rango {frames_sorted[0]}..{frames_sorted[-1]}")

    reproject_csv(args.tracking, args.out_tracking, H, frames_sorted, False)
    reproject_csv(args.candidates, args.out_candidates, H, frames_sorted, True)

    # sidecar de fps para que ball_viterbi/train lean el timeline correcto
    meta_in = args.tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(os.path.expanduser(meta_in)):
        meta_out = os.path.expanduser(args.out_tracking).rsplit(".", 1)[0] + ".meta.json"
        with open(os.path.expanduser(meta_in)) as fh:
            json.dump(json.load(fh), open(meta_out, "w"))
        print(f"  sidecar -> {meta_out}")

    print("\nSiguiente:")
    print(f"  python3 data_cleanup/ball_viterbi.py \\")
    print(f"    --tracking-csv {args.out_tracking} \\")
    print(f"    --candidates {args.out_candidates} \\")
    print(f"    --output {os.path.expanduser(args.out_tracking).replace('.csv','_vit.csv')}")
    print("  luego train.py con --tracking ese _vit.csv y los 3 --labels de siempre.")


if __name__ == "__main__":
    main()

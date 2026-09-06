#!/usr/bin/env python3
"""Evaluador reproducible de la PELOTA contra ground truth a mano.

La regla del proyecto: % de frames con la pelota del pipeline a <=N px de la
verdad (`ball_gt/*_ball_labels.csv`). Nada de proxies (pelota/jugadores en el
area, celdas). Un solo comando, local, segundos.

Reporta:
  - acc@20/50/100 px (post-Viterbi, sobre frames GT-visibles)
  - precision cuando el pipeline PONE una pelota, y FP en frames GT-invisibles
  - recall del DETECTOR (candidato a <=100px existe) = techo de la seleccion
  - brecha de seleccion: recall del detector - acc post-Viterbi
  - detecciones aisladas (un frame suelto, sin pelota real en vecinos +-2)
  - distribucion de la duracion de los huecos del camino
  - estratificado por tercio de la imagen (lejano/alto vs cercano/bajo)

Uso:
    python3 data_cleanup/eval_ball.py \\
        --tracking   ~/football_data/matches/clip-test-sahi/tracking_vit_fix.csv \\
        --candidates ~/football_data/matches/clip-test-sahi/tracking_ball_candidates.csv \\
        --labels     events_model/dataset/ball_gt/spain-france_ball_labels.csv
"""
import argparse
import csv
import math
import os


def load_labels(path):
    vis, invis = {}, set()
    for r in csv.DictReader(open(os.path.expanduser(path))):
        f = int(r["Frame"])
        if int(r["visible"]) == 1 and r["X_img"] != "":
            vis[f] = (float(r["X_img"]), float(r["Y_img"]))
        else:
            invis.add(f)
    return vis, invis


def load_ball(path):
    """{frame: (cx, cy)} del centro de la caja de la pelota del tracking."""
    d = {}
    for r in csv.DictReader(open(os.path.expanduser(path))):
        if r.get("Object") != "ball":
            continue
        x1, y1, x2, y2 = float(r["X1"]), float(r["Y1"]), float(r["X2"]), float(r["Y2"])
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        d[int(r["Frame"])] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return d


def load_candidates(path):
    d = {}
    for r in csv.DictReader(open(os.path.expanduser(path))):
        f = int(r["Frame"])
        d.setdefault(f, []).append(((float(r["X1"]) + float(r["X2"])) / 2,
                                    (float(r["Y1"]) + float(r["Y2"])) / 2))
    return d


def pct(sorted_vals, q):
    return sorted_vals[int(q * (len(sorted_vals) - 1))] if sorted_vals else 0


def isolation_status(frame, vis, invis, candidates, near=100.0):
    """Unknown neighbors cannot establish temporal isolation."""
    neighbors = [frame + d for d in (-2, -1, 1, 2)]
    for g in neighbors:
        if g in vis and any(math.dist(c, vis[g]) <= near
                            for c in candidates.get(g, [])):
            return "supported"
    if all(g in vis or g in invis for g in neighbors):
        return "isolated"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--near", type=float, default=100.0)
    args = ap.parse_args()

    vis, invis = load_labels(args.labels)
    ball = load_ball(args.tracking)
    cand = load_candidates(args.candidates)
    n = len(vis)
    if not n:
        raise SystemExit("No hay etiquetas visibles para evaluar exactitud")
    print(f"GT: {n} frames con pelota visible, {len(invis)} invisibles\n")

    # --- exactitud post-Viterbi sobre GT-visibles ---
    errs = []
    for f, (gx, gy) in vis.items():
        if f in ball:
            errs.append(math.hypot(ball[f][0] - gx, ball[f][1] - gy))
    within = lambda t: 100 * sum(1 for e in errs if e <= t) / n
    print("EXACTITUD (post-Viterbi, sobre GT-visibles)")
    print(f"  acc@20px  {within(20):5.1f}%")
    print(f"  acc@50px  {within(50):5.1f}%")
    print(f"  acc@100px {within(100):5.1f}%   <- la regla")
    print(f"  cobertura {len(errs)}/{n} frames con pelota puesta")

    # --- precision + FP en invisibles ---
    picks = [f for f in ball if f in vis]
    ok = sum(1 for f in picks if math.hypot(ball[f][0] - vis[f][0],
                                            ball[f][1] - vis[f][1]) <= args.near)
    prec = 100 * ok / len(picks) if picks else 0
    fp_invis = sum(1 for f in ball if f in invis)
    print(f"\nPRECISION cuando pone pelota: {prec:.0f}% ({ok}/{len(picks)})")
    print(f"  falsos en frames GT-INVISIBLES: {fp_invis}/{len(invis)} "
          f"({100*fp_invis/max(1,len(invis)):.0f}%)")

    # --- recall del detector (techo) y brecha de seleccion ---
    det = sum(1 for f, (gx, gy) in vis.items()
              if cand.get(f) and min(math.hypot(cx - gx, cy - gy)
                                     for cx, cy in cand[f]) <= args.near)
    print(f"\nDETECTOR (candidato a <=100px existe): {100*det/n:.0f}%   <- techo de la seleccion")
    print(f"  brecha de seleccion (detector - post-Viterbi): {100*det/n - within(100):.0f} pts")

    # --- detecciones aisladas entre las que el path perdio ---
    missing_real = [f for f, (gx, gy) in vis.items()
                    if f not in ball and cand.get(f) and
                    min(math.hypot(cx-gx, cy-gy) for cx, cy in cand[f]) <= args.near]
    statuses = [isolation_status(f, vis, invis, cand, args.near)
                for f in missing_real]
    print(f"\nPELOTA VISTA POR EL DETECTOR PERO PERDIDA POR EL PATH: {len(missing_real)}")
    print(f"  con apoyo GT en vecinos: {statuses.count('supported')}")
    print(f"  aisladas verificadas (4 vecinos anotados): {statuses.count('isolated')}")
    print(f"  indeterminadas (faltan etiquetas vecinas): {statuses.count('unknown')}")

    # --- huecos del camino ---
    fs = sorted(ball)
    gaps = [b - a - 1 for a, b in zip(fs, fs[1:]) if b - a > 1]
    gaps.sort()
    if gaps:
        print(f"\nHUECOS del camino (frames sin pelota entre detecciones):")
        print(f"  n={len(gaps)}  p50 {pct(gaps,.5)}  p95 {pct(gaps,.95)}  max {max(gaps)}")

    # --- estratificado por tercio de la imagen (y de la caja: alto=lejano) ---
    ys = sorted(gy for _f, (gx, gy) in vis.items())
    lo_y, hi_y = pct(ys, .33), pct(ys, .66)
    def acc_band(pred):
        e = [math.hypot(ball[f][0]-gx, ball[f][1]-gy)
             for f, (gx, gy) in vis.items() if pred(gy) and f in ball]
        tot = sum(1 for _f, (gx, gy) in vis.items() if pred(gy))
        return (100*sum(1 for x in e if x <= 100)/tot) if tot else 0, tot
    a_far, n_far = acc_band(lambda y: y <= lo_y)     # arriba en la imagen = lejano
    a_near, n_near = acc_band(lambda y: y >= hi_y)
    print(f"\nESTRATIFICADO por tercio de la imagen (acc@100px):")
    print(f"  tercio LEJANO (arriba, y<= {lo_y:.0f}px): {a_far:.0f}% ({n_far} frames)")
    print(f"  tercio CERCANO (abajo, y>={hi_y:.0f}px): {a_near:.0f}% ({n_near} frames)")


if __name__ == "__main__":
    main()

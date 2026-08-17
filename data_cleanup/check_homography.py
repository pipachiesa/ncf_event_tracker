"""Mide si el sistema de coordenadas de cancha es ESTABLE, usando los jugadores.

LA IDEA
Los veintipico de jugadores no se mueven todos juntos ni en la misma direccion.
Asi que el desplazamiento MEDIANO de todo el plantel entre dos frames
consecutivos deberia ser chico y parecido al movimiento real de una persona
(20-40 cm por frame a 15 fps). Cuando ese valor salta a varios metros, no se
movio el plantel: se movio la CANCHA debajo de ellos. Es una medida directa de
la calidad de la homografia que no necesita ground truth ni el video.

POR QUE IMPORTA MAS QUE CUALQUIER METRICA DE PELOTA
Todas las features del clasificador (distancias, velocidades, zonas) se calculan
en coordenadas de cancha. Si esas coordenadas saltan metros, ninguna limpieza
posterior de la pelota puede recuperar la informacion.

MEDIDO en spain-france (89.027 frames, homography_every=5): 6,36% de los frames
con salto global > 1 m y 1,67% con salto > 10 m, y el **93,6% de esos saltos cae
en frames multiplo de 5**, o sea exactamente cuando main.py re-estimaba la
transformacion. Con 4 keypoints (el minimo exacto) el error de reproyeccion es
cero por construccion, asi que era imposible detectar una mala solucion.

Uso:
    python3 data_cleanup/check_homography.py --tracking-csv <a.csv> [<b.csv> ...]
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

# Umbral de "esto no lo hizo un jugador". Un futbolista corriendo a 9 m/s a
# 15 fps recorre 60 cm por frame; un metro entero de TODO el plantel a la vez no
# tiene explicacion futbolistica.
JUMP_CM = 100.0
MIN_COMMON = 5      # ids compartidos minimos para que la mediana signifique algo


def read_fps(tracking, default=15.0):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    return default


def analyse(path):
    frames = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["Object"] == "ball":
                continue
            try:
                x, y = float(r["X_Pitch"]), float(r["Y_Pitch"])
            except (TypeError, ValueError):
                continue
            if x == 0 and y == 0:
                continue
            frames[int(r["Frame"])][r["Object ID"]] = (x, y)

    fs = sorted(frames)
    shifts = []
    for a, b in zip(fs, fs[1:]):
        if b - a != 1:
            continue
        common = frames[a].keys() & frames[b].keys()
        if len(common) < MIN_COMMON:
            continue
        d = sorted(math.hypot(frames[b][k][0] - frames[a][k][0],
                              frames[b][k][1] - frames[a][k][1])
                   for k in common)
        shifts.append((b, d[len(d) // 2]))
    return shifts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", nargs="+", required=True)
    p.add_argument("--every", type=int, default=5,
                   help="homography_every con el que se trackeo (para el test "
                        "de periodicidad)")
    args = p.parse_args()

    for path in args.tracking_csv:
        path = os.path.expanduser(path)
        fps = read_fps(path)
        shifts = analyse(path)
        if not shifts:
            print(f"\n=== {os.path.basename(path)} ===\n  sin datos suficientes")
            continue
        vals = sorted(s for _f, s in shifts)
        n = len(vals)

        def q(x):
            return vals[int(x * (n - 1))]

        big = [f for f, s in shifts if s > JUMP_CM]
        print(f"\n=== {os.path.basename(path)}  (fps={fps:g}, {n} frames) ===")
        print("  desplazamiento mediano del plantel entre frames (cm)")
        print(f"    p50 {q(.50):8.1f}   p90 {q(.90):8.1f}   "
              f"p99 {q(.99):8.1f}    <- p50 sano ~15-40 cm")
        for t in (100, 200, 500, 1000):
            c = sum(1 for v in vals if v > t)
            print(f"    frames con salto > {t/100:>4.0f} m: {c:>6}  "
                  f"({100.0*c/n:5.2f}%)")

        # Si los saltos se concentran en los frames de refresco, la causa es la
        # re-estimacion de la homografia y no el tracking de jugadores.
        if big and args.every > 1:
            hits = sum(1 for f in big if f % args.every == 1)
            frac = 100.0 * hits / len(big)
            print(f"    de esos saltos, {frac:.1f}% caen en frames de refresco "
                  f"(cada {args.every}); al azar seria {100.0/args.every:.1f}%")
            if frac > 2 * 100.0 / args.every:
                print("    -> LA CAUSA ES LA HOMOGRAFIA, no el tracking.")


if __name__ == "__main__":
    main()

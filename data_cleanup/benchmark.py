"""Puntua un tracking + sus eventos con las MISMAS metricas siempre.

Sirve para comparar dos implementaciones (o dos corridas) sin discutir de
opiniones: se corre sobre cada una y gana la que tenga mejores numeros.

Toda mejora de este proyecto se valido asi, y varias "mejoras" evidentes
resultaron neutras o peores al medirlas (ReID sobre los eventos, borrado de
pelotas quietas, la fisica de posesion estilo FIFA). Un test unitario dice que
el codigo corre; esto dice si los datos mejoraron.

METRICAS
  Trayectoria de la pelota
    imposibles %   movimientos > 35 m/s (un tiro potente llega a ~35).
                   Es la metrica raiz: con la trayectoria rota, toda la logica
                   de posesion de mas arriba mide ruido. Baseline 24.5%.
    presente %     frames con posicion de pelota (interpolada incluida).
    p90 / p99      velocidad implicita; una pelota real no pasa de ~35 m/s.
  Identidad
    ids            cantidad de identidades de jugador (menos = menos fragmentado)
    vida media     segundos que vive una identidad
  Eventos (si se pasa --events)
    conteos vs los valores tipicos de un partido real
    error total    suma de |detectado - esperado| normalizada

Uso:
    python3 data_cleanup/benchmark.py --tracking A.csv --events A_events.csv
    python3 data_cleanup/benchmark.py --tracking B.csv --events B_events.csv
"""

import argparse
import csv
import json
import math
import os
from collections import Counter

# Cancha reglamentaria; ver data_cleanup/pitch_config.py.
PITCH_L_CM, PITCH_W_CM = 10500.0, 6800.0
PITCH_L_M, PITCH_W_M = 105.0, 68.0
MAX_REAL_SPEED_MS = 35.0

# Valores tipicos de un partido de 90 minutos, para dimensionar el error.
EXPECTED = {
    "PASS": 1250, "BALL LOST": 150, "RECOVERY": 150,
    "SET PIECE": 55, "CHALLENGE": 100, "SHOT": 27,
}


def read_fps(tracking, default=24.0):
    meta = tracking.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    return default


def score_tracking(path, fps):
    balls, ids, id_frames = [], set(), Counter()
    total_ball_rows = 0
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["Object"] == "ball":
                total_ball_rows += 1
                x, y = float(row["X_Pitch"]), float(row["Y_Pitch"])
                if not (x == 0 and y == 0):
                    balls.append((int(row["Frame"]),
                                  x / PITCH_L_CM * PITCH_L_M,
                                  y / PITCH_W_CM * PITCH_W_M))
            else:
                ids.add(row["Object ID"])
                id_frames[row["Object ID"]] += 1
    balls.sort()

    speeds = []
    for (f0, x0, y0), (f1, x1, y1) in zip(balls, balls[1:]):
        if 1 <= f1 - f0 <= 3:
            speeds.append(math.hypot(x1 - x0, y1 - y0) / ((f1 - f0) / fps))
    speeds.sort()

    def pct(q):
        return speeds[int(q * (len(speeds) - 1))] if speeds else 0.0

    life = (sum(id_frames.values()) / len(ids) / fps) if ids else 0.0
    return {
        "presente_pct": 100.0 * len(balls) / max(1, total_ball_rows),
        "imposibles_pct": 100.0 * sum(1 for s in speeds if s > MAX_REAL_SPEED_MS)
                          / max(1, len(speeds)),
        "p90": pct(0.90), "p99": pct(0.99),
        "ids": len(ids), "vida_media_s": life,
    }


def score_events(path):
    counts = Counter(r["Type"] for r in csv.DictReader(open(path)))
    err = sum(abs(counts.get(k, 0) - v) / v for k, v in EXPECTED.items())
    return counts, 100.0 * err / len(EXPECTED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking", required=True)
    p.add_argument("--events", default=None)
    p.add_argument("--label", default=None, help="nombre para la salida")
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking)
    fps = read_fps(tracking)
    name = args.label or os.path.basename(tracking)
    print(f"\n=== {name}  (fps={fps:g}) ===")

    t = score_tracking(tracking, fps)
    print(f"\nPELOTA")
    print(f"  presente          {t['presente_pct']:6.1f}%")
    print(f"  IMPOSIBLES        {t['imposibles_pct']:6.1f}%   "
          f"<- metrica raiz (baseline 24.5%, objetivo <5%)")
    print(f"  velocidad p90/p99 {t['p90']:6.1f} / {t['p99']:.1f} m/s   "
          f"(real: nunca > 35)")
    print(f"\nIDENTIDAD")
    print(f"  ids               {t['ids']:6d}")
    print(f"  vida media        {t['vida_media_s']:6.1f} s")

    if args.events:
        counts, err = score_events(os.path.expanduser(args.events))
        print(f"\nEVENTOS")
        print(f"  {'tipo':<12} {'detectado':>10} {'esperado':>9} {'error':>8}")
        for k, v in EXPECTED.items():
            got = counts.get(k, 0)
            print(f"  {k:<12} {got:>10} {v:>9} {100.0*abs(got-v)/v:>7.0f}%")
        print(f"\n  ERROR MEDIO   {err:.0f}%   (menor = mejor)")

    print()


if __name__ == "__main__":
    main()

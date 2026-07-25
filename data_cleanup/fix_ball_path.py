"""Elimina saltos FISICAMENTE IMPOSIBLES en la trayectoria de la pelota.

DIAGNOSTICO (medido en spain-france, 77.260 saltos consecutivos):

    velocidad implicita     p50    2,7 m/s   <- normal
                            p75   31,6 m/s
                            p90  417,3 m/s   <- 1.502 km/h
                            p99  998,0 m/s   <- 3.593 km/h

    25% de los movimientos superan los 35 m/s (maximo fisico de un tiro).

La pelota no se mueve: SALTA entre candidatos. ``main.py`` guarda una sola
deteccion por frame (la de mayor confianza), asi que cuando el punto de penal,
una linea o algo del publico le gana en confianza a la pelota real, la posicion
"teletransporta" y vuelve al frame siguiente.

CONSECUENCIAS (por que esto es la causa raiz y no un detalle):
  * la posesion parpadea -- la pelota aparece junto a otro jugador por 1 frame,
    lo que genera pares BALL LOST + RECOVERY fantasma (~800/partido);
  * la fisica de posesion estilo FIFA (cambio de direccion/velocidad = toque)
    es INAPLICABLE: con la trayectoria saltando, el 77% de los intervalos
    "cambia de direccion" y el 87% "cambia de velocidad" por puro ruido.

ESTRATEGIA: un pico aislado se delata solo. Si desde el punto anterior hasta
el candidato hace falta una velocidad imposible, y desde el candidato al
siguiente tambien, PERO el salto directo anterior->siguiente es plausible,
entonces el candidato es un falso positivo: la pelota real nunca se fue de ahi.
Se pone en cero (= "no detectada", lo que el pipeline ya sabe manejar e
interpolar) en vez de inventar una posicion.

A diferencia de ``clean_ball.py`` (que borraba pelotas quietas y EMPEORO los
eventos porque esas eran la pelota real en pausas), aca solo se borra lo que
viola la fisica, que no puede ser la pelota real.

Uso:
    python3 data_cleanup/fix_ball_path.py \\
        --tracking-csv ~/football_data/matches/<m>/tracking.csv \\
        --output       ~/football_data/matches/<m>/tracking_fixed.csv
"""

import argparse
import csv
import json
import os
import sys

PITCH_L_CM, PITCH_W_CM = 12000.0, 7000.0
PITCH_L_M, PITCH_W_M = 105.0, 68.0
BALL_COLS = ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch")

# Un tiro potente ronda los 35 m/s. Por encima de MAX_SPEED_MS no hay pelota
# posible; se usa con holgura para no castigar el ruido normal de tracking.
MAX_SPEED_MS = 40.0


def read_fps(tracking_csv, default=24.0):
    meta = tracking_csv.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    print(f"AVISO: no encuentro {meta}; asumo {default:g} fps.")
    return default


def to_metres(x_cm, y_cm):
    return (float(x_cm) / PITCH_L_CM * PITCH_L_M,
            float(y_cm) / PITCH_W_CM * PITCH_W_M)


def speed(a, b, fps):
    """m/s implicita entre dos puntos (frame, x_m, y_m)."""
    df = b[0] - a[0]
    if df <= 0:
        return float("inf")
    dist = ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5
    return dist / (df / fps)


def find_outliers(points, fps, max_speed=MAX_SPEED_MS, max_passes=4):
    """Indices de detecciones que violan la fisica como pico aislado."""
    alive = [True] * len(points)
    dropped = set()

    for _ in range(max_passes):
        found = 0
        idx = [i for i in range(len(points)) if alive[i]]
        for pos in range(1, len(idx) - 1):
            i = idx[pos]
            prev, cand, nxt = points[idx[pos - 1]], points[i], points[idx[pos + 1]]
            v_in = speed(prev, cand, fps)
            v_out = speed(cand, nxt, fps)
            if v_in <= max_speed or v_out <= max_speed:
                continue
            # Ida Y vuelta imposibles. Si saltear el candidato deja un tramo
            # plausible, la pelota real siguio su camino y el candidato es
            # basura (punto de penal, linea, publico).
            if speed(prev, nxt, fps) <= max_speed:
                alive[i] = False
                dropped.add(i)
                found += 1
        if not found:
            break
    return dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-speed", type=float, default=MAX_SPEED_MS,
                   help="m/s por encima de los cuales el movimiento es imposible.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking_csv)
    fps = read_fps(tracking)

    rows, points, row_of = [], [], []
    with open(tracking) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
            if row["Object"] != "ball":
                continue
            try:
                xp, yp = float(row["X_Pitch"]), float(row["Y_Pitch"])
                x1, y1 = float(row["X1"]), float(row["Y1"])
            except (TypeError, ValueError):
                continue
            if (x1 == 0 and y1 == 0) or (xp == 0 and yp == 0):
                continue
            xm, ym = to_metres(xp, yp)
            points.append((int(row["Frame"]), xm, ym))
            row_of.append(len(rows) - 1)

    if len(points) < 3:
        sys.exit("Muy pocas detecciones de pelota.")
    print(f"{len(rows)} filas, {len(points)} detecciones de pelota, fps={fps:g}")

    def bad_fraction(pts):
        bad = tot = 0
        for i in range(1, len(pts)):
            if pts[i][0] - pts[i - 1][0] > 3:
                continue
            tot += 1
            if speed(pts[i - 1], pts[i], fps) > args.max_speed:
                bad += 1
        return 100.0 * bad / max(1, tot)

    before = bad_fraction(points)
    outliers = find_outliers(points, fps, args.max_speed)
    after = bad_fraction([p for i, p in enumerate(points) if i not in outliers])

    print(f"\nsaltos imposibles (>{args.max_speed:g} m/s): "
          f"{before:.1f}% -> {after:.1f}%")
    print(f"detecciones eliminadas: {len(outliers)} "
          f"({100 * len(outliers) / len(points):.1f}%)")
    print(f"quedan: {len(points) - len(outliers)} detecciones limpias")

    if args.dry_run:
        print("\n--dry-run: no se escribio nada.")
        return

    for i in outliers:
        row = rows[row_of[i]]
        for col in BALL_COLS:
            row[col] = "0"

    out = os.path.expanduser(args.output)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV corregido: {out}")

    src = tracking.rsplit(".", 1)[0] + ".meta.json"
    dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(src) and not os.path.exists(dst):
        with open(src) as a, open(dst, "w") as b:
            b.write(a.read())
        print(f"Sidecar copiado: {dst}")


if __name__ == "__main__":
    main()

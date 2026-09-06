"""Regenerar eventos desde un tracking.csv ya existente (CPU, sin GPU ni YOLO).

Sirve para reprocesar eventos despues de tocar las reglas, sin volver a trackear
(que son horas de GPU). Usa la convencion de rutas de events_model/matchpaths.py:

    ~/football_data/matches/<match>/tracking.csv
    ~/football_data/matches/<match>/tracking.meta.json   <- fps efectivo

Uso:
    python3 data_cleanup/regen_events.py --match spain-france
    python3 data_cleanup/regen_events.py --tracking /ruta/al/tracking.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.match import Match


def _interpolate_ball_csv(tracking, gap_max, meta_src):
    """Escribe un CSV vecino con los huecos de balon <= ``gap_max`` rellenados.

    La interpolacion lineal produce trayectorias SUAVES, asi que sube la
    cobertura del balon sin reintroducir los saltos imposibles que arruinaban
    las posesiones. Se escribe a un archivo aparte (``*_interp.csv``) para no
    pisar el tracking original.
    """
    import csv

    # Se interpolan TODAS las coordenadas, incluido el bounding box en pixeles.
    # No alcanza con X_Pitch/Y_Pitch: el label_tool dibuja el marcador de la
    # pelota a partir del bbox, asi que dejarlo en un valor nominal ponia la
    # pelota interpolada en el pixel (1,1) -- la esquina de la pantalla -- y
    # generaba saltos enormes contra los frames reales.
    KEYS = ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch")

    rows, ball = [], {}
    with open(tracking) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
            if row["Object"] != "ball":
                continue
            x, y = float(row["X_Pitch"]), float(row["Y_Pitch"])
            vals = None if (x == 0 and y == 0) else \
                {k: float(row[k]) for k in KEYS}
            ball[int(row["Frame"])] = (len(rows) - 1, vals)

    seen = sorted(f for f, (_i, p) in ball.items() if p)
    filled = 0
    for a, b in zip(seen, seen[1:]):
        gap = b - a - 1
        if not 1 <= gap <= gap_max:
            continue
        va, vb = ball[a][1], ball[b][1]
        for k in range(1, gap + 1):
            frame = a + k
            if frame not in ball:
                continue
            t = k / (gap + 1)
            row = rows[ball[frame][0]]
            for key in KEYS:
                row[key] = f"{va[key] + (vb[key] - va[key]) * t:.2f}"
            filled += 1

    out = tracking.rsplit(".", 1)[0] + "_interp.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    cov = 100.0 * (len(seen) + filled) / max(1, len(ball))
    print(f"Interpolados {filled} frames (huecos <= {gap_max}); "
          f"balon presente {100.0*len(seen)/max(1,len(ball)):.1f}% -> {cov:.1f}%")

    dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta_src) and not os.path.exists(dst):
        with open(meta_src) as a, open(dst, "w") as b:
            b.write(a.read())
    return os.path.dirname(out) + os.sep, os.path.basename(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", help="Nombre del partido en ~/football_data/matches/")
    parser.add_argument("--tracking", help="Ruta explicita al tracking CSV.")
    parser.add_argument("--out", help="Donde escribir el CSV de eventos.")
    parser.add_argument("--long-blank-seconds", type=float, default=None,
                        help="Segundos de balon continuamente ausente para "
                             "considerar que salio de juego (sube = menos set "
                             "pieces fantasma).")
    parser.add_argument("--interp-gap", type=int, default=0,
                        help="Rellena linealmente los huecos de balon de hasta N "
                             "frames antes de generar eventos. La interpolacion "
                             "es SUAVE, asi que recupera cobertura sin "
                             "reintroducir saltos imposibles. Medido en "
                             "spain-france: 25 lleva el balon de 74.4%% a 90.1%% "
                             "y los pases de 478 a 594, con los set pieces "
                             "igual o mejor (131 -> 125). 0 = no tocar.")
    args = parser.parse_args()

    if args.tracking:
        tracking = os.path.abspath(args.tracking)
    elif args.match:
        tracking = os.path.expanduser(
            f"~/football_data/matches/{args.match}/tracking.csv")
    else:
        parser.error("pasa --match o --tracking")

    if not os.path.exists(tracking):
        sys.exit(f"No existe el tracking CSV: {tracking}")

    directory = os.path.dirname(tracking) + os.sep
    file_name = os.path.basename(tracking)

    # El sidecar tiene que compartir el nombre base del CSV: tracking.csv ->
    # tracking.meta.json. Si falta, Match asume 24 fps y TODOS los umbrales del
    # generador (que estan en frames) cambian de significado en segundos.
    meta = os.path.join(directory, file_name.rsplit(".", 1)[0] + ".meta.json")
    if not os.path.exists(meta):
        print(f"AVISO: no encuentro {meta} -- se asumiran 24 fps. Si el tracking "
              f"se corrio con --frame-stride, los tiempos y umbrales van a estar mal.")

    if args.interp_gap > 0:
        directory, file_name = _interpolate_ball_csv(
            tracking, args.interp_gap, meta)

    match = Match()
    match.import_raw_data(directory, file_name)
    print(f"Importados {match.frames} frames, {len(match.players)} objetos, "
          f"fps={match.fps}")

    events = match.generate_events(long_blank_seconds=args.long_blank_seconds)
    print("Resumen:", events.summary())

    out = args.out or os.path.join(directory, "events.csv")
    events.export(path=os.path.dirname(out) + os.sep,
                  file_name=os.path.basename(out))
    print("Eventos escritos en:", out)


if __name__ == "__main__":
    main()

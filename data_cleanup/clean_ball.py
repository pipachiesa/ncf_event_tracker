"""Elimina detecciones de pelota que en realidad son marcas fijas de la cancha.

El detector confunde con la pelota a objetos blancos y chicos que NO se mueven:
el punto de penal, marcas del cesped, carteles, manchas del cartel LED. Son
falsos positivos con una firma inconfundible: quedan inmoviles al pixel exacto
durante segundos o minutos, algo que una pelota en juego nunca hace.

Medido sobre spain-france (99 min, camara fija): 25,9% de las detecciones de
pelota estaban en rachas inmoviles de 1s o mas, incluida una de 37,6 segundos
clavada en el mismo pixel.

Dos criterios, cualquiera alcanza para marcar una racha como artefacto:

1. DEMASIADO LARGA: una racha inmovil de mas de ``--max-still-seconds``. Una
   pelota parada antes de un corner o un tiro libre puede estar quieta unos
   segundos, pero no 15.
2. RECURRENTE: la misma posicion (+-``--recur-px``) aparece quieta en momentos
   del partido separados por mas de ``--recur-minutes``. Una marca de la cancha
   esta siempre ahi; una pelota detenida antes de un saque, una sola vez.

Las detecciones marcadas se ponen en cero, que es como el pipeline representa
"no se detecto la pelota" -- no se inventa nada, simplemente se borra lo falso.

Uso:
    python3 data_cleanup/clean_ball.py \\
        --tracking-csv ~/football_data/matches/<m>/tracking.csv \\
        --output       ~/football_data/matches/<m>/tracking_clean.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

BALL_COLS = ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch")


def read_fps(tracking_csv, default=24.0):
    meta = tracking_csv.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            return float(json.load(fh).get("effective_fps") or default)
    print(f"AVISO: no encuentro {meta}; asumo {default:g} fps.")
    return default


def find_still_runs(dets, still_px, min_run):
    """Rachas consecutivas donde el centro se mueve menos de ``still_px``."""
    runs = []
    if not dets:
        return runs
    start = 0
    for i in range(1, len(dets)):
        (f0, x0, y0), (f1, x1, y1) = dets[i - 1], dets[i]
        # Un hueco de mas de 2 frames corta la racha (no es continuidad real).
        if f1 - f0 <= 2 and abs(x1 - x0) < still_px and abs(y1 - y0) < still_px:
            continue
        if i - start >= min_run:
            runs.append((start, i - 1))
        start = i
    if len(dets) - start >= min_run:
        runs.append((start, len(dets) - 1))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--still-px", type=float, default=2.0,
                   help="Movimiento por debajo del cual se considera inmovil.")
    p.add_argument("--min-still-seconds", type=float, default=1.0,
                   help="Duracion minima para considerar una racha.")
    p.add_argument("--max-still-seconds", type=float, default=8.0,
                   help="Racha inmovil mas larga que esto = artefacto seguro.")
    p.add_argument("--recur-px", type=float, default=12.0,
                   help="Tolerancia para decir que dos rachas estan en el mismo lugar.")
    p.add_argument("--recur-minutes", type=float, default=5.0,
                   help="Separacion temporal entre rachas en el mismo lugar "
                        "que delata una marca fija.")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo reportar, no escribir el CSV.")
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking_csv)
    fps = read_fps(tracking)
    min_run = max(2, int(args.min_still_seconds * fps))
    max_run = int(args.max_still_seconds * fps)
    recur_frames = args.recur_minutes * 60 * fps

    rows = []
    dets = []          # (frame, cx, cy) solo de pelota detectada
    det_row_idx = []   # indice en rows de cada deteccion
    with open(tracking) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
            if row["Object"] != "ball":
                continue
            try:
                x1, y1 = float(row["X1"]), float(row["Y1"])
                x2, y2 = float(row["X2"]), float(row["Y2"])
            except (TypeError, ValueError):
                continue
            if x1 == 0 and y1 == 0:
                continue
            dets.append((int(row["Frame"]), (x1 + x2) / 2, (y1 + y2) / 2))
            det_row_idx.append(len(rows) - 1)

    if not dets:
        sys.exit("No hay detecciones de pelota en el CSV.")
    print(f"{len(rows)} filas, {len(dets)} detecciones de pelota, fps={fps:g}")

    runs = find_still_runs(dets, args.still_px, min_run)
    print(f"Rachas inmoviles de >={args.min_still_seconds:g}s: {len(runs)}")

    # Criterio 2: agrupar rachas por posicion para detectar recurrencia.
    by_pos = defaultdict(list)
    for a, b in runs:
        cx, cy = dets[a][1], dets[a][2]
        by_pos[(round(cx / args.recur_px), round(cy / args.recur_px))].append((a, b))

    drop_runs, reasons = [], {}
    for pos, group in by_pos.items():
        span = dets[group[-1][1]][0] - dets[group[0][0]][0]
        recurrent = len(group) > 1 and span > recur_frames
        for a, b in group:
            length = b - a + 1
            if recurrent:
                drop_runs.append((a, b)); reasons[(a, b)] = "recurrente"
            elif length > max_run:
                drop_runs.append((a, b)); reasons[(a, b)] = "demasiado larga"

    drop_idx = set()
    for a, b in drop_runs:
        drop_idx.update(range(a, b + 1))

    print(f"Rachas marcadas como artefacto: {len(drop_runs)} "
          f"({sum(1 for r in reasons.values() if r == 'recurrente')} recurrentes, "
          f"{sum(1 for r in reasons.values() if r == 'demasiado larga')} demasiado largas)")

    worst = sorted(drop_runs, key=lambda ab: ab[0] - ab[1])[:8]
    if worst:
        print(f"\n{'seg':>6} {'min partido':>12} {'pixel':>14}  motivo")
        print("-" * 52)
        for a, b in worst:
            fr, cx, cy = dets[a]
            print(f"{(b - a + 1) / fps:>6.1f} {fr / fps / 60:>12.1f} "
                  f"{str((int(cx), int(cy))):>14}  {reasons[(a, b)]}")

    kept = len(dets) - len(drop_idx)
    print(f"\nDetecciones de pelota: {len(dets)} -> {kept} "
          f"({100 * len(drop_idx) / len(dets):.1f}% eliminadas)")

    if args.dry_run:
        print("\n--dry-run: no se escribio nada.")
        return

    for i in drop_idx:
        row = rows[det_row_idx[i]]
        for col in BALL_COLS:
            row[col] = "0"

    out = os.path.expanduser(args.output)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV limpio: {out}")

    meta_src = tracking.rsplit(".", 1)[0] + ".meta.json"
    meta_dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(meta_src) and not os.path.exists(meta_dst):
        with open(meta_src) as a, open(meta_dst, "w") as b:
            b.write(a.read())
        print(f"Sidecar copiado: {meta_dst}")


if __name__ == "__main__":
    main()

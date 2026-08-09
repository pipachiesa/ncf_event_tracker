"""Descarta detecciones que estan FUERA de la cancha: suplentes calentando,
cuerpo tecnico, alcanzapelotas y las pelotas de calentamiento del costado.

MEDIDO en spain-france (99 min):
  * 8.3% de las detecciones de jugador caen fuera o pegadas al borde lateral;
  * 2.76% de las detecciones de pelota estan fuera de los limites del campo.

Por que importa mas de lo que parece: ``main.py`` no descarta lo que queda
fuera, lo CLAMPEA dentro de una banda (PLAYER_PITCH_MARGIN). O sea que un
suplente calentando en la banda entra al CSV como si estuviera jugando pegado
a la linea, y puede quedar como "jugador mas cercano a la pelota". Peor: una
pelota de calentamiento fuera del campo puede ser elegida como LA pelota del
partido y arrastrar toda la deteccion de posesion.

Se descartan filas, no se corrigen: una posicion fuera del campo no tiene
arreglo posible, y para el pipeline "no detectado" es un estado valido y
manejado (la interpolacion puentea los huecos cortos).

Uso:
    python3 data_cleanup/clean_offpitch.py \\
        --tracking-csv ~/football_data/matches/<m>/tracking.csv \\
        --output       ~/football_data/matches/<m>/tracking_onpitch.csv
"""

import argparse
import csv
import os

PITCH_L_CM, PITCH_W_CM = 12000.0, 7000.0

# Tolerancia (fraccion del ancho/largo) antes de considerar que algo esta
# realmente afuera. La homografia tiene ruido y un jugador sobre la linea
# puede proyectarse un poco afuera, asi que el corte no va en el borde exacto.
PLAYER_MARGIN = 0.02
# La pelota SI puede salir del campo legitimamente (saque lateral, corner), asi
# que se le da mas aire; lo que se busca eliminar son las pelotas del costado
# que estan permanentemente fuera.
BALL_MARGIN = 0.06


def outside(x, y, margin):
    return (x < -margin * PITCH_L_CM or x > (1 + margin) * PITCH_L_CM or
            y < -margin * PITCH_W_CM or y > (1 + margin) * PITCH_W_CM)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--player-margin", type=float, default=PLAYER_MARGIN)
    p.add_argument("--ball-margin", type=float, default=BALL_MARGIN)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tracking = os.path.expanduser(args.tracking_csv)
    kept, dropped_p, dropped_b, ball_blanked = [], 0, 0, 0

    with open(tracking) as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        for row in reader:
            try:
                x, y = float(row["X_Pitch"]), float(row["Y_Pitch"])
            except (TypeError, ValueError):
                kept.append(row)
                continue

            if x == 0 and y == 0:
                kept.append(row)
                continue

            if row["Object"] == "ball":
                if outside(x, y, args.ball_margin):
                    # La fila de pelota existe una vez por frame: se vacia en
                    # lugar de borrarse, que es como el pipeline representa
                    # "no se vio la pelota".
                    for c in ("X1", "Y1", "X2", "Y2", "X_Pitch", "Y_Pitch"):
                        row[c] = "0"
                    ball_blanked += 1
                kept.append(row)
            else:
                if outside(x, y, args.player_margin):
                    dropped_p += 1
                    continue
                kept.append(row)

    n_players = sum(1 for r in kept if r["Object"] != "ball") + dropped_p
    print(f"jugadores descartados (fuera de cancha): {dropped_p} "
          f"({100.0 * dropped_p / max(1, n_players):.1f}%)")
    print(f"pelotas anuladas (fuera de cancha)     : {ball_blanked}")

    if args.dry_run:
        print("\n--dry-run: no se escribio nada.")
        return

    out = os.path.expanduser(args.output)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(f"\nCSV: {out}")

    src = tracking.rsplit(".", 1)[0] + ".meta.json"
    dst = out.rsplit(".", 1)[0] + ".meta.json"
    if os.path.exists(src) and not os.path.exists(dst):
        with open(src) as a, open(dst, "w") as b:
            b.write(a.read())
        print(f"Sidecar: {dst}")


if __name__ == "__main__":
    main()

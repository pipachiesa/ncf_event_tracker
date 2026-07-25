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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", help="Nombre del partido en ~/football_data/matches/")
    parser.add_argument("--tracking", help="Ruta explicita al tracking CSV.")
    parser.add_argument("--out", help="Donde escribir el CSV de eventos.")
    parser.add_argument("--long-blank-seconds", type=float, default=None,
                        help="Segundos de balon continuamente ausente para "
                             "considerar que salio de juego (sube = menos set "
                             "pieces fantasma).")
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

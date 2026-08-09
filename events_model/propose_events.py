"""
Genera el CSV de eventos propuestos por las reglas para un tracking CSV.
Es el paso previo al etiquetado: estas propuestas son las que se corrigen
en label_tool.py.

Uso (convencion por partido, resuelve las rutas solo):
    python3 events_model/propose_events.py --match psg_bayern

Uso (rutas explicitas):
    python3 events_model/propose_events.py \
        --tracking "~/football_data/matches/psg_bayern/tracking.csv" \
        --out events_model/dataset/psg_bayern_proposed.csv
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_cleanup"))

import matchpaths  # noqa: E402
from lib.match import Match  # noqa: E402
from regen_events import _interpolate_ball_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Eventos por reglas -> CSV Metrica")
    ap.add_argument("--match", default=None,
                    help="nombre del partido (convencion "
                         "~/football_data/matches/<match>/tracking.csv)")
    ap.add_argument("--tracking", default=None,
                    help="CSV crudo de tracking (override de --match)")
    ap.add_argument("--out", default=None,
                    help="CSV de propuestas de salida (override de --match)")
    ap.add_argument("--block-minutes", type=float, default=None,
                    help="Recorta un bloque de N minutos en vez del partido "
                         "entero. Etiquetar bloques cortos de tramos DISTINTOS "
                         "da mas variedad de contextos que etiquetar todo "
                         "seguido desde el minuto 0.")
    ap.add_argument("--start-minute", type=float, default=None,
                    help="Minuto de inicio del bloque (default: al azar).")
    ap.add_argument("--exclude", default="",
                    help="Tramos ya etiquetados a evitar al sortear, en "
                         "minutos: '0-8,10-18'.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla para que el sorteo sea reproducible.")
    ap.add_argument("--interp-gap", type=int, default=25,
                    help="Rellena huecos de balon de hasta N frames antes de "
                         "proponer. La interpolacion lineal es SUAVE: sube la "
                         "cobertura sin reintroducir saltos imposibles, y mas "
                         "cobertura = mas momentos reales propuestos para "
                         "etiquetar (medido: 370 -> 568 pases). 0 = no tocar.")
    args = ap.parse_args()

    if not args.match and not (args.tracking and args.out):
        ap.error("pasa --match <partido> o --tracking y --out")

    tracking = os.path.expanduser(
        args.tracking or matchpaths.tracking_path(args.match))
    out = os.path.expanduser(args.out or matchpaths.proposed_path(args.match))

    if not os.path.exists(tracking):
        ap.error(f"no existe el tracking: {tracking}")

    directory = os.path.dirname(tracking) + os.sep
    file_name = os.path.basename(tracking)
    if args.interp_gap > 0:
        meta = os.path.join(directory, file_name.rsplit(".", 1)[0] + ".meta.json")
        directory, file_name = _interpolate_ball_csv(
            tracking, args.interp_gap, meta)

    match = Match()
    match.import_raw_data(directory, file_name)
    log = match.generate_events()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    path = log.export(path=os.path.dirname(out) or ".",
                      file_name=os.path.basename(out))
    print(f"{len(log)} eventos {dict(log.summary())} -> {path}")

    if args.block_minutes:
        _write_block(path, match, args, out)


def _write_block(path, match, args, out):
    """Recorta el CSV de propuestas a un bloque temporal y avisa como etiquetarlo."""
    total_s = match.frames / match.fps
    length = args.block_minutes * 60.0

    if args.start_minute is not None:
        start = args.start_minute * 60.0
    else:
        busy = []
        for chunk in filter(None, args.exclude.split(",")):
            a, b = chunk.split("-")
            busy.append((float(a) * 60.0, float(b) * 60.0))
        rng = random.Random(args.seed)
        # Se sortea entre los arranques que no pisan lo ya etiquetado.
        options = [t for t in range(0, int(total_s - length), 30)
                   if not any(t < b and t + length > a for a, b in busy)]
        if not options:
            sys.exit("No queda tramo libre: agranda el partido o baja --exclude.")
        start = float(rng.choice(options))

    end = start + length
    rows = [r for r in csv.DictReader(open(path))
            if start <= float(r["Start Time [s]"]) <= end]
    if not rows:
        sys.exit(f"El bloque {start/60:.0f}-{end/60:.0f} min no tiene propuestas.")

    tag = f"{int(start//60)}m_{int(end//60)}m"
    block = out.replace("_proposed.csv", f"_{tag}_proposed.csv")
    with open(block, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    types = {}
    for r in rows:
        types[r["Type"]] = types.get(r["Type"], 0) + 1
    print(f"\nBLOQUE {start/60:.0f}-{end/60:.0f} min: {len(rows)} propuestas {types}")
    print(f"  {block}")
    print(f"\nPara etiquetarlo:\n"
          f"  python3 events_model/label_tool.py \\\n"
          f"    --video {os.path.expanduser(matchpaths.video_path(args.match))} \\\n"
          f"    --events {block} \\\n"
          f"    --tracking {os.path.expanduser(matchpaths.tracking_path(args.match))}")


if __name__ == "__main__":
    main()

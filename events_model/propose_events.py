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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_cleanup"))

import matchpaths  # noqa: E402
from lib.match import Match  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Eventos por reglas -> CSV Metrica")
    ap.add_argument("--match", default=None,
                    help="nombre del partido (convencion "
                         "~/football_data/matches/<match>/tracking.csv)")
    ap.add_argument("--tracking", default=None,
                    help="CSV crudo de tracking (override de --match)")
    ap.add_argument("--out", default=None,
                    help="CSV de propuestas de salida (override de --match)")
    args = ap.parse_args()

    if not args.match and not (args.tracking and args.out):
        ap.error("pasa --match <partido> o --tracking y --out")

    tracking = os.path.expanduser(
        args.tracking or matchpaths.tracking_path(args.match))
    out = os.path.expanduser(args.out or matchpaths.proposed_path(args.match))

    if not os.path.exists(tracking):
        ap.error(f"no existe el tracking: {tracking}")

    match = Match()
    match.import_raw_data(os.path.dirname(tracking) + os.sep,
                          os.path.basename(tracking))
    log = match.generate_events()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    path = log.export(path=os.path.dirname(out) or ".",
                      file_name=os.path.basename(out))
    print(f"{len(log)} eventos {dict(log.summary())} -> {path}")


if __name__ == "__main__":
    main()

"""
Genera el CSV de eventos propuestos por las reglas para un tracking CSV.
Es el paso previo al etiquetado: estas propuestas son las que se corrigen
en label_tool.py.

Uso:
    python3 events_model/propose_events.py \
        --tracking "~/Downloads/psg_bayern_720p (2) (2).csv" \
        --out events_model/dataset/psg_bayern_events.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_cleanup"))

from lib.match import Match  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Eventos por reglas -> CSV Metrica")
    ap.add_argument("--tracking", required=True, help="CSV crudo de tracking")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tracking = os.path.expanduser(args.tracking)
    match = Match()
    match.import_raw_data(os.path.dirname(tracking) + os.sep,
                          os.path.basename(tracking))
    log = match.generate_events()

    out = os.path.expanduser(args.out)
    path = log.export(path=os.path.dirname(out) or ".",
                      file_name=os.path.basename(out))
    print(f"{len(log)} eventos {dict(log.summary())} -> {path}")


if __name__ == "__main__":
    main()
